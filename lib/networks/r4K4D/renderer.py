import os.path
import torch
import ctypes
import OpenGL.GL as gl
from OpenGL.GL import shaders
from enum import Enum
import glfw
import glm
from glm import vec2, vec3, vec4, mat3, mat4, mat4x3, mat2x3  # This is actually highly optimized
import nerfacc
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import PerspectiveCameras, PointsRasterizer, AlphaCompositor
from pytorch3d.renderer.points.rasterizer import rasterize_points

from lib.utils.base_utils import *
from lib.utils.vis_utils import *

def FORMAT_CUDART_ERROR(err):
    from cuda import cudart
    return (
        f"{cudart.cudaGetErrorName(err)[1].decode('utf-8')}({int(err)}): "
        f"{cudart.cudaGetErrorString(err)[1].decode('utf-8')}"
    )

def CHECK_CUDART_ERROR(args):
    from cuda import cudart

    if isinstance(args, tuple):
        assert len(args) >= 1
        err = args[0]
        if len(args) == 1:
            ret = None
        elif len(args) == 2:
            ret = args[1]
        else:
            ret = args[1:]
    else:
        err = args
        ret = None

    assert isinstance(err, cudart.cudaError_t), type(err)
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(FORMAT_CUDART_ERROR(err))

    return ret

def get_bounds(xyz, padding=0.05):  # 5mm padding? really?
    # xyz: n_batch, n_points, 3

    min_xyz = torch.min(xyz, dim=1)[0]  # torch min with dim is ...
    max_xyz = torch.max(xyz, dim=1)[0]
    min_xyz -= padding
    max_xyz += padding
    bounds = torch.stack([min_xyz, max_xyz], dim=1)
    return bounds
    diagonal = bounds[..., 1:] - bounds[..., :1]  # n_batch, 1, 3
    bounds[..., 1:] = bounds[..., :1] + torch.ceil(diagonal / voxel_size) * voxel_size  # n_batch, 1, 3
    return bounds


def hardware_rendering_framebuffer(H: int, W: int, gl_tex_dtype=gl.GL_RGBA16F):
    # Prepare for write frame buffers
    color_buffer = gl.glGenTextures(1)
    depth_upper = gl.glGenTextures(1)
    depth_lower = gl.glGenTextures(1)
    depth_attach = gl.glGenTextures(1)
    fbo = gl.glGenFramebuffers(1)  # generate 1 framebuffer, storereference in fb

    # Init the texture (call the resizing function), will simply allocate empty memory
    # The internal format describes how the texture shall be stored in the GPU. The format describes how the format of your pixel data in client memory (together with the type parameter).
    gl.glBindTexture(gl.GL_TEXTURE_2D, color_buffer)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl_tex_dtype, W, H, 0, gl.GL_RGBA, gl.GL_FLOAT, ctypes.c_void_p(0))  # 16 * 4
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)

    gl.glBindTexture(gl.GL_TEXTURE_2D, depth_upper)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_R32F, W, H, 0, gl.GL_RED, gl.GL_FLOAT, ctypes.c_void_p(0))  # 32
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)

    gl.glBindTexture(gl.GL_TEXTURE_2D, depth_lower)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_R32F, W, H, 0, gl.GL_RED, gl.GL_FLOAT, ctypes.c_void_p(0))  # 32
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)

    gl.glBindTexture(gl.GL_TEXTURE_2D, depth_attach)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_DEPTH_COMPONENT24, W, H, 0, gl.GL_DEPTH_COMPONENT, gl.GL_UNSIGNED_INT, ctypes.c_void_p(0))  # 32
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)

    # Bind texture to fbo
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
    gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, color_buffer, 0)  # location 0
    gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT1, gl.GL_TEXTURE_2D, depth_upper, 0)  # location 1
    gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT2, gl.GL_TEXTURE_2D, depth_lower, 0)  # location 1
    gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_TEXTURE_2D, depth_attach, 0)
    gl.glDrawBuffers(3, [gl.GL_COLOR_ATTACHMENT0, gl.GL_COLOR_ATTACHMENT1, gl.GL_COLOR_ATTACHMENT2])

    # Check framebuffer status
    if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
        # log(red('Framebuffer not complete, exiting...'))
        raise RuntimeError('Incomplete framebuffer')

    # Restore the original state
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    return color_buffer, depth_upper, depth_lower, depth_attach, fbo


def use_gl_program(program: Union[shaders.ShaderProgram, dict]):
    if isinstance(program, dict):
        # Recompile the program if the user supplied sources
        program = dotdict(program)
        program = shaders.compileProgram(
            shaders.compileShader(program.VERT_SHADER_SRC, gl.GL_VERTEX_SHADER),
            shaders.compileShader(program.FRAG_SHADER_SRC, gl.GL_FRAGMENT_SHADER)
        )
    return gl.glUseProgram(program)

def load_shader_source(file: str = 'splat.frag'):
    # Ideally we can just specify the shader name instead of an variable
    if not os.path.exists(file):
        file = f'{os.path.dirname(__file__)}/shaders/{file}'
    if not os.path.exists(file):
        file = file.replace('shaders/', '')
    if not os.path.exists(file):
        raise RuntimeError(f'Shader file: {file} does not exist')
    with open(file, 'r') as f:
        return f.read()


def load_mesh(filename: str, device='cuda', load_uv=False, load_aux=False, backend='pytorch3d'):
    from pytorch3d.io import load_ply, load_obj
    if backend == 'trimesh':
        import trimesh
        mesh: trimesh.Trimesh = trimesh.load(filename)
        return mesh.vertices, mesh.faces

    vm, fm = None, None
    if filename.endswith('.npz'):
        mesh = np.load(filename)
        v = torch.from_numpy(mesh['verts'])
        f = torch.from_numpy(mesh['faces'])

        if load_uv:
            vm = torch.from_numpy(mesh['uvs'])
            fm = torch.from_numpy(mesh['uvfaces'])
    else:
        if filename.endswith('.ply'):
            v, f = load_ply(filename)
        elif filename.endswith('.obj'):
            v, faces_attr, aux = load_obj(filename)
            f = faces_attr.verts_idx

            if load_uv:
                vm = aux.verts_uvs
                fm = faces_attr.textures_idx
        else:
            raise NotImplementedError(f'Unrecognized input format for: {filename}')

    v = v.to(device, non_blocking=True).contiguous()
    f = f.to(device, non_blocking=True).contiguous()

    if load_uv:
        vm = vm.to(device, non_blocking=True).contiguous()
        fm = fm.to(device, non_blocking=True).contiguous()

    if load_uv:
        if load_aux:
            return v, f, vm, fm, aux
        else:
            return v, f, vm, fm
    else:
        return v, f


def load_pts(filename: str):
    from pyntcloud import PyntCloud
    cloud = PyntCloud.from_file(filename)
    verts = cloud.xyz
    if 'red' in cloud.points and 'green' in cloud.points and 'blue' in cloud.points:
        r = np.asarray(cloud.points['red'])
        g = np.asarray(cloud.points['green'])
        b = np.asarray(cloud.points['blue'])
        colors = (np.stack([r, g, b], axis=-1) / 255).astype(np.float32)
    elif 'r' in cloud.points and 'g' in cloud.points and 'b' in cloud.points:
        r = np.asarray(cloud.points['r'])
        g = np.asarray(cloud.points['g'])
        b = np.asarray(cloud.points['b'])
        colors = (np.stack([r, g, b], axis=-1) / 255).astype(np.float32)
    else:
        colors = None

    if 'nx' in cloud.points and 'ny' in cloud.points and 'nz' in cloud.points:
        nx = np.asarray(cloud.points['nx'])
        ny = np.asarray(cloud.points['ny'])
        nz = np.asarray(cloud.points['nz'])
        norms = np.stack([nx, ny, nz], axis=-1)
    else:
        norms = None

    # if 'alpha' in cloud.points:
    #     cloud.points['alpha'] = cloud.points['alpha'] / 255

    reserved = ['x', 'y', 'z', 'red', 'green', 'blue', 'r', 'g', 'b', 'nx', 'ny', 'nz']
    scalars = dotdict({k: np.asarray(cloud.points[k])[..., None] for k in cloud.points if k not in reserved})  # one extra dimension at the back added
    return verts, colors, norms, scalars

class Mesh:
    class RenderType(Enum):
        POINTS = 1
        LINES = 2
        TRIS = 3
        QUADS = 4
        STRIPS = 5

    # Helper class to render a mesh on opengl
    # This implementation should only be used for debug visualization
    # Since no differentiable mechanism will be added
    # We recommend using nvdiffrast and pytorch3d's point renderer directly if you will to optimize these structures directly

    def __init__(self,
                 verts: torch.Tensor = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 1]]),  # need to call update after update
                 faces: torch.Tensor = torch.tensor([[0, 1, 2]]),  # need to call update after update
                 colors: torch.Tensor = None,
                 normals: torch.Tensor = None,
                 scalars: dotdict[str, torch.Tensor] = dotdict(),
                 render_type: RenderType = RenderType.TRIS,

                 # Misc info
                 name: str = 'mesh',
                 filename: str = '',
                 visible: bool = True,

                 # Render options
                 shade_flat: bool = False,  # smooth shading
                 point_radius: float = 0.015,
                 render_normal: bool = False,

                 # Storage options
                 store_device: str = 'cpu',
                 compute_device: str = 'cuda',
                 vert_sizes=[3, 3, 3],  # pos + color + norm

                 # Init options
                 est_normal_thresh: int = 100000,

                 # Ignore unused input
                 **kwargs,
                 ) -> None:
        super().__init__()
        self.name = name
        self.visible = visible
        self.render_type = render_type

        self.shade_flat = shade_flat
        self.point_radius = point_radius
        self.render_normal = render_normal

        self.store_device = store_device
        self.compute_device = compute_device
        self.vert_sizes = vert_sizes

        self.est_normal_thresh = est_normal_thresh

        # Uniform and program
        self.compile_shaders()
        self.uniforms = dotdict()  # uniform values

        # Before initialization
        self.max_verts = 0
        self.max_faces = 0

        # OpenGL data
        if filename:
            self.load_from_file(filename)
        else:
            self.load_from_data(verts, faces, colors, normals, scalars)




    def compile_shaders(self):
        try:
            self.mesh_program = shaders.compileProgram(
                shaders.compileShader(load_shader_source('mesh.vert'), gl.GL_VERTEX_SHADER),
                shaders.compileShader(load_shader_source('mesh.frag'), gl.GL_FRAGMENT_SHADER)
            )
            self.point_program = shaders.compileProgram(
                shaders.compileShader(load_shader_source('point.vert'), gl.GL_VERTEX_SHADER),
                shaders.compileShader(load_shader_source('point.frag'), gl.GL_FRAGMENT_SHADER), validate=False
            )
        except Exception as e:
            print(str(e).encode('utf-8').decode('unicode_escape'))
            raise e

    @property
    def n_verts_bytes(self):
        return len(self.verts) * self.vert_size * self.verts.element_size()

    @property
    def n_faces_bytes(self):
        return len(self.faces) * self.face_size * self.faces.element_size()

    @property
    def verts_data(self):  # a heavy copy operation
        verts = torch.cat([self.verts, self.colors, self.normals], dim=-1).ravel().numpy()  # MARK: Maybe sync
        verts = np.asarray(verts, dtype=np.float32, order='C')
        return verts

    @property
    def faces_data(self):  # a heavy copy operation
        faces = self.faces.ravel().numpy()  # N, 3
        faces = np.asarray(faces, dtype=np.uint32, order='C')
        return faces

    @property
    def face_size(self):
        return self.render_type.value

    @property
    def vert_size(self):
        return sum(self.vert_sizes)

    def load_from_file(self, filename: str = 'assets/meshes/bunny.ply'):
        verts, faces, colors, normals, scalars = self.load_data_from_file(filename)
        self.load_from_data(verts, faces, colors, normals, scalars)

    def load_data_from_file(self, filename: str = 'assets/meshes/bunny.ply'):
        self.name = os.path.split(filename)[-1]
        verts, faces, colors, normals, scalars = None, None, None, None, None
        verts, faces = load_mesh(filename, device=self.store_device)
        if not len(faces):
            verts, colors, normals, scalars = load_pts(filename)
            self.render_type = Mesh.RenderType.POINTS
        else:
            self.render_type = Mesh.RenderType(faces.shape[-1])  # use value
        return verts, faces, colors, normals, scalars

    def load_from_data(self, verts: torch.Tensor, faces: torch.Tensor, colors: torch.Tensor = None, normals: torch.Tensor = None, scalars: dotdict[str, torch.Tensor] = dotdict()):
        # Data type conversion
        verts = torch.as_tensor(verts)  # convert to tensor if input is of other types
        if verts.dtype == torch.float32:
            pass  # supports this for now
        elif verts.dtype == torch.float16:
            pass  # supports this for now
        else:
            verts = verts.type(torch.float)  # convert to float32 if input is of higher precision
        gl_dtype = gl.GL_FLOAT if verts.dtype == torch.float else gl.GL_HALF_FLOAT
        self.vert_gl_types = [gl_dtype] * len(self.vert_sizes)

        # Prepare main mesh data: vertices and faces
        self.verts = torch.as_tensor(verts, device=self.store_device)
        self.faces = torch.as_tensor(faces, device=self.store_device, dtype=torch.int32)  # NOTE: No uint32 support

        # Prepare colors and normals
        if colors is not None:
            self.colors = torch.as_tensor(colors, device=self.store_device, dtype=self.verts.dtype)
        else:
            bounds = get_bounds(self.verts[None])[0]
            self.colors = (self.verts - bounds[0]) / (bounds[1] - bounds[0])
        if normals is not None:
            self.normals = torch.as_tensor(normals, device=self.store_device, dtype=self.verts.dtype)
        else:
            self.estimate_vertex_normals()

        # Prepare other scalars
        if scalars is not None:
            for k, v in scalars.items():
                setattr(self, k, torch.as_tensor(v, device=self.store_device, dtype=self.verts.dtype))  # is this ok?

        # Prepare OpenGL related buffer
        self.update_gl_buffers()

    def estimate_vertex_normals(self):
        def est_pcd_norms():
            if self.verts.dtype == torch.half:
                self.normals = self.verts
            else:
                from pytorch3d.structures import Pointclouds, Meshes
                pcd = Pointclouds([self.verts]).to(self.compute_device)
                self.normals = pcd.estimate_normals()[0].cpu().to(self.verts.dtype)  # no batch dim

        def est_tri_norms():
            if self.verts.dtype == torch.half:
                self.normals = self.verts
            else:
                from pytorch3d.structures import Pointclouds, Meshes
                mesh = Meshes([self.verts], [self.faces]).to(self.compute_device)
                self.normals = mesh.verts_normals_packed().cpu().to(self.verts.dtype)  # no batch dim

        if not len(self.verts) > self.est_normal_thresh:
            if self.render_type == Mesh.RenderType.TRIS: est_tri_norms()
            elif self.render_type == Mesh.RenderType.POINTS: est_pcd_norms()
            else:
                # log(yellow(f'Unsupported mesh type: {self.render_type} for normal estimation, skipping'))
                self.normals = self.verts
        else:
            # log(yellow(f'Number of points for mesh too large: {len(self.verts)} > {self.est_normal_thresh}, skipping normal estimation'))
            self.normals = self.verts

    # def offscreen_render(self, eglctx: "eglContextManager", camera: Camera):
    #     eglctx.resize(camera.W, camera.H)
    #     self.render(camera)

    # def render(self, camera: Camera):
    #     if not self.visible: return
    #
    #     # For point rendering
    #     if self.render_type == Mesh.RenderType.POINTS:
    #         gl.glUseProgram(self.point_program)
    #         self.use_gl_program(self.point_program)
    #     else:
    #         gl.glUseProgram(self.mesh_program)
    #         self.use_gl_program(self.mesh_program)
    #
    #     self.upload_gl_uniforms(camera)
    #     gl.glBindVertexArray(self.vao)
    #
    #     if self.render_type == Mesh.RenderType.POINTS:
    #         gl.glDrawArrays(gl.GL_POINTS, 0, len(self.verts))  # number of vertices
    #     elif self.render_type == Mesh.RenderType.LINES:
    #         gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
    #         gl.glDrawElements(gl.GL_LINES, len(self.faces) * self.face_size, gl.GL_UNSIGNED_INT, ctypes.c_void_p(0))  # number of indices
    #     elif self.render_type == Mesh.RenderType.TRIS:
    #         gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
    #         gl.glDrawElements(gl.GL_TRIANGLES, len(self.faces) * self.face_size, gl.GL_UNSIGNED_INT, ctypes.c_void_p(0))  # number of indices
    #     elif self.render_type == Mesh.RenderType.QUADS:
    #         gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
    #         gl.glDrawElements(gl.GL_QUADS, len(self.faces) * self.face_size, gl.GL_UNSIGNED_INT, ctypes.c_void_p(0))  # number of indices
    #     elif self.render_type == Mesh.RenderType.STRIPS:
    #         gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, len(self.verts))
    #     else:
    #         raise NotImplementedError
    #
    #     gl.glBindVertexArray(0)

    def use_gl_program(self, program: shaders.ShaderProgram):
        use_gl_program(program)
        self.uniforms.shade_flat = gl.glGetUniformLocation(program, "shade_flat")
        self.uniforms.point_radius = gl.glGetUniformLocation(program, "point_radius")
        self.uniforms.render_normal = gl.glGetUniformLocation(program, "render_normal")
        self.uniforms.H = gl.glGetUniformLocation(program, "H")
        self.uniforms.W = gl.glGetUniformLocation(program, "W")
        self.uniforms.n = gl.glGetUniformLocation(program, "n")
        self.uniforms.f = gl.glGetUniformLocation(program, "f")
        self.uniforms.P = gl.glGetUniformLocation(program, "P")
        self.uniforms.K = gl.glGetUniformLocation(program, "K")
        self.uniforms.V = gl.glGetUniformLocation(program, "V")
        self.uniforms.M = gl.glGetUniformLocation(program, "M")

    def upload_gl_uniforms(self, camera: Camera):
        K = camera.gl_ixt  # hold the reference
        V = camera.gl_ext  # hold the reference
        M = glm.identity(mat4)
        P = K * V * M

        gl.glUniform1i(self.uniforms.shade_flat, self.shade_flat)
        gl.glUniform1f(self.uniforms.point_radius, self.point_radius)
        gl.glUniform1i(self.uniforms.render_normal, self.render_normal)
        gl.glUniform1i(self.uniforms.H, camera.H)  # o2w
        gl.glUniform1i(self.uniforms.W, camera.W)  # o2w
        gl.glUniform1f(self.uniforms.n, camera.n)  # o2w
        gl.glUniform1f(self.uniforms.f, camera.f)  # o2w
        gl.glUniformMatrix4fv(self.uniforms.P, 1, gl.GL_FALSE, glm.value_ptr(P))  # o2clip
        gl.glUniformMatrix4fv(self.uniforms.K, 1, gl.GL_FALSE, glm.value_ptr(K))  # c2clip
        gl.glUniformMatrix4fv(self.uniforms.V, 1, gl.GL_FALSE, glm.value_ptr(V))  # w2c
        gl.glUniformMatrix4fv(self.uniforms.M, 1, gl.GL_FALSE, glm.value_ptr(M))  # o2w

    def update_gl_buffers(self):
        # Might be overwritten
        self.resize_buffers(len(self.verts) if hasattr(self, 'verts') else 0,
                            len(self.faces) if hasattr(self, 'faces') else 0)  # maybe repeated

        if hasattr(self, 'verts'):
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
            gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, self.n_verts_bytes, self.verts_data)  # hold the reference
        if hasattr(self, 'faces'):
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            gl.glBufferSubData(gl.GL_ELEMENT_ARRAY_BUFFER, 0, self.n_faces_bytes, self.faces_data)

    def resize_buffers(self, v: int = 0, f: int = 0):
        if v > self.max_verts or f > self.max_faces:
            if v > self.max_verts: self.max_verts = v
            if f > self.max_faces: self.max_faces = f
            self.init_gl_buffers(v, f)

    def init_gl_buffers(self, v: int = 0, f: int = 0):
        # This will only init the corresponding buffer object
        n_verts_bytes = v * self.vert_size * self.verts.element_size() if v > 0 else self.n_verts_bytes
        n_faces_bytes = f * self.face_size * self.faces.element_size() if f > 0 else self.n_faces_bytes

        # Housekeeping
        if hasattr(self, 'vao'):
            gl.glDeleteVertexArrays(1, [self.vao])
            gl.glDeleteBuffers(2, [self.vbo, self.ebo])

        self.vao = gl.glGenVertexArrays(1)
        self.vbo = gl.glGenBuffers(1)
        self.ebo = gl.glGenBuffers(1)

        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, n_verts_bytes, ctypes.c_void_p(0), gl.GL_DYNAMIC_DRAW)  # NOTE: Using pointers here won't work

        # https://stackoverflow.com/questions/67195932/pyopengl-cannot-render-any-vao
        cumsum = 0
        for i, (s, t) in enumerate(zip(self.vert_sizes, self.vert_gl_types)):
            # FIXME: OpenGL.error.Error: Attempt to retrieve context when no valid context
            gl.glVertexAttribPointer(i, s, t, gl.GL_FALSE, self.vert_size * self.verts.element_size(), ctypes.c_void_p(cumsum * self.verts.element_size()))  # we use 32 bit float
            gl.glEnableVertexAttribArray(i)
            cumsum += s

        if n_faces_bytes > 0:
            # Some implementation has no faces, we dangerously ignore ebo here, assuming they will never be used
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, n_faces_bytes, ctypes.c_void_p(0), gl.GL_DYNAMIC_DRAW)
            gl.glBindVertexArray(0)

    # def render_imgui(mesh, viewer: 'VolumetricVideoViewer', batch: dotdict):
    #     from imgui_bundle import imgui
    #     from easyvolcap.utils.imgui_utils import push_button_color, pop_button_color
    #
    #     i = batch.i
    #     will_delete = batch.will_delete
    #     slider_width = batch.slider_width
    #
    #     imgui.push_item_width(slider_width * 0.5)
    #     mesh.name = imgui.input_text(f'Mesh name##{i}', mesh.name)[1]
    #
    #     if imgui.begin_combo(f'Mesh type##{i}', mesh.render_type.name):
    #         for t in Mesh.RenderType:
    #             if imgui.selectable(t.name, mesh.render_type == t)[1]:
    #                 mesh.render_type = t  # construct enum from name
    #             if mesh.render_type == t:
    #                 imgui.set_item_default_focus()
    #         imgui.end_combo()
    #     imgui.pop_item_width()
    #
    #     if hasattr(mesh, 'point_radius'):
    #         mesh.point_radius = imgui.slider_float(f'Point radius##{i}', mesh.point_radius, 0.0005, 3.0)[1]  # 0.1mm
    #
    #     if hasattr(mesh, 'pts_per_pix'):
    #         mesh.pts_per_pix = imgui.slider_int('Point per pixel', mesh.pts_per_pix, 0, 60)[1]  # 0.1mm
    #
    #     if hasattr(mesh, 'shade_flat'):
    #         push_button_color(0x55cc33ff if not mesh.shade_flat else 0x8855aaff)
    #         if imgui.button(f'Smooth##{i}' if not mesh.shade_flat else f' Flat ##{i}'):
    #             mesh.shade_flat = not mesh.shade_flat
    #         pop_button_color()
    #
    #     if hasattr(mesh, 'render_normal'):
    #         imgui.same_line()
    #         push_button_color(0x55cc33ff if not mesh.render_normal else 0x8855aaff)
    #         if imgui.button(f'Color ##{i}' if not mesh.render_normal else f'Normal##{i}'):
    #             mesh.render_normal = not mesh.render_normal
    #         pop_button_color()
    #
    #     if hasattr(mesh, 'visible'):
    #         imgui.same_line()
    #         push_button_color(0x55cc33ff if not mesh.visible else 0x8855aaff)
    #         if imgui.button(f'Show##{i}' if not mesh.visible else f'Hide##{i}'):
    #             mesh.visible = not mesh.visible
    #         pop_button_color()
    #
    #     # Render the delete button
    #     imgui.same_line()
    #     push_button_color(0xff5533ff)
    #     if imgui.button(f'Delete##{i}'):
    #         will_delete.append(i)
    #     pop_button_color()

class Splat(Mesh):  # FIXME: Not rendering, need to debug this
    def __init__(self,
                 *args,
                 H: int = 512,
                 W: int = 512,
                 tex_dtype: str = torch.half,

                 pts_per_pix: int = 24,  # render less for the static background since we're only doing a demo
                 blit_last_ratio: float = 0.0,
                 volume_rendering: bool = True,
                 radii_mult_volume: float = 1.00,  # 2 / 3 is the right integration, but will leave holes, 1.0 will make it bloat, 0.85 looks visually better
                 radii_mult_solid: float = 0.85,  # 2 / 3 is the right integration, but will leave holes, 1.0 will make it bloat, 0.85 looks visually better

                 point_smooth: bool = True,
                 alpha_blending: bool = True,
                 **kwargs):
        kwargs = dotdict(kwargs)
        kwargs.vert_sizes = kwargs.get('vert_sizes', [3, 3, 1, 1])
        self.tex_dtype = getattr(torch, tex_dtype) if isinstance(tex_dtype, str) else tex_dtype
        self.gl_tex_dtype = gl.GL_RGBA16F if self.tex_dtype == torch.half else gl.GL_RGBA32F

        super().__init__(*args, **kwargs)
        self.use_gl_program(self.splat_program)

        self.pts_per_pix = pts_per_pix
        self.blit_last_ratio = blit_last_ratio
        self.volume_rendering = volume_rendering
        self.radii_mult_volume = radii_mult_volume
        self.radii_mult_solid = radii_mult_solid

        self.point_smooth = point_smooth
        self.alpha_blending = alpha_blending

        self.max_H, self.max_W = H, W
        self.H, self.W = H, W
        self.init_textures()

        # from easyvolcap.models.samplers.gaussiant_sampler import GaussianTSampler
        # self.render_radius = MethodType(GaussianTSampler.render_radius, self)  # override the method

    # @property
    # def verts_data(self):  # a heavy copy operation
    #     verts = torch.cat([self.verts, self.colors, self.radius, self.alpha], dim=-1).ravel().numpy()  # MARK: Maybe sync
    #     verts = np.asarray(verts, dtype=np.float32, order='C')  # this should only be invoked once
    #     return verts

    def use_gl_program(self, program: shaders.ShaderProgram):
        super().use_gl_program(program)
        # Special controlling variables
        self.uniforms.alpha_blending = gl.glGetUniformLocation(program, f'alpha_blending')
        self.uniforms.point_smooth = gl.glGetUniformLocation(program, f'point_smooth')
        self.uniforms.radii_mult = gl.glGetUniformLocation(program, f'radii_mult')

        # Special rendering variables
        self.uniforms.pass_index = gl.glGetUniformLocation(program, f'pass_index')
        self.uniforms.read_color = gl.glGetUniformLocation(program, f'read_color')
        self.uniforms.read_upper = gl.glGetUniformLocation(program, f'read_upper')
        self.uniforms.read_lower = gl.glGetUniformLocation(program, f'read_lower')
        gl.glUniform1i(self.uniforms.read_color, 0)
        gl.glUniform1i(self.uniforms.read_upper, 1)
        gl.glUniform1i(self.uniforms.read_lower, 2)

    # def compile_shaders(self):
    #     try:
    #         self.splat_program = shaders.compileProgram(
    #             shaders.compileShader(load_shader_source('splat.vert'), gl.GL_VERTEX_SHADER),
    #             shaders.compileShader(load_shader_source('splat.frag'), gl.GL_FRAGMENT_SHADER)
    #         )
    #         self.usplat_program = shaders.compileProgram(
    #             shaders.compileShader(load_shader_source('usplat.vert'), gl.GL_VERTEX_SHADER),
    #             shaders.compileShader(load_shader_source('usplat.frag'), gl.GL_FRAGMENT_SHADER)
    #         )
    #     except Exception as e:
    #         print(str(e).encode('utf-8').decode('unicode_escape'))
    #         raise e

    def rasterize(self, camera: Camera = None, length: int = None):
        if self.volume_rendering:
            return self.rasterize_volume(camera, length)
        else:
            return self.rasterize_solid(camera, length)

    def rasterize_volume(self, camera: Camera = None, length: int = None):  # some implementation requires no uploading of camera
        """
        Let's try to analyze what's happening here

        We want to:
            1. Render the front-most color to color buffer
            2. UNUSED: Render the front-most depth + some large margin to a depth upper limit buffer
            3. Render the front-most depth + some small margin to a depth lower limit buffer
            4. Switch between the render target and sampling target
            5. Use the previous rendered color, depth upper limit and lower limit as textures
            6. When current depth is smaller than the lower limit, we've already rendered this in the first pass, discard
            7. UNUSED: When current depth is larger than the upper limit, it will probabily not contribute much to final results, discard
            8. UNUSED: When the accumulated opacity reaches almost 1, subsequent rendering would not have much effect, return directly
            9. When the point coordinates falls out of bound of the current sphere, dicard (this could be optimized with finutining in rectangle)
            10. Finally, try to render the final color using the volume rendering equation (by accumulating alpha values from front to back)

        Required cleanup checklist:
            1. Before rendering the first pass, we need to clear the color and depth texture, this is not done, need to check multi-frame accumulation on this
            2. Before rendering next pass, it's also recommended to blit color and depth values from previous pass to avoid assign them in the shader
        """

        front_fbo, front_color, front_upper, front_lower = self.read_fbo, self.read_color, self.read_upper, self.read_lower
        back_fbo, back_color, back_upper, back_lower = self.write_fbo, self.write_color, self.write_upper, self.write_lower

        # Only clear the output once
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, front_fbo)  # for offscreen rendering to textures
        gl.glClearBufferfv(gl.GL_COLOR, 0, [0.0, 0.0, 0.0, 0.0])
        # gl.glClearBufferfv(gl.GL_COLOR, 1, [1e9])
        gl.glClearBufferfv(gl.GL_COLOR, 2, [0.0, 0.0, 0.0, 0.0])
        gl.glClearBufferfv(gl.GL_DEPTH, 0, [1e9])  # this is for depth testing

        # Only clear the output once
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, back_fbo)  # for offscreen rendering to textures
        gl.glClearBufferfv(gl.GL_COLOR, 0, [0.0, 0.0, 0.0, 0.0])
        # gl.glClearBufferfv(gl.GL_COLOR, 1, [1e9])
        gl.glClearBufferfv(gl.GL_COLOR, 2, [0.0, 0.0, 0.0, 0.0])
        gl.glClearBufferfv(gl.GL_DEPTH, 0, [1e9])  # this is for depth testing

        # Prepare for the actual rendering, previous operations could rebind the vertex array
        self.use_gl_program(self.splat_program)  # TODO: Implement this with a mapping and a lazy modification
        self.upload_gl_uniforms(camera)
        gl.glBindVertexArray(self.vao)

        # The actual multi pass rendering process happens here
        for pass_index in range(self.pts_per_pix):
            # Swap buffers to render the next pass
            front_fbo, front_color, front_upper, front_lower, back_fbo, back_color, back_upper, back_lower = \
                back_fbo, back_color, back_upper, back_lower, front_fbo, front_color, front_upper, front_lower

            # Bind the read texture and bind the write render frame buffer
            gl.glBindTextures(0, 3, [front_color, front_upper, front_lower])

            # Move content from write_fbo to screen fbo
            if pass_index > self.pts_per_pix * self.blit_last_ratio:  # no blitting almost has no effect on the rendering
                gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, front_fbo)
                gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, back_fbo)
                for i in range(3):
                    gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0 + i)
                    gl.glDrawBuffer(gl.GL_COLOR_ATTACHMENT0 + i)
                    gl.glBlitFramebuffer(0, 0, self.W, self.H, 0, 0, self.W, self.H, gl.GL_COLOR_BUFFER_BIT, gl.GL_NEAREST)
                gl.glDrawBuffers(3, [gl.GL_COLOR_ATTACHMENT0, gl.GL_COLOR_ATTACHMENT1, gl.GL_COLOR_ATTACHMENT2])

            # Clear depth buffer for depth testing
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, back_fbo)  # for offscreen rendering to textures
            gl.glClearBufferfv(gl.GL_DEPTH, 0, [1e9])  # this is for depth testing
            gl.glUniform1i(self.uniforms.pass_index, pass_index)  # pass index

            # The actual drawing pass with render things out to the write_fbo
            gl.glDrawArrays(gl.GL_POINTS, 0, length if length is not None else len(self.verts))  # number of vertices

        # Restore states of things
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glBindVertexArray(0)
        return back_fbo

    def upload_gl_uniforms(self, camera: Camera):
        super().upload_gl_uniforms(camera)
        gl.glUniform1i(self.uniforms.point_smooth, self.point_smooth)
        gl.glUniform1i(self.uniforms.alpha_blending, self.alpha_blending)

        if self.volume_rendering:
            gl.glUniform1f(self.uniforms.radii_mult, self.radii_mult_volume)  # radii mult
        else:
            gl.glUniform1f(self.uniforms.radii_mult, self.radii_mult_solid)  # radii mult

    def rasterize_solid(self, camera: Camera = None, length: int = None):
        # Only clear the output once
        back_fbo = self.write_fbo
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, back_fbo)  # for offscreen rendering to textures
        gl.glClearBufferfv(gl.GL_COLOR, 0, [0.0, 0.0, 0.0, 0.0])  # color
        # gl.glClearBufferfv(gl.GL_COLOR, 1, [0.0]) # depth upper
        gl.glClearBufferfv(gl.GL_COLOR, 2, [0.0, 0.0, 0.0, 0.0])  # depth lower
        gl.glClearBufferfv(gl.GL_DEPTH, 0, [1e9])  # this is for depth testing

        # Prepare for the actual rendering, previous operations could rebind the vertex array
        self.use_gl_program(self.usplat_program)
        self.upload_gl_uniforms(camera)
        gl.glUniform1i(self.uniforms.pass_index, 0)  # pass index
        gl.glBindVertexArray(self.vao)

        # The actual drawing pass with render things out to the write_fbo
        gl.glDrawArrays(gl.GL_POINTS, 0, length if length is not None else len(self.verts))  # number of vertices

        # Restore states of things
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glBindVertexArray(0)
        return back_fbo

    # def show(self, back_fbo: int):
    #     # Move content from write_fbo to screen fbo
    #     gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, back_fbo)
    #     gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, 0)  # render the final content onto screen
    #     gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
    #     gl.glBlitFramebuffer(0, 0, self.W, self.H, 0, 0, self.W, self.H, gl.GL_COLOR_BUFFER_BIT, gl.GL_NEAREST)
    #     gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    # def render(self, camera):
    #     if not self.visible: return
    #     self.show(self.rasterize(camera))

    def resize_textures(self, H: int, W: int):  # analogy to update_gl_buffers
        self.H, self.W = H, W
        if self.H > self.max_H or self.W > self.max_W:  # max got updated
            self.max_H, self.max_W = max(int(self.H * 1.05), self.max_H), max(int(self.W * 1.05), self.max_W)
            self.init_textures()

    def init_textures(self):
        if hasattr(self, 'write_fbo'):
            gl.glDeleteFramebuffers(2, [self.write_fbo, self.read_fbo])
            gl.glDeleteTextures(8, [self.write_color, self.write_upper, self.write_lower, self.write_attach, self.read_color, self.read_upper, self.read_lower, self.read_attach])
        self.write_color, self.write_upper, self.write_lower, self.write_attach, self.write_fbo = hardware_rendering_framebuffer(self.max_H, self.max_W, self.gl_tex_dtype)
        self.read_color, self.read_upper, self.read_lower, self.read_attach, self.read_fbo = hardware_rendering_framebuffer(self.max_H, self.max_W, self.gl_tex_dtype)
        # log(f'Created texture of h, w: {self.max_H}, {self.max_W}')

class openglRenderer(Splat):
    def __init__(self,
                 dtype=torch.half,
                 **kwargs,
                 ):
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.gl_dtype = gl.GL_HALF_FLOAT if self.dtype == torch.half else gl.GL_FLOAT
        kwargs = dotdict(kwargs)
        kwargs.blit_last_ratio = kwargs.get('blit_last_ratio', 0.90)
        kwargs.vert_sizes = kwargs.get('vert_sizes', [3, 3, 1, 1])
        super().__init__(**kwargs)  # verts, color, radius, alpha

    # @property
    # def verts_data(self):  # a heavy copy operation
    #     verts = torch.cat([self.verts, self.colors, self.radius, self.alpha], dim=-1).ravel().numpy()  # MARK: Maybe sync
    #     verts = np.asarray(verts, dtype=torch_dtype_to_numpy_dtype(self.dtype), order='C')  # this should only be invoked once
    #     return verts

    # def init_gl_buffers(self, v: int = 0, f: int = 0):
    #     from cuda import cudart
    #     if hasattr(self, 'cu_vbo'):
    #         CHECK_CUDART_ERROR(cudart.cudaGraphicsUnregisterResource(self.cu_vbo))
    #
    #     super().init_gl_buffers(v, f)
    #
    #     # Register vertex buffer obejct
    #     flags = cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard
    #     try:
    #         self.cu_vbo = CHECK_CUDART_ERROR(cudart.cudaGraphicsGLRegisterBuffer(self.vbo, flags))
    #     except RuntimeError as e:
    #         # log(red(f'Your system does not support CUDA-GL interop, will use pytorch3d\'s implementation instead'))
    #         # log(red(f'This can be done by specifying {blue("model_cfg.sampler_cfg.use_cudagl=False model_cfg.sampler_cfg.use_diffgl=False")} at the end of your command'))
    #         # log(red(f'Note that this implementation is extremely slow, we recommend running on a native system that support the interop'))
    #         # log(red(f'An alternative is to install diff_point_rasterization and use the approximated tile-based rasterization, enabled by the `render_gs` switch'))
    #         # # raise RuntimeError(str(e) + ": This unrecoverable, please read the error message above")
    #         raise e
    #
    # def init_textures(self):
    #     from cuda import cudart
    #     if hasattr(self, 'cu_read_color'):
    #         CHECK_CUDART_ERROR(cudart.cudaGraphicsUnregisterResource(self.cu_read_color))
    #         CHECK_CUDART_ERROR(cudart.cudaGraphicsUnregisterResource(self.cu_write_color))
    #         CHECK_CUDART_ERROR(cudart.cudaGraphicsUnregisterResource(self.cu_read_lower))
    #         CHECK_CUDART_ERROR(cudart.cudaGraphicsUnregisterResource(self.cu_write_lower))
    #
    #     super().init_textures()
    #
    #     # Register image to read from
    #     flags = cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsReadOnly
    #     self.cu_read_color = CHECK_CUDART_ERROR(cudart.cudaGraphicsGLRegisterImage(self.read_color, gl.GL_TEXTURE_2D, flags))
    #     self.cu_write_color = CHECK_CUDART_ERROR(cudart.cudaGraphicsGLRegisterImage(self.write_color, gl.GL_TEXTURE_2D, flags))
    #     self.cu_read_lower = CHECK_CUDART_ERROR(cudart.cudaGraphicsGLRegisterImage(self.read_lower, gl.GL_TEXTURE_2D, flags))
    #     self.cu_write_lower = CHECK_CUDART_ERROR(cudart.cudaGraphicsGLRegisterImage(self.write_lower, gl.GL_TEXTURE_2D, flags))

    def forward(self, xyz: torch.Tensor, rgb: torch.Tensor, rad: torch.Tensor, occ: torch.Tensor, batch):
        """
        Renders a 3D point cloud using OpenGL and returns the rendered RGB image, accumulated alpha image, and depth map.

        Args:
            xyz (torch.Tensor): A tensor of shape (B, N, 3) containing the 3D coordinates of the points.
            rgb (torch.Tensor): A tensor of shape (B, N, 3) containing the RGB color values of the points.
            rad (torch.Tensor): A tensor of shape (B, N, 1) containing the radii of the points.
            batch (dotdict): A dictionary containing the camera parameters and other metadata for the batch.

        Returns:
            A tuple containing the rendered RGB image, accumulated alpha image, and depth map, all as torch.Tensors.
            The RGB image has shape (1, H, W, 3), the alpha image has shape (1, H, W, 1), and the depth map has shape (1, H, W, 1).

        The method first resizes the OpenGL texture to match the height and width of the output image. It then sets the OpenGL viewport and scissor to only render in the region of the viewport specified by the output image size.

        It concatenates the `xyz`, `rgb`, and `rad` tensors along the last dimension and flattens the result into a 1D tensor.

        The method then uploads the input data to OpenGL for rendering and performs depth peeling using OpenGL. The method uploads the camera parameters to OpenGL and renders the point cloud, saving the output buffer to the `back_fbo` attribute of the class.

        Finally, the method copies the rendered image and depth back to the CPU as torch.Tensors and reshapes them to match the output image size. The RGB image is returned with shape (1, H, W, 3), the accumulated alpha image is returned with shape (1, H, W, 1), and the depth map is returned with shape (1, H, W, 1).
        """
        from cuda import cudart
        kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice

        # !: BATCH
        H, W = batch.meta.H[0].item(), batch.meta.W[0].item()
        self.resize_textures(H, W)  # maybe resize the texture
        self.resize_buffers(xyz.shape[1])  # maybe resize the buffer
        _, _, old_W, old_H = gl.glGetIntegerv(gl.GL_VIEWPORT)
        gl.glViewport(0, 0, W, H)
        gl.glScissor(0, 0, W, H)  # only render in this small region of the viewport

        # Prepare for input data
        data = torch.cat([xyz, rgb, rad, occ], dim=-1).type(self.dtype).ravel()

        # Upload to opengl for rendering
        CHECK_CUDART_ERROR(cudart.cudaGraphicsMapResources(1, self.cu_vbo, torch.cuda.current_stream().cuda_stream))
        cu_vbo_ptr, cu_vbo_size = CHECK_CUDART_ERROR(cudart.cudaGraphicsResourceGetMappedPointer(self.cu_vbo))
        assert cu_vbo_size >= data.numel() * data.element_size(), f'PyTorch(CUDA) and OpenGL vertex buffer size mismatch ({data.numel() * data.element_size()} v.s. {cu_vbo_size}), CUDA side should be less than or equal to the OpenGL side'
        CHECK_CUDART_ERROR(cudart.cudaMemcpyAsync(cu_vbo_ptr,
                                                  data.data_ptr(),
                                                  data.numel() * data.element_size(),
                                                  kind,
                                                  torch.cuda.current_stream().cuda_stream))
        CHECK_CUDART_ERROR(cudart.cudaGraphicsUnmapResources(1, self.cu_vbo, torch.cuda.current_stream().cuda_stream))

        # Perform rasterization (depth peeling using OpenGL)
        if 'meta_stream' in batch.meta: batch.meta.meta_stream.synchronize()  # wait for gpu -> cpu copy to finish
        back_fbo = self.rasterize(Camera(batch=batch.meta), xyz.shape[-2])  # will upload and render, save output buffer to back_fbo

        # Copy rendered image and depth back as tensor
        cu_tex = self.cu_write_color if back_fbo == self.write_fbo else self.cu_read_color  # double buffered depth peeling
        cu_dpt = self.cu_write_lower if back_fbo == self.write_fbo else self.cu_read_lower  # double buffered depth peeling

        # Prepare the output # !: BATCH
        rgb_map = torch.empty((H, W, 4), dtype=self.tex_dtype, device='cuda')  # to hold the data from opengl
        dpt_map = torch.empty((H, W, 1), dtype=torch.float, device='cuda')  # to hold the data from opengl

        # The resources in resources may be accessed by CUDA until they are unmapped.
        # The graphics API from which resources were registered should not access any resources while they are mapped by CUDA.
        # If an application does so, the results are undefined.
        CHECK_CUDART_ERROR(cudart.cudaGraphicsMapResources(1, cu_tex, torch.cuda.current_stream().cuda_stream))
        CHECK_CUDART_ERROR(cudart.cudaGraphicsMapResources(1, cu_dpt, torch.cuda.current_stream().cuda_stream))
        cu_tex_arr = CHECK_CUDART_ERROR(cudart.cudaGraphicsSubResourceGetMappedArray(cu_tex, 0, 0))
        cu_dpt_arr = CHECK_CUDART_ERROR(cudart.cudaGraphicsSubResourceGetMappedArray(cu_dpt, 0, 0))
        CHECK_CUDART_ERROR(cudart.cudaMemcpy2DFromArrayAsync(rgb_map.data_ptr(),  # dst
                                                             W * 4 * rgb_map.element_size(),  # dpitch
                                                             cu_tex_arr,  # src
                                                             0,  # wOffset
                                                             0,  # hOffset
                                                             W * 4 * rgb_map.element_size(),  # width Width of matrix transfer (columns in bytes)
                                                             H,  # height
                                                             kind,  # kind
                                                             torch.cuda.current_stream().cuda_stream))  # stream
        CHECK_CUDART_ERROR(cudart.cudaMemcpy2DFromArrayAsync(dpt_map.data_ptr(),
                                                             W * 1 * dpt_map.element_size(),
                                                             cu_dpt_arr,
                                                             0,
                                                             0,
                                                             W * 1 * dpt_map.element_size(),
                                                             H,
                                                             kind,
                                                             torch.cuda.current_stream().cuda_stream))
        CHECK_CUDART_ERROR(cudart.cudaGraphicsUnmapResources(1, cu_tex, torch.cuda.current_stream().cuda_stream))  # MARK: SYNC
        CHECK_CUDART_ERROR(cudart.cudaGraphicsUnmapResources(1, cu_dpt, torch.cuda.current_stream().cuda_stream))  # MARK: SYNC

        # Ouput reshaping
        rgb_map, dpt_map = rgb_map[None].flip(1), dpt_map[None].flip(1)
        rgb_map, acc_map = rgb_map[..., :3], rgb_map[..., 3:]
        dpt_map = torch.where(dpt_map == 0, dpt_map.max(), dpt_map)

        # Some house keepings
        gl.glViewport(0, 0, old_W, old_H)
        gl.glScissor(0, 0, old_W, old_H)
        return rgb_map, acc_map, dpt_map


def get_pytorch3d_ndc_K(K: torch.Tensor, H: int, W: int):
    M = min(H, W)
    K = torch.cat([K, torch.zeros_like(K[..., -1:, :])], dim=-2)
    K = torch.cat([K, torch.zeros_like(K[..., :, -1:])], dim=-1)
    K[..., 3, 2] = 1  # ...? # HACK: pytorch3d magic
    K[..., 2, 2] = 0  # ...? # HACK: pytorch3d magic
    K[..., 2, 3] = 1  # ...? # HACK: pytorch3d magic

    K[..., 0, 1] = 0
    K[..., 1, 0] = 0
    K[..., 2, 0] = 0
    K[..., 2, 1] = 0

    K[..., 0, 0] = K[..., 0, 0] * 2.0 / M  # fx
    K[..., 1, 1] = K[..., 1, 1] * 2.0 / M  # fy
    K[..., 0, 2] = -(K[..., 0, 2] - W / 2.0) * 2.0 / M  # px
    K[..., 1, 2] = -(K[..., 1, 2] - H / 2.0) * 2.0 / M  # py
    return K

def get_pytorch3d_camera_params(H, W, K, R, T):
    # Extract pytorc3d camera parameters from batch input
    # R and T are applied on the right (requires a transposed R from OpenCV camera format)
    # Coordinate system is different from that of OpenCV (cv: right down front, 3d: left up front)
    # However, the correction has to be down on both T and R... (instead of just R)
    C = -R.mT @ T  # B, 3, 1
    R = R.clone()
    R[..., 0, :] *= -1  # flip x row
    R[..., 1, :] *= -1  # flip y row
    T = (-R @ C)[..., 0]  # c2w back to w2c
    R = R.mT  # applied left (left multiply to right multiply, god knows why...)


    K = get_pytorch3d_ndc_K(K, H, W)

    return H, W, K, R, T, C

def torchRender(xyz, rgb, rad, density, H, W, K, R, T, K_points = 8):
    # rgb_compose=torch.ones_like(rgb_compose,device=rgb_compose.device,dtype=rgb_compose.dtype)
    density = density.squeeze()
    rad = rad.squeeze()
    H_d, W_d = H.item(), W.item()

    # ndc_corrd, ndc_radius = get_ndc(xyz, K, R, T, H, W, radius)
    H, W, K, R, T, C = get_pytorch3d_camera_params(H_d, W_d, K, R, T)
    # K, R, T, C = to_x([K, R, T, C], torch.float)
    ndc_pcd = PointsRasterizer().transform(Pointclouds(xyz), cameras=PerspectiveCameras(K=K, R=R, T=T, device=xyz.device)).points_padded()  # B, N, 3
    ndc_rad = abs(K[..., 1, 1][..., None] * rad[..., 0] / (ndc_pcd[..., -1] + 1e-10))  # z: B, 1 * B, N, world space radius

    ndc_rad = ndc_rad
    ndc_rad = ndc_rad.reshape(-1)

    ndc_pcd = ndc_pcd.unsqueeze(0)
    ndc_pcd = ndc_pcd.reshape(1, -1, 3)

    idx, zbuf, dists = rasterize_points(Pointclouds(ndc_pcd, rgb), (H_d, W_d), radius=ndc_rad, points_per_pixel=K_points, max_points_per_bin = 300000)

    msk = idx != -1
    idx = torch.where(msk, idx, 0).long()

    # Prepare controller for composition
    pix_rad = ndc_rad[idx]  # B, H, W, K (B, HWK -> B, N -> B, H, W, K)
    pix_dens = density[idx]
    pix_dens = pix_dens * (1 - dists / (pix_rad * pix_rad))  # B, H, W, K
    pix_dens = pix_dens.clip(0, 1)
    pix_dens = torch.where(msk, pix_dens, 0)

    # Prepare values for composition
    dpt = (xyz - C.mT).norm(dim=-1, keepdim=True)
    rgb = torch.cat([rgb, density.reshape(1,-1, 1), dpt], dim=-1)  # B, N, 3 + C

    # The actual computation
    compositor = AlphaCompositor()
    rgb = compositor(idx.permute(0, 3, 1, 2),
                          pix_dens.permute(0, 3, 1, 2),
                          # NOTE: This will pass gradient back to point position and radius
                          rgb.view(-1, rgb.shape[-1]).permute(1, 0)).permute(0, 2, 3, 1)  # B, H, W, 3

    rgb, acc, dpt = rgb[..., :-2], rgb[..., -2:-1], rgb[..., -1:]
    dpt = dpt + (1 - acc) * dpt.max()  # only for the looks (rendered depth are already premultiplied)

    return rgb, acc, dpt
