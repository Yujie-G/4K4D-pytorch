import numpy as np
import torch
import json
import glm
from glm import vec2, vec3, vec4, mat3, mat4, mat4x3, mat2x3  # This is actually highly optimized
import OpenGL.GL as gl
from OpenGL import EGL as egl
import ctypes
from ctypes import pointer

from lib.utils.base_utils import *

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

def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d

def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T

    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy

def to_tensor(batch, ignore_list: bool = False) -> Union[torch.Tensor, dotdict[str, torch.Tensor]]:
    if isinstance(batch, (tuple, list)) and not ignore_list:
        batch = [to_tensor(b, ignore_list) for b in batch]
    elif isinstance(batch, dict):
        batch = dotdict({k: to_tensor(v, ignore_list) for k, v in batch.items()})
    elif isinstance(batch, torch.Tensor):
        pass
    else:  # numpy and others
        batch = torch.as_tensor(batch)
    return batch

class Camera:
    # Helper class to manage camera parameters
    def __init__(self,
                 H: int = 768,
                 W: int = 1366,
                 K: torch.Tensor = torch.tensor([[1366.0, 0.0, 683], [0.0, 1366.0, 384.0], [0.0, 0.0, 1.0]]),  # intrinsics
                 R: torch.Tensor = torch.tensor([[-0.9977766275405884, 0.06664637476205826, 0.0], [0.004728599451482296, 0.07079283893108368, -0.9974799156188965], [-0.0664784237742424, -0.9952622056007385, -0.07095059007406235]]),  # extrinsics
                 T: torch.Tensor = torch.tensor([[-2.059340476989746e-5], [2.5779008865356445e-6], [-3.000047445297241]]),  # extrinsics
                 n: float = 0.002,  # bounds limit
                 f: float = 1000,  # bounds limit
                 t: float = 0.0,  # temporal dimension (implemented as a float instead of int)
                 v: float = 0.0,  # view dimension (implemented as a float instead of int)
                 bounds: torch.Tensor = torch.tensor([[-100.0, -100.0, -100.0], [100.0, 100.0, 100.0]]),  # bounding box

                 # camera update hyperparameters
                 origin: torch.Tensor = torch.tensor([0.0, 0.0, 0.0]),
                 world_up: torch.Tensor = torch.tensor([0.0, 0.0, 1.0]),
                 movement_speed: float = 1.0,  # gui movement speed
                 movement_force: float = 1.0,  # include some physiscs
                 drag_coeff_mult: float = 1.0,  # include some physiscs
                 constant_drag: float = 1.0,
                 mass: float = 0.1,
                 moment_of_inertia: float = 0.1,
                 movement_torque: float = 1.0,
                 angular_friction: float = 2.0,
                 constant_torque: float = 1.0,

                 min_interval: float = 0.0334,  # simulate at at least 30 fps
                 pause_physics: bool = False,

                 batch: dotdict = None,  # will ignore all other inputs
                 string: str = None,  # will ignore all other inputs
                 **kwargs,
                 ) -> None:

        # Batch (network input parameters)
        if string is None:
            if batch is None:
                batch = dotdict()
                batch.H, batch.W, batch.K, batch.R, batch.T, batch.n, batch.f, batch.t, batch.v, batch.bounds = H, W, K, R, T, n, f, t, v, bounds
            batch = to_tensor(batch, ignore_list=True)
            self.from_batch(batch)

            # Other configurables
            self.origin = vec3(*origin)
            # self.origin = self.center  # rotate about center
            self.world_up = vec3(*world_up)
            self.movement_speed = movement_speed
            # self.front = self.front  # will trigger an update
        else:
            self.from_string(string)

        # Internal states to facilitate camera position change
        self.is_dragging = False  # rotation
        self.about_origin = False  # about origin rotation
        self.is_panning = False  # translation
        self.lock_fx_fy = True
        self.drag_start = vec2(0.0)

        # Internal states to facilitate moving with mass
        self.mass = mass
        self.force = vec3(0.0)
        self.speed = vec3(0.0)  # no movement
        self.acc = vec3(0.0)
        self.drag_coeff_mult = drag_coeff_mult
        self.movement_force = movement_force
        self.constant_drag = constant_drag
        self.pause_physics = pause_physics
        self.min_interval = min_interval

        self.torque = vec3(0.0)
        self.moment_of_inertia = moment_of_inertia
        self.angular_speed = vec3(0.0)  # relative angular speed on three euler angles
        self.angular_acc = vec3(0.0)
        self.angular_friction = angular_friction
        self.constant_torque = constant_torque
        self.movement_torque = movement_torque

    def step(self, interval: float):
        if self.pause_physics: return

        # Limit interval to make the simulation more stable
        interval = min(interval, self.min_interval)

        # Compute the drag force
        speed2 = glm.dot(self.speed, self.speed)
        if speed2 > 1.0:
            # Drag at opposite direction of movement
            drag = -speed2 * (self.speed / speed2) * self.drag_coeff_mult
        elif speed2 > 0:
            # Constant drag if speed is blow a threshold to make it stop faster
            drag = -self.constant_drag * self.speed
        else:
            drag = vec3(0.0)

        # Compute acceleration and final speed
        self.acc = (self.force + drag) / self.mass
        self.speed += self.acc * interval

        # Compute displacement in this interval
        speed2 = glm.dot(self.speed, self.speed)
        if speed2 > 0:
            direction = mat3(self.right, -glm.normalize(glm.cross(self.right, self.world_up)), self.world_up)
            movement = direction @ (self.speed - self.acc * interval / 2) * interval
            self.center += movement

        # Compute rotation change

        # Compute the drag torque
        speed2 = glm.dot(self.angular_speed, self.angular_speed)
        if speed2 > 0.1:
            # Drag at opposite direction of movement
            drag = -speed2 * (self.angular_speed / speed2) * self.angular_friction
        elif speed2 > 0.0:
            # Constant drag if speed is blow a threshold to make it stop faster
            drag = -self.constant_torque * self.angular_speed
        else:
            drag = vec3(0.0)

        # Compute angular acceleration and final angular speed
        self.angular_acc = (self.torque + drag) / self.moment_of_inertia
        self.angular_speed += self.angular_acc * interval

        # Angular movement direction
        delta = self.angular_speed * interval  # about x, y and z axis (euler angle)

        # Limit look up
        dot = glm.dot(self.world_up, self.front)
        self.drag_ymin = -np.arccos(-dot) + 0.01  # drag up, look down
        self.drag_ymax = np.pi + self.drag_ymin - 0.02  # remove the 0.01 of drag_ymin

        # Rotate about euler angle
        EPS = 1e-7
        if abs(delta.x) > EPS or abs(delta.y) > EPS or abs(delta.z) > EPS:
            m = mat4(1.0)
            m = glm.rotate(m, np.clip(delta.x, self.drag_ymin, self.drag_ymax), self.right)
            m = glm.rotate(m, delta.y, -self.world_up)
            m = glm.rotate(m, delta.z, self.front)
            center = self.center
            self.front = m @ self.front  # might overshoot and will update center
            self.center = center

    @property
    def w2p(self):
        ixt = mat4(self.ixt)
        ixt[3, 3] = 0
        ixt[2, 3] = 1
        return ixt @ self.ext  # w2c -> c2p = w2p

    @property
    def V(self): return self.c2w

    @property
    def ixt(self): return self.K

    @property
    def gl_ext(self):
        gl_c2w = self.c2w
        gl_c2w[0] *= 1  # do notflip x
        gl_c2w[1] *= -1  # flip y
        gl_c2w[2] *= -1  # flip z
        gl_ext = glm.affineInverse(gl_c2w)
        return gl_ext  # use original opencv ext since we've taken care of the intrinsics in gl_ixt

    @property
    def gl_ixt(self):
        # Construct opengl camera matrix with projection & clipping
        # https://fruty.io/2019/08/29/augmented-reality-with-opencv-and-opengl-the-tricky-projection-matrix/
        # https://gist.github.com/davegreenwood/3a32d779f81f08dce32f3bb423672191
        # fmt: off
        gl_ixt = mat4(
                      2 * self.fx / self.W,                          0,                                       0,  0,
                       2 * self.s / self.W,       2 * self.fy / self.H,                                       0,  0,
                1 - 2 * (self.cx / self.W), 2 * (self.cy / self.H) - 1,   (self.f + self.n) / (self.n - self.f), -1,
                                         0,                          0, 2 * self.f * self.n / (self.n - self.f),  0,
        )
        # fmt: on

        return gl_ixt

    @property
    def ext(self): return self.w2c

    @property
    def w2c(self):
        w2c = mat4(self.R)
        w2c[3] = vec4(*self.T, 1.0)
        return w2c

    @property
    def c2w(self):
        return glm.affineInverse(self.w2c)

    @property
    def right(self) -> vec3: return vec3(self.R[0, 0], self.R[1, 0], self.R[2, 0])  # c2w R, 0 -> 3,

    @property
    def down(self) -> vec3: return vec3(self.R[0, 1], self.R[1, 1], self.R[2, 1])  # c2w R, 1 -> 3,

    @property
    def front(self) -> vec3: return vec3(self.R[0, 2], self.R[1, 2], self.R[2, 2])  # c2w R, 2 -> 3,

    @front.setter
    def front(self, v: vec3):
        front = v  # the last row of R
        self.R[0, 2], self.R[1, 2], self.R[2, 2] = front.x, front.y, front.z
        right = glm.normalize(glm.cross(self.front, self.world_up))  # right
        self.R[0, 0], self.R[1, 0], self.R[2, 0] = right.x, right.y, right.z
        down = glm.cross(self.front, self.right)  # down
        self.R[0, 1], self.R[1, 1], self.R[2, 1] = down.x, down.y, down.z

    @property
    def center(self):
        return -glm.transpose(self.R) @ self.T  # 3,

    @center.setter
    def center(self, v: vec3):
        self.T = -self.R @ v  # 3, 1

    @property
    def s(self): return self.K[1, 0]

    @s.setter
    def s(self, s): self.K[1, 0] = s

    @property
    def fx(self): return self.K[0, 0]

    @fx.setter
    def fx(self, v: float):
        v = min(v, 1e5)
        v = max(v, 1e-3)
        if self.lock_fx_fy:
            self.K[1, 1] = v / self.K[0, 0] * self.K[1, 1]
        self.K[0, 0] = v

    @property
    def fy(self): return self.K[1, 1]

    @fy.setter
    def fy(self, v: float):
        if self.lock_fx_fy:
            self.K[0, 0] = v / self.K[1, 1] * self.K[0, 0]
        self.K[1, 1] = v

    @property
    def cx(self): return self.K[2, 0]

    @cx.setter
    def cx(self, v: float):
        self.K[2, 0] = v

    @property
    def cy(self): return self.K[2, 1]

    @cy.setter
    def cy(self, v: float):
        self.K[2, 1] = v

    def begin_dragging(self,
                       x: float, y: float,
                       is_panning: bool,
                       about_origin: bool,
                       ):
        self.is_dragging = True
        self.is_panning = is_panning
        self.about_origin = about_origin
        self.drag_start = vec2([x, y])

    def end_dragging(self):
        self.is_dragging = False

    def update_dragging(self, x: float, y: float):
        if not self.is_dragging:
            return

        current = vec2(x, y)
        delta = current - self.drag_start
        delta /= max(self.H, self.W)
        delta *= -1

        self.drag_start = vec2([x, y])
        self.drag_start_front = self.front  # a recording
        self.drag_start_down = self.down
        self.drag_start_right = self.right
        self.drag_start_center = self.center
        self.drag_start_origin = self.origin
        self.drag_start_world_up = self.world_up

        # Need to find the max or min delta y to align with world_up
        dot = glm.dot(self.world_up, self.front)
        self.drag_ymin = -np.arccos(-dot) + 0.01  # drag up, look down
        self.drag_ymax = np.pi + self.drag_ymin - 0.02  # remove the 0.01 of drag_ymin

        if self.is_panning:
            delta *= self.movement_speed
            center_delta = delta[0] * self.drag_start_right + delta[1] * self.drag_start_down
            self.center = self.drag_start_center + center_delta
            if self.about_origin:
                self.origin = self.drag_start_origin + center_delta
        else:
            m = mat4(1.0)
            m = glm.rotate(m, delta.x % 2 * np.pi, self.world_up)
            m = glm.rotate(m, np.clip(delta.y, self.drag_ymin, self.drag_ymax), self.drag_start_right)
            self.front = m @ self.drag_start_front  # might overshoot

            if self.about_origin:
                self.center = -m @ (self.origin - self.drag_start_center) + self.origin

    def move(self, x_offset: float, y_offset: float):
        speed_factor = 1e-1
        movement = y_offset * speed_factor
        movement = movement * self.front * self.movement_speed
        self.center += movement

        if self.is_dragging:
            self.drag_start_center += movement

    def to_batch(self):
        meta = dotdict()
        meta.H = torch.as_tensor(self.H)
        meta.W = torch.as_tensor(self.W)
        meta.K = torch.as_tensor(self.K.to_list(), dtype=torch.float).mT
        meta.R = torch.as_tensor(self.R.to_list(), dtype=torch.float).mT
        meta.T = torch.as_tensor(self.T.to_list(), dtype=torch.float)[..., None]
        meta.n = torch.as_tensor(self.n, dtype=torch.float)
        meta.f = torch.as_tensor(self.f, dtype=torch.float)
        meta.t = torch.as_tensor(self.t, dtype=torch.float)
        meta.v = torch.as_tensor(self.v, dtype=torch.float)
        meta.bounds = torch.as_tensor(self.bounds.to_list(), dtype=torch.float)  # no transpose for bounds

        # GUI related elements
        meta.mass = torch.as_tensor(self.mass, dtype=torch.float)
        meta.moment_of_inertia = torch.as_tensor(self.moment_of_inertia, dtype=torch.float)
        meta.movement_force = torch.as_tensor(self.movement_force, dtype=torch.float)
        meta.movement_torque = torch.as_tensor(self.movement_torque, dtype=torch.float)
        meta.movement_speed = torch.as_tensor(self.movement_speed, dtype=torch.float)
        meta.origin = torch.as_tensor(self.origin.to_list(), dtype=torch.float)
        meta.world_up = torch.as_tensor(self.world_up.to_list(), dtype=torch.float)

        batch = dotdict()
        batch.update(meta)
        batch.meta.update(meta)
        return batch

    # def to_easymocap(self):
    #     batch = self.to_batch()
    #     camera = to_numpy(batch)
    #     return camera
    #
    # def from_easymocap(self, camera: dict):
    #     batch = to_tensor(camera)
    #     self.from_batch(batch)
    #     return self
    #
    # def to_string(self) -> str:
    #     batch = to_list(self.to_batch().meta)
    #     return json.dumps(batch)

    def from_string(self, string: str):
        batch = to_tensor(dotdict(json.loads(string)), ignore_list=True)
        self.from_batch(batch)

    def from_batch(self, batch: dotdict):
        H, W, K, R, T, n, f, t, v, bounds = batch.H, batch.W, batch.K, batch.R, batch.T, batch.n, batch.f, batch.t, batch.v, batch.bounds

        # Batch (network input parameters)
        self.H = int(H)
        self.W = int(W)
        self.K = mat3(*K.mT.ravel())
        self.R = mat3(*R.mT.ravel())
        self.T = vec3(*T.ravel())  # 3,
        self.n = float(n)
        self.f = float(f)
        self.t = float(t)
        self.v = float(v)
        self.bounds = mat2x3(*bounds.ravel())  # 2, 3

        if 'mass' in batch: self.mass = float(batch.mass)
        if 'moment_of_inertia' in batch: self.moment_of_inertia = float(batch.moment_of_inertia)
        if 'movement_force' in batch: self.movement_force = float(batch.movement_force)
        if 'movement_torque' in batch: self.movement_torque = float(batch.movement_torque)
        if 'movement_speed' in batch: self.movement_speed = float(batch.movement_speed)
        if 'origin' in batch: self.origin = vec3(*batch.origin.ravel())  # 3,
        if 'world_up' in batch: self.world_up = vec3(*batch.world_up.ravel())  # 3,
        return self

    def custom_pose(self, R: torch.Tensor, T: torch.Tensor, K: torch.Tensor):
        # self.K = mat3(*K.mT.ravel())
        self.R = mat3(*R.mT.ravel())
        self.T = vec3(*T.ravel())

