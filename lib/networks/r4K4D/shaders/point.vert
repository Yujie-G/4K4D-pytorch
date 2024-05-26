#version 130
#pragma vscode_glsllint_stage : vert

uniform mat4x4 K;
uniform mat4x4 M;
uniform mat4x4 V;
uniform int H;  // viewport size
uniform int W;
uniform float point_radius = 0.0015;

attribute vec3 position;
attribute vec3 color;   // rgb
attribute vec3 normal;  // normal

varying vec3 world_frag_pos;
varying vec3 cam_frag_pos;
varying vec3 vert_color;   // pass through
varying vec3 vert_normal;  // pass through

mat3 transpose(mat3 m) {
    return mat3(
        m[0][0], m[1][0], m[2][0],
        m[0][1], m[1][1], m[2][1],
        m[0][2], m[1][2], m[2][2]
    );
}

mat3 inverse(mat3 m) {
    float det = m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2]) -
                m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    return mat3(
        (m[1][1] * m[2][2] - m[2][1] * m[1][2]) / det,
        (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / det,
        (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det,
        (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / det,
        (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det,
        (m[1][0] * m[0][2] - m[0][0] * m[1][2]) / det,
        (m[1][0] * m[2][1] - m[2][0] * m[1][1]) / det,
        (m[2][0] * m[0][1] - m[0][0] * m[2][1]) / det,
        (m[0][0] * m[1][1] - m[1][0] * m[0][1]) / det
    );
}

void main() {
    vec4 pos = vec4(position, 1.0);  // in object space
    vec4 norm = vec4(normal, 1.0);   // in object space
    vec4 world = M * pos;            // in world space
    vec4 cam = V * world;            // in camera space

    // Outputs
    world_frag_pos = vec3(world);
    cam_frag_pos = vec3(cam);
    gl_Position = K * cam;  // doing a perspective projection to clip space

    vert_color = color;
    vert_normal = transpose(inverse(mat3(V * M))) * vec3(norm);  // in camera space

    // https://stackoverflow.com/questions/25780145/gl-pointsize-corresponding-to-world-space-size
    gl_PointSize = abs(H * K[1][1] * point_radius / gl_Position.w);  // need to determine size in pixels
    // radiusPixel = gl_PointSize / 2.0;
}
