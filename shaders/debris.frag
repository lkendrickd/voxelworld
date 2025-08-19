#version 330 core

layout (location = 0) out vec4 fragColor;

uniform sampler2DArray u_texture_array_0;
uniform int u_voxel_id;   // layer index to match parent voxel
in vec2 uv;

void main() {
    // Use the middle band (side texture) of the 3-horizontal atlas layout
    vec2 face_uv = uv;
    face_uv.x = uv.x / 3.0 - 1.0 / 3.0; // pick face_id = 1 (sides)
    vec3 col = texture(u_texture_array_0, vec3(face_uv, float(u_voxel_id))).rgb;
    fragColor = vec4(col, 1.0);
}
