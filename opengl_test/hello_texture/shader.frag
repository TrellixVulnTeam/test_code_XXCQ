#version 330 core
out vec4 FragColor;

in vec3 ourColor;
in vec2 TexCoord;

// texture samplers
uniform sampler2D texture1;
uniform sampler2D texture2;

uniform float textureRatio;

void main()
{
	// mix() 函数是 GLSL 提供的函数，可以 linearly interpolate between both textures。这里就是，80% 的第一个纹理和 20% 的第二个纹理。
    // 如果第三个参数是 0.0，那么就直接是第一个纹理；如果是 1.0，那么就是第二个纹理。
	FragColor = mix(texture(texture1, TexCoord), texture(texture2, TexCoord), textureRatio) * vec4(ourColor, 1.0);
}