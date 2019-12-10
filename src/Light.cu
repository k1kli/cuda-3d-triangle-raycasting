#include "cuda_runtime.h"
#include "../include/Light.cuh"
Light::Light(float3 color, float3 position) {
	this->color = color;
	this->position = position;
}
Light::~Light() {
}
