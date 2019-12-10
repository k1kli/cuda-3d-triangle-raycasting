/*
 * Light.cpp
 *
 *  Created on: 9 gru 2019
 *      Author: root
 */

#include "Light.h"
#include "cuda_runtime.h"


Light::Light(float3 color, float3 position) {
	this->color = color;
	this->position = position;
}

Light::~Light() {
	// TODO Auto-generated destructor stub
}

