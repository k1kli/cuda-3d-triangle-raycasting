/*
 * Light.h
 *
 *  Created on: 9 gru 2019
 *      Author: root
 */

#ifndef LIGHT_H_
#define LIGHT_H_
#include "cuda_runtime.h"

class Light {
public:
	float3 color;
	float3 position;
	Light(float3 color, float3 position);
	virtual ~Light();
};

#endif /* LIGHT_H_ */
