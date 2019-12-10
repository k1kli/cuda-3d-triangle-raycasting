/*
 * Material.h
 *
 *  Created on: 9 gru 2019
 *      Author: root
 */

#ifndef MATERIAL_H_
#define MATERIAL_H_
#include "cuda_runtime.h"

class Material {
public:
	float smoothness;
	unsigned color;
	float diffuse;
	float specular;
	Material();
	virtual ~Material();
	void SendToGPU();
};

#endif /* MATERIAL_H_ */
