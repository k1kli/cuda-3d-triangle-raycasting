/*
 * Material.cpp
 *
 *  Created on: 9 gru 2019
 *      Author: root
 */

#include "../include/Material.cuh"
#include "../include/DisplayCalculatorKernels.cuh"

Material::Material() {
	// TODO Auto-generated constructor stub

}

Material::~Material() {
	// TODO Auto-generated destructor stub
}

void Material::SendToGPU() {
	UpdateMaterialGPU(color, diffuse, specular, smoothness);
}
