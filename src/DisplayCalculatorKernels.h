/*
 * DisplayCalculatorKernels.h
 *
 *  Created on: 28 lis 2019
 *      Author: karol
 */

#ifndef DISPLAYCALCULATORKERNELS_H_
#define DISPLAYCALCULATORKERNELS_H_
#include "Mesh.h"

__global__ void CastRaysOrthogonal(
		float4 cameraTopLeftCorner, float4 rayDirection, float4 up,
		float cameraFovWidth, float cameraFovHeight,
		int width, int height,
		int * colorMap, DeviceMeshData * mesh);



#endif /* DISPLAYCALCULATORKERNELS_H_ */
