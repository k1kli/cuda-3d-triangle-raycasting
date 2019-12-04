/*
 * DisplayCalculatorKernels.h
 *
 *  Created on: 28 lis 2019
 *      Author: karol
 */

#ifndef DISPLAYCALCULATORKERNELS_H_
#define DISPLAYCALCULATORKERNELS_H_
#include "Mesh.h"

void SaveToConstantMemory(float3 * h_vertices, short * h_triangles, int verticesLenght, int trianglesLength);

__global__ void CastRaysOrthogonal(
		float3 cameraBottomLeftCorner, float3 rayDirection, float3 xOffset, float3 yOffset,
		int width, int height,
		int * colorMap, DeviceMeshData mesh);

__global__ void CastRaysPerspective(
		float3 cameraCenter,
		float nearDistance, float farDistance,
		float3 xFarOffset, float3 yFarOffset,
		float3 forward,
		int width, int height,
		int * colorMap, DeviceMeshData mesh);

#endif /* DISPLAYCALCULATORKERNELS_H_ */
