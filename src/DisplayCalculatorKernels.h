/*
 * DisplayCalculatorKernels.h
 *
 *  Created on: 28 lis 2019
 *      Author: karol
 */

#ifndef DISPLAYCALCULATORKERNELS_H_
#define DISPLAYCALCULATORKERNELS_H_
#include "Mesh.h"

void SaveToConstantMemory(short * h_triangles, int verticesLenght, int trianglesLength);
void SaveVerticesToConstantMemory(float3 * d_vertices, int length);

__global__ void CastRaysOrthogonal(
		float3 cameraBottomLeftCorner, float3 xOffset, float3 yOffset,
		int width, int height,
		int * colorMap, DeviceMeshData mesh);


#endif /* DISPLAYCALCULATORKERNELS_H_ */
