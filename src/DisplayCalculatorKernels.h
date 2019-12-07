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

__device__ __host__ bool RayIntersectsWith(float3 &  rayStartingPoint,
		float3 & v1, float3 &  v2, float3 &  v3);

#endif /* DISPLAYCALCULATORKERNELS_H_ */
