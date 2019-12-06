/*
 * DisplayCalculator.cpp
 *
 *  Created on: 27 lis 2019
 *      Author: karol
 */

#include "DisplayCalculator.h"
#include "Mesh.h"
#include "defines.h"
#include <cuda_runtime.h>
#include <helper_math.h>
#include <helper_cuda.h>
#include "DisplayCalculatorKernels.h"

DisplayCalculator::DisplayCalculator() {
	// TODO Auto-generated constructor stub

}

DisplayCalculator::~DisplayCalculator() {
	// TODO Auto-generated destructor stub
}

void DisplayCalculator::GenerateDisplay() {
	if(!mesh.IsInitialized())
	{
		throw "Initialize mesh first";
	}
	DeviceMeshData meshData = mesh.GetDeviceMeshData();
	float3 rayDirection = make_float3(0.0f,0.0f, 1.0f);
	float3 rightDirection = make_float3(1.0f,0.0f, 0.0f);
	float3 upDirection = make_float3(0.0f,1.0f, 0.0f);
	float3 cameraBottomLeftCorner = cameraPosition + rightDirection * (-fovWidth/2) + upDirection * (-fovHeight/2);
	float3 xOffset = rightDirection*(fovWidth/mapWidth);
	float3 yOffset = upDirection*(fovHeight/mapHeight);
	dim3 threads(32,32,1);
	dim3 blocks(DIVROUNDUP(mapWidth, threads.x), DIVROUNDUP(mapHeight, threads.y),1);
	CastRaysOrthogonal<<<blocks, threads>>>(
			cameraBottomLeftCorner,xOffset,yOffset, mapWidth, mapHeight, d_colorMap, meshData);

	getLastCudaError("CastRaysOrthogonal failed");
	cudaDeviceSynchronize();
}

void DisplayCalculator::SetCameraPosition(float3 position) {
	this->cameraPosition = position;
}
void DisplayCalculator::SetCameraFieldOfView(float width, float height)
{
	fovWidth = width;
	fovHeight = height;
}
