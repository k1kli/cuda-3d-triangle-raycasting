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
	float3 rayDirection = normalize(lookAt - cameraPosition);
	float3 rightDirection = cross(rayDirection, cameraUpDirection);
	float3 u = cross(rightDirection, rayDirection);
	float3 cameraBottomLeftCorner = cameraPosition + rightDirection * (-fovWidth/2) + u * (-fovHeight/2);
	float3 xOffset = rightDirection*(fovWidth/mapWidth);
	float3 yOffset = u*(fovHeight/mapHeight);
	dim3 threads(32,32,1);
	dim3 blocks(DIVROUNDUP(mapWidth, threads.x), DIVROUNDUP(mapHeight, threads.y),1);
	CastRaysOrthogonal<<<blocks, threads>>>(
			cameraBottomLeftCorner,rayDirection, xOffset,yOffset, mapWidth, mapHeight, d_colorMap, meshData);

	getLastCudaError("CastRaysOrthogonal failed");
	cudaDeviceSynchronize();
}

void DisplayCalculator::GenerateDisplayPerspective()
{
	if(!mesh.IsInitialized())
	{
		throw "Initialize mesh first";
	}
	DeviceMeshData meshData = mesh.GetDeviceMeshData();
	float3 forwardDirection = normalize(lookAt - cameraPosition);
	float3 rightDirection = cross(forwardDirection, cameraUpDirection);
	float3 u = cross(rightDirection, forwardDirection);

	float nearDistance = 1.0f;
	float farDistance = 5.0f;

	float3 xFarOffset = rightDirection*(fovWidth/mapWidth);
	float3 yFarOffset = u*(fovHeight/mapHeight);
	dim3 threads(32,32,1);
	dim3 blocks(DIVROUNDUP(mapWidth, threads.x), DIVROUNDUP(mapHeight, threads.y),1);
	CastRaysPerspective<<<blocks, threads>>>(
			cameraPosition,
			nearDistance, farDistance,
			xFarOffset, yFarOffset,
			forwardDirection,
			mapWidth, mapHeight,
			d_colorMap, meshData
			);

	getLastCudaError("CastRaysPerspective failed");
	cudaDeviceSynchronize();
}

void DisplayCalculator::SetCameraPosition(float3 position) {
	this->cameraPosition = position;
}

void DisplayCalculator::SetCameraLookAt(float3 lookAt, float3 upDirection) {
	this->lookAt = lookAt;
	this->cameraUpDirection = normalize(upDirection);
}
void DisplayCalculator::SetCameraFieldOfView(float width, float height)
{
	fovWidth = width;
	fovHeight = height;
}
