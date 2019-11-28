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
	DeviceMeshData * d_meshData = mesh.GetDeviceMeshDataPointer();
	float4 rightDirection = make_float4(cross(make_float3(lookAt), make_float3(cameraUpDirection)),0);
	float4 cameraTopLeftCorner = cameraPosition + rightDirection * (-fovWidth/2) + cameraUpDirection * (fovHeight/2);
	dim3 threads(32,32,1);
	dim3 blocks(DIVROUNDUP(mapWidth, threads.x), DIVROUNDUP(mapHeight, threads.y),1);
	getLastCudaError("elsewhere");
	CastRaysOrthogonal<<<blocks, threads>>>(
			cameraTopLeftCorner,lookAt, cameraUpDirection, fovWidth, fovHeight, mapWidth, mapHeight, d_colorMap, d_meshData);

	getLastCudaError("CastRaysOrthogonal failed");
	cudaFree(d_meshData);
}

void DisplayCalculator::SetCameraPosition(float4 position) {
	this->cameraPosition = position;
}

void DisplayCalculator::SetCameraLookAt(float3 lookAt, float3 upDirection) {
	this->lookAt = make_float4(normalize(lookAt),0);
	this->cameraUpDirection = make_float4(normalize(upDirection),0);
}
void DisplayCalculator::SetCameraFieldOfView(float width, float height)
{
	fovWidth = width;
	fovHeight = height;
}
