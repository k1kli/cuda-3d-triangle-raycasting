/*
 * DisplayCalculator.cpp
 *
 *  Created on: 27 lis 2019
 *      Author: karol
 */

#include "../include/DisplayCalculator.cuh"
#include "../include/Mesh.cuh"
#include "../include/defines.h"
#include <cuda_runtime.h>
#include <helper_math.h>
#include <helper_cuda.h>
#include "../include/DisplayCalculatorKernels.cuh"

DisplayCalculator::DisplayCalculator(bool onCPU) {
	this->onCPU = onCPU;
	mesh = Mesh(onCPU);

}

DisplayCalculator::~DisplayCalculator() {
	// TODO Auto-generated destructor stub
}

void DisplayCalculator::GenerateDisplay() {
	if(onCPU)
	{
		GenerateDisplayCPU();
		return;
	}
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
	CastRaysOrthogonal<<<blocks, threads, threads.x*threads.y*3*sizeof(float4)>>>(
			cameraBottomLeftCorner,xOffset,yOffset, mapWidth, mapHeight, colorMap, meshData);

	cudaDeviceSynchronize();
	getLastCudaError("CastRaysOrthogonal failed");
}

void DisplayCalculator::GenerateDisplayCPU() {
	float3 rayDirection = make_float3(0.0f,0.0f, 1.0f);
	float3 rightDirection = make_float3(1.0f,0.0f, 0.0f);
	float3 upDirection = make_float3(0.0f,1.0f, 0.0f);
	float3 cameraBottomLeftCorner = cameraPosition + rightDirection * (-fovWidth/2) + upDirection * (-fovHeight/2);
	float3 xOffset = rightDirection*(fovWidth/mapWidth);
	float3 yOffset = upDirection*(fovHeight/mapHeight);
	for(int y = 0; y < mapHeight; y++)
	{
		for(int x = 0; x < mapWidth; x++)
		{
			float3 rayStartingPoint = cameraBottomLeftCorner + xOffset*x +yOffset*y;
			colorMap[mapWidth*y+x] = GetColorOfClosestHitpointCPU(rayStartingPoint);
		}
	}
}

unsigned int DisplayCalculator::GetColorOfClosestHitpointCPU(float3 & rayStartingPoint)
{
	float closestDistance = INFINITY;
	float3 closestHitPointNormal;
	for(int triangleId = 0; triangleId < mesh.trianglesLength; triangleId+=3)
	{
		float3 p1 = mesh.cpu_points_transformed[mesh.triangles[triangleId]];
		float3 p2 = mesh.cpu_points_transformed[mesh.triangles[triangleId+1]];
		float3 p3 = mesh.cpu_points_transformed[mesh.triangles[triangleId+2]];
		float minX = MIN(p1.x, MIN(p2.x,p3.x));
		float maxX = MAX(p1.x, MAX(p2.x,p3.x));
		float minY = MIN(p1.y, MIN(p2.y,p3.y));
		float maxY = MAX(p1.y, MAX(p2.y,p3.y));
		if(maxX < rayStartingPoint.x ||maxY < rayStartingPoint.y ||
				minX > rayStartingPoint.x ||minY > rayStartingPoint.y)
		{
			continue;
		}
		if(RayIntersectsWith(rayStartingPoint,p1, p2, p3))
		{
			float3 hitPointNormal = normalize(cross(p2-p1, p3-p1));
			float distance = dot(p1-rayStartingPoint, hitPointNormal)/hitPointNormal.z;
			if(closestDistance > distance)
			{
				closestDistance = distance;
				closestHitPointNormal = hitPointNormal;
			}
		}
	}
	if(closestDistance == INFINITY)
	{
		return 0x3333FFFF;
	}
	else
	{

		float3 hitPoint = make_float3(rayStartingPoint.x, rayStartingPoint.y, rayStartingPoint.z+closestDistance);


		const float3 lightPos(make_float3(-2.0f,3.0f,-2.0f));
		unsigned int color = CalculateLightCPU(hitPoint, closestHitPointNormal);
		return color;

	}
}
unsigned int DisplayCalculator::CalculateLightCPU(float3 & hitPoint, float3 &  normalVector)
{
	const float3 toObserver(make_float3(0,0,-1.0f));
	const float3 zero(make_float3(0,0,0));
	const float3 one(make_float3(1.0f,1.0f,1.0f));
	float3 floatColor = zero;
	for(int i = 0; i < sceneData.lights.size(); i++)
	{
		float3 toLight = normalize(sceneData.lights[i].position - hitPoint);
		float3 reflectVector = 2*dot(toLight, normalVector)*normalVector-toLight;
		float firstDot = dot(toLight,normalVector);
		float secondDot = dot(reflectVector, toObserver);
		secondDot = powf(secondDot, mesh.material.smoothness);
		floatColor += sceneData.lights[i].color*(mesh.material.diffuse*firstDot+mesh.material.specular*secondDot);
	}
	floatColor = clamp(floatColor,zero,one);
	unsigned int objectColor = mesh.material.color;
	unsigned int res =
			255u |
			((unsigned int)(floatColor.x*((objectColor>>16)&255))<<24) |
			((unsigned int)(floatColor.y*((objectColor>>8)&255))<<16) |
			((unsigned int)(floatColor.z*(objectColor&255))<<8);
	return res;

}

void DisplayCalculator::SetCameraPosition(float3 position) {
	this->cameraPosition = position;
}
void DisplayCalculator::SetCameraFieldOfView(float width, float height)
{
	fovWidth = width;
	fovHeight = height;
}
