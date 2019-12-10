#include "DisplayCalculatorKernels.h"
#include "cuda_runtime.h"
#include "Mesh.h"
#include "stdio.h"
#include <helper_math.h>
#include "defines.h"


__device__ __constant__ float3 c_lightColors[128];
__device__ __constant__ float3 c_lightPositions[128];
//__device__ __constant__ int c_lightCount;
__device__ __constant__ uint objectColor = 0xFFFFFFFF;
__device__ __constant__ float3 zero;
__device__ __constant__ float3 one;
__device__ __constant__ float3 toObserver;
__device__ __constant__ float3 ray;

#define c_lightCount 2

void InitConstantMemory()
{


	float3 h_zero = make_float3(0.0f, 0.0f, 0.0f);

	cudaMemcpyToSymbol(zero, &h_zero, sizeof(float3),0,cudaMemcpyHostToDevice);

	float3 h_one = make_float3(1.0f, 1.0f, 1.0f);

	cudaMemcpyToSymbol(one, &h_one, sizeof(float3),0,cudaMemcpyHostToDevice);


	float3 h_toObserver = make_float3(0.0f, 0.0f, -1.0f);

	cudaMemcpyToSymbol(toObserver, &h_toObserver, sizeof(float3),0,cudaMemcpyHostToDevice);

	float3 h_ray = make_float3(0.0f, 0.0f, 1.0f);

	cudaMemcpyToSymbol(ray, &h_ray, sizeof(float3),0,cudaMemcpyHostToDevice);
}

void UpdateLightsGPU(float3 * lightColors, float3 * lightPositions, int lightCount)
{

	cudaMemcpyToSymbol(c_lightColors, lightColors, sizeof(float3)*lightCount,0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(c_lightPositions, lightPositions, sizeof(float3)*lightCount,0,cudaMemcpyHostToDevice);
	//cudaMemcpyToSymbol(c_lightCount, &lightCount, sizeof(int),0,cudaMemcpyHostToDevice);
}

__device__ float GetDistanceToClosestHitpointInBatch(float3 &  rayStartingPoint,
		DeviceMeshData * p_mesh, float4 * reachableTriangles, int reachableTrianglesSize,
		float3 * closestHitPointNormal);



__device__ unsigned int CalculateLight(float3 &  hitPoint,float3 &  normalVector,
		float diffuseFactor, float specularFactor, int m);


__global__ void CastRaysOrthogonal(
		float3 cameraBottomLeftCorner, float3 xOffset, float3 yOffset,
		int width, int height,
		int * colorMap, DeviceMeshData mesh)
{
	const unsigned int threadX = threadIdx.x;
	const unsigned int threadY = threadIdx.y;
	const unsigned int blockX = blockIdx.x;
	const unsigned int blockY = blockIdx.y;

	const unsigned int mapIndexX = blockX*blockDim.x+threadX;
	const unsigned int mapIndexY = blockY*blockDim.y+threadY;
	const unsigned int mapIndex = mapIndexY*width+mapIndexX;
	float3 rayStartingPoint = cameraBottomLeftCorner
			+xOffset*mapIndexX
			+yOffset*mapIndexY;
	int reachableTrianglesSize = blockDim.x*blockDim.y*3;

	float closestDistance = INFINITY;
	float3 closestHitPointNormal;
	extern __shared__ float4 reachableTriangles[];
	{
		float2 blockStart = make_float2(cameraBottomLeftCorner + xOffset*blockX*blockDim.x + yOffset*blockY*blockDim.y);
		float2 blockEnd = make_float2(make_float3(blockStart,0) + xOffset*blockDim.x + yOffset*blockDim.y);
		for(int i = 0; i < mesh.trianglesLength/3; i+=blockDim.x*blockDim.y)
		{
			int inBatchTriangleId = (threadX+blockDim.x*threadY)*3;
			int triangleId = i*3+inBatchTriangleId;
			if(triangleId < mesh.trianglesLength)
			{
				reachableTriangles[inBatchTriangleId] = mesh.d_points[mesh.d_triangles[triangleId]];
				reachableTriangles[inBatchTriangleId+1] = mesh.d_points[mesh.d_triangles[triangleId+1]];
				reachableTriangles[inBatchTriangleId+2] = mesh.d_points[mesh.d_triangles[triangleId+2]];
				float minX = MIN(reachableTriangles[inBatchTriangleId].x, MIN(reachableTriangles[inBatchTriangleId+1].x,reachableTriangles[inBatchTriangleId+2].x));
				float maxX = MAX(reachableTriangles[inBatchTriangleId].x, MAX(reachableTriangles[inBatchTriangleId+1].x,reachableTriangles[inBatchTriangleId+2].x));
				float minY = MIN(reachableTriangles[inBatchTriangleId].y, MIN(reachableTriangles[inBatchTriangleId+1].y,reachableTriangles[inBatchTriangleId+2].y));
				float maxY = MAX(reachableTriangles[inBatchTriangleId].y, MAX(reachableTriangles[inBatchTriangleId+1].y,reachableTriangles[inBatchTriangleId+2].y));
				if(!(minX <= blockEnd.x && minY <= blockEnd.y && maxX >= blockStart.x && maxY >= blockStart.y))
				{
					reachableTriangles[inBatchTriangleId] = make_float4(INFINITY,0,0, 0);
				}
			}
			__syncthreads();
			if(mapIndexX < width && mapIndexY < height)
			{
				float3 hitPointNormal;
				float distance =  GetDistanceToClosestHitpointInBatch(rayStartingPoint,
						&mesh, reachableTriangles, MIN(reachableTrianglesSize, mesh.trianglesLength-i*3),
						&hitPointNormal);
				if(distance < closestDistance)
				{
					closestDistance = distance;
					closestHitPointNormal = hitPointNormal;
				}
			}
			__syncthreads();
		}
	}
	if(mapIndexX < width && mapIndexY < height)
	{
		if(closestDistance == INFINITY)
		{
			colorMap[mapIndex] = 0xFF5555FF;
		}
		else
		{

			float3 hitPoint = make_float3(rayStartingPoint.x, rayStartingPoint.y, rayStartingPoint.z+closestDistance);


			unsigned int color = CalculateLight(hitPoint, closestHitPointNormal, 0.7f, 0.3f, 30);
			colorMap[mapIndex] = color;

		}
	}

}
//based on http://geomalgorithms.com/a06-_intersect-2.html#intersect3D_RayTriangle()
__device__ float GetDistanceToClosestHitpointInBatch(float3 &  rayStartingPoint,
		DeviceMeshData * p_mesh, float4 * reachableTriangles, int reachableTrianglesSize,
		float3 * closestHitPointNormal)
{
	float closestDistance = INFINITY;
	for(int triangleId = 0; triangleId < reachableTrianglesSize; triangleId+=3)
	{
		if(reachableTriangles[triangleId].x == INFINITY)
			continue;
		float3 p1 = *((float3 *)&reachableTriangles[triangleId]);
		float3 p2 = *((float3 *)&reachableTriangles[triangleId+1]);
		float3 p3 = *((float3 *)&reachableTriangles[triangleId+2]);
		if(RayIntersectsWith(rayStartingPoint,p1, p2, p3))
		{
			float3 hitPointNormal = normalize(cross(p2-p1, p3-p1));
			float distance = dot(p1-rayStartingPoint, hitPointNormal)/hitPointNormal.z;
			if(closestDistance > distance)
			{
				closestDistance = distance;
				*closestHitPointNormal = hitPointNormal;
			}
		}
	}
	return closestDistance;

}
__device__ unsigned int CalculateLight(float3 &  hitPoint,float3 &  normalVector,
		float diffuseFactor, float specularFactor, int m)
{
	float3 floatColor = zero;
	for(int i = 0; i < c_lightCount; i++)
	{
		float3 toLight = normalize(c_lightPositions[i] - hitPoint);
		float3 reflectVector = 2*dot(toLight, normalVector)*normalVector-toLight;
		float firstDot = dot(toLight,normalVector);
		float secondDot = dot(reflectVector, toObserver);
		secondDot = powf(secondDot, m);
		floatColor += c_lightColors[i]*(diffuseFactor*firstDot+specularFactor*secondDot);
	}
	floatColor = clamp(floatColor, zero, one);
	unsigned int res =
			255u <<24 |
			((unsigned int)(floatColor.x*((objectColor>>16)&255))<<16) |
			((unsigned int)(floatColor.y*((objectColor>>8)&255))<<8) |
			(unsigned int)(floatColor.z*(objectColor&255));
	return res;

}

__device__ __host__ bool RayIntersectsWith(float3 & rayStartingPoint,
		float3 &  v1, float3 &  v2, float3 &  v3)
{
	float a = v2.y-v3.y;
	float b = v1.x-v3.x;
	float c = v3.x-v2.x;
	float d = v1.y-v3.y;
	float e = rayStartingPoint.x-v3.x;
	float f = rayStartingPoint.y-v3.y;
	float det = a*b+c*d;
	int detSign = det > 0 ? 1 : -1;
	float s = (a*e+c*f) * detSign;
	if(s < 0) return false;
	float t = (-d*e+b*f) * detSign;
	return t>= 0 && s+t <= det*detSign;
}
