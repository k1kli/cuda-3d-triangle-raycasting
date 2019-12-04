#include "DisplayCalculatorKernels.h"
#include "cuda_runtime.h"
#include "Mesh.h"
#include "stdio.h"
#include <helper_math.h>


__device__ __constant__ float3 c_vertices[4096];//48kb
__device__ __constant__ short c_triangles[6*1024];//12kb
__device__ __constant__ float3 lightColor;
__device__ __constant__ float3 lightPos;
__device__ __constant__ uint objectColor = 0xFFFFFF;
__device__ __constant__ float3 zero;
__device__ __constant__ float3 one;

void SaveToConstantMemory(float3 * h_vertices, short * h_triangles, int verticesLenght, int trianglesLength)
{
	cudaMemcpyToSymbol(c_vertices, h_vertices, sizeof(float3)*verticesLenght,0,cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(c_triangles, h_triangles, sizeof(short)*trianglesLength,0,cudaMemcpyHostToDevice);

	float3 h_lightColor = make_float3(1.0f,1.0f,1.0f);

	cudaMemcpyToSymbol(lightColor, &h_lightColor, sizeof(float3),0,cudaMemcpyHostToDevice);

	float3 h_zero = make_float3(0.0f,0.0f,0.0f);

	cudaMemcpyToSymbol(zero, &h_zero, sizeof(float3),0,cudaMemcpyHostToDevice);

	float3 h_one = make_float3(1.0f,1.0f,1.0f);

	cudaMemcpyToSymbol(one, &h_one, sizeof(float3),0,cudaMemcpyHostToDevice);

	float3 h_lightPos = make_float3(-2.0f,3.0f,-2.0f);

	cudaMemcpyToSymbol(lightPos, &h_lightPos, sizeof(float3),0,cudaMemcpyHostToDevice);
}

__device__ unsigned int GetColorOfClosestHitpoint(float3 &  ray, float3 &  rayStartingPoint,
		DeviceMeshData * p_mesh);


__device__ bool GetIntersectionPointWith(float3 &  ray, float3 &  rayStartingPoint,
		float3 & v0, float3 &  v1, float3 &  v2, float3 * tuv);


__device__ unsigned int CalculateLight(float3 toLight, float3  toObserver, float3 &  normalVector,
		float diffuseFactor, float specularFactor, int m);


__global__ void CastRaysOrthogonal(
		float3 cameraBottomLeftCorner, float3 rayDirection, float3 xOffset, float3 yOffset,
		int width, int height,
		int * colorMap, DeviceMeshData mesh)
{
	const unsigned int threadX = threadIdx.x;
	const unsigned int threadY = threadIdx.y;
	const unsigned int blockX = blockIdx.x;
	const unsigned int blockY = blockIdx.y;

	const unsigned int mapIndexX = blockX*blockDim.x+threadX;
	const unsigned int mapIndexY = blockY*blockDim.y+threadY;
	if(mapIndexX < width && mapIndexY < height)
	{
		const unsigned int mapIndex = mapIndexY*width+mapIndexX;
		float3 rayStartingPoint = cameraBottomLeftCorner
				+xOffset*mapIndexX
				+yOffset*mapIndexY;
		colorMap[mapIndex] = GetColorOfClosestHitpoint(rayDirection, rayStartingPoint, &mesh);

	}
}
__global__ void CastRaysPerspective(
		float3 cameraCenter,
		float nearDistance, float farDistance,
		float3 xFarOffset, float3 yFarOffset,
		float3 forward,
		int width, int height,
		int * colorMap, DeviceMeshData mesh)
{
	const unsigned int threadX = threadIdx.x;
	const unsigned int threadY = threadIdx.y;
	const unsigned int blockX = blockIdx.x;
	const unsigned int blockY = blockIdx.y;
//	if(threadX + blockX == 0 && threadY+blockY == 0)
//	{
//		printf("c_triangles[012] = (%s, %s, %s)\n", c_triangles[0], c_triangles[1], c_triangles[2]);
//	}

	const unsigned int mapIndexX = blockX*blockDim.x+threadX;
	const unsigned int mapIndexY = blockY*blockDim.y+threadY;
	if(mapIndexX < width && mapIndexY < height)
	{
		const unsigned int mapIndex = mapIndexY*width+mapIndexX;
		float3 rayStartingPoint = cameraCenter - forward*nearDistance;
		float3 rayDirection = normalize(forward * farDistance
				-xFarOffset*(mapIndexX-width/2.f)
				+yFarOffset*(mapIndexY-height/2.f));

		colorMap[mapIndex] = GetColorOfClosestHitpoint(rayDirection, rayStartingPoint, &mesh);

	}
}
//based on http://geomalgorithms.com/a06-_intersect-2.html#intersect3D_RayTriangle()
__device__ unsigned int GetColorOfClosestHitpoint(float3 &  ray, float3 &  rayStartingPoint,
		DeviceMeshData * p_mesh)
{
	int trianglesCount = p_mesh->trianglesLength/3;
	int idOfClosest;
	float3 closestTUVCoords = make_float3(INFINITY, 0,0);
	for(int i = 0; i < trianglesCount; i++)
	{
		int triangleId = i;
		triangleId*=3;
		float3 p1 = c_vertices[c_triangles[triangleId]];
		float3 p2 = c_vertices[c_triangles[triangleId+1]];
		float3 p3 = c_vertices[c_triangles[triangleId+2]];
		float3 tuv;
		if(GetIntersectionPointWith(ray, rayStartingPoint,
				p1, p2, p3, &tuv))
		{
			if(tuv.x<closestTUVCoords.x)
			{
				closestTUVCoords = tuv;
				idOfClosest = triangleId;
			}
		}
	}
	if(closestTUVCoords.x == INFINITY)
	{
		return 0xFF5555FF;
	}
	else
	{
		int l = 1-closestTUVCoords.y-closestTUVCoords.z;
//		float3 hitPointNormal = l*c_normals[c_triangles[idOfClosest]];
//
//		hitPointNormal +=closestTUVCoords.y*c_normals[c_triangles[idOfClosest+1]];
//
//		hitPointNormal += closestTUVCoords.z*c_normals[c_triangles[idOfClosest+2]];

		float3 hitPoint = l*c_vertices[c_triangles[idOfClosest]];
		hitPoint += closestTUVCoords.y*c_vertices[c_triangles[idOfClosest+1]];
		hitPoint += closestTUVCoords.z*c_vertices[c_triangles[idOfClosest+2]];

		float3 hitPointNormal = normalize(cross(
				c_vertices[c_triangles[idOfClosest+1]]-c_vertices[c_triangles[idOfClosest]],
				c_vertices[c_triangles[idOfClosest+2]]-c_vertices[c_triangles[idOfClosest]]
				                                                  ));
		float3 toLight = normalize(lightPos - hitPoint);
		unsigned int color = CalculateLight(toLight, -ray, hitPointNormal, 0.3f, 0.7f, 80);
		return color;

	}
}
__device__ unsigned int CalculateLight(float3 toLight, float3 toObserver, float3 &  normalVector,
		float diffuseFactor, float specularFactor, int m)
{
	float3 reflectVector = 2*dot(toLight, normalVector)*normalVector-toLight;
	float firstDot = dot(toLight,normalVector);
	float secondDot = dot(reflectVector, toObserver);
	secondDot = powf(secondDot, m);
	float3 floatColor = clamp(lightColor*(diffuseFactor*firstDot+specularFactor*secondDot),zero, one);
	unsigned int res =
			255u <<24 |
			((unsigned int)(floatColor.x*((objectColor>>16)&255))<<16) |
			((unsigned int)(floatColor.y*((objectColor>>8)&255))<<8) |
			(unsigned int)(floatColor.z*(objectColor&255));
	return res;

}

inline __device__ bool GetIntersectionPointWith(float3 &  ray, float3 &  rayStartingPoint,
		float3 &  v0, float3 &  v1, float3 &  v2, float3 * tuv)
{
	float3 v0v1 = v1-v0;
	float3 v0v2 = v2-v0;
	float3 pvec = cross(ray,v0v2);
	float det = dot(v0v1,pvec);
	if(det<0.01f)
	{
		return false;
	}
	float invDet = 1.0f / det;
	float3 tvec = rayStartingPoint-v0;
	float3 qvec = cross(tvec, v0v1);

	*tuv = make_float3(dot(v0v2,qvec),dot(tvec,pvec), dot(ray, qvec));
	*tuv *= invDet;
	return !(tuv->y < 0 || tuv->y > 1 || tuv->z < 0 || tuv->y + tuv->z > 1);
}
