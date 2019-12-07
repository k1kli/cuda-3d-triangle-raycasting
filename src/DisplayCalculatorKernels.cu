#include "DisplayCalculatorKernels.h"
#include "cuda_runtime.h"
#include "Mesh.h"
#include "stdio.h"
#include <helper_math.h>
#include "defines.h"


__device__ __constant__ float3 c_vertices[4096];//48kb
__device__ __constant__ short c_triangles[6*1024];//12kb
__device__ __constant__ float3 lightColor;
__device__ __constant__ float3 lightPos;
__device__ __constant__ uint objectColor = 0xFFFFFF;
__device__ __constant__ float3 zero;
__device__ __constant__ float3 one;
__device__ __constant__ float3 toObserver;
__device__ __constant__ float3 ray;

void SaveToConstantMemory(short * h_triangles, int verticesLenght, int trianglesLength)
{

	cudaMemcpyToSymbol(c_triangles, h_triangles, sizeof(short)*trianglesLength,0,cudaMemcpyHostToDevice);

	float3 h_lightColor = make_float3(1.0f,1.0f,1.0f);

	cudaMemcpyToSymbol(lightColor, &h_lightColor, sizeof(float3),0,cudaMemcpyHostToDevice);

	float3 h_zero = make_float3(0.0f,0.0f,0.0f);

	cudaMemcpyToSymbol(zero, &h_zero, sizeof(float3),0,cudaMemcpyHostToDevice);

	float3 h_one = make_float3(1.0f,1.0f,1.0f);

	cudaMemcpyToSymbol(one, &h_one, sizeof(float3),0,cudaMemcpyHostToDevice);

	float3 h_lightPos = make_float3(-2.0f,3.0f,-2.0f);

	cudaMemcpyToSymbol(lightPos, &h_lightPos, sizeof(float3),0,cudaMemcpyHostToDevice);

	float3 h_toObserver = make_float3(0.0f, 0.0f, -1.0f);

	cudaMemcpyToSymbol(toObserver, &h_toObserver, sizeof(float3),0,cudaMemcpyHostToDevice);

	float3 h_ray = make_float3(0.0f, 0.0f, 1.0f);

	cudaMemcpyToSymbol(ray, &h_ray, sizeof(float3),0,cudaMemcpyHostToDevice);
}
void SaveVerticesToConstantMemory(float3 * d_vertices, int length)
{
	cudaMemcpyToSymbol(c_vertices, d_vertices, sizeof(float3)*length,0,cudaMemcpyDeviceToDevice);
}

__device__ unsigned int GetColorOfClosestHitpoint(float3 &  rayStartingPoint,
		DeviceMeshData * p_mesh, bool * reachableTriangles);


__device__ bool RayIntersectsWith(float3 &  rayStartingPoint,
		float3 & v1, float3 &  v2, float3 &  v3);


__device__ unsigned int CalculateLight(float3 toLight, float3 &  normalVector,
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
	extern __shared__ bool reachableTriangles[];
	{
		float3 blockStart = cameraBottomLeftCorner + xOffset*blockX*blockDim.x + yOffset*blockY*blockDim.y;
		float3 blockEnd = blockStart + xOffset*blockDim.x + yOffset*blockDim.y;
		float blockMinX = blockStart.x;
		float blockMinY = blockStart.y;
		float blockMaxX =  blockEnd.x;
		float blockMaxY = blockEnd.y;
		for(int i = 0; i < mesh.trianglesLength/3; i+=blockDim.x*blockDim.y)
		{
			int triangleId = (i+threadX+blockDim.x*threadY)*3;
			if(triangleId < mesh.trianglesLength)
			{
				float3 p1 = c_vertices[c_triangles[triangleId]];
				float3 p2 = c_vertices[c_triangles[triangleId+1]];
				float3 p3 = c_vertices[c_triangles[triangleId+2]];
				float minX = MIN(p1.x, MIN(p2.x,p3.x));
				float maxX = MAX(p1.x, MAX(p2.x,p3.x));
				float minY = MIN(p1.y, MIN(p2.y,p3.y));
				float maxY = MAX(p1.y, MAX(p2.y,p3.y));
				reachableTriangles[triangleId] = true;
				if(!(minX <= blockMaxX && minY <= blockMaxY && maxX >= blockMinX && maxY >= blockMinY))
				{
					reachableTriangles[triangleId] = false;
				}
			}
		}
		__syncthreads();
	}
	if(mapIndexX < width && mapIndexY < height)
	{
		const unsigned int mapIndex = mapIndexY*width+mapIndexX;
		float3 rayStartingPoint = cameraBottomLeftCorner
				+xOffset*mapIndexX
				+yOffset*mapIndexY;
		colorMap[mapIndex] = GetColorOfClosestHitpoint(rayStartingPoint, &mesh, reachableTriangles);

	}
}
//based on http://geomalgorithms.com/a06-_intersect-2.html#intersect3D_RayTriangle()
__device__ unsigned int GetColorOfClosestHitpoint(float3 &  rayStartingPoint,
		DeviceMeshData * p_mesh, bool * reachableTriangles)
{
	float closestDistance = INFINITY;
	float3 closestHitPointNormal;
	for(int triangleId = 0; triangleId < p_mesh->trianglesLength; triangleId+=3)
	{
		if(!reachableTriangles[triangleId])
			continue;
		float3 p1 = c_vertices[c_triangles[triangleId]];
		float3 p2 = c_vertices[c_triangles[triangleId+1]];
		float3 p3 = c_vertices[c_triangles[triangleId+2]];
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
		return 0xFF5555FF;
	}
	else
	{

		float3 hitPoint = make_float3(rayStartingPoint.x, rayStartingPoint.y, rayStartingPoint.z+closestDistance);


		float3 toLight = normalize(lightPos - hitPoint);
		unsigned int color = CalculateLight(toLight, closestHitPointNormal, 0.3f, 0.7f, 80);
		return color;

	}
}
__device__ unsigned int CalculateLight(float3 toLight, float3 &  normalVector,
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

inline __device__ bool RayIntersectsWith(float3 & rayStartingPoint,
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
