/*
 * Mesh.cpp
 *
 *  Created on: 27 lis 2019
 *      Author: karol
 */

#include "Mesh.h"
#include <cuda_runtime.h>
#include <helper_math.h>
#include <helper_cuda.h>

Mesh::Mesh() {
	// TODO Auto-generated constructor stub

}

Mesh::~Mesh() {
	if(points != nullptr)
	{
		delete[] points;
	}
	if(triangles != nullptr)
	{
		delete[] triangles;
	}
	if(normals != nullptr)
	{
		delete[] normals;
	}
	if(triangleNormals != nullptr)
	{
		delete[] triangleNormals;
	}

	if(d_points != nullptr)
	{
		cudaFree(d_points);
	}
	if(d_triangles != nullptr)
	{
		cudaFree(d_triangles);
	}
	if(d_normals != nullptr)
	{
		cudaFree(d_normals);
	}
	if(d_triangleNormals != nullptr)
	{
		cudaFree(d_triangleNormals);
	}

}
void Mesh::SetPoints(float3 points[], int length)
{
	if(this->points != nullptr)
	{
		delete[] this->points;
	}
	this->points = new float4[length];
	for(int i = 0; i < length; i++)
	{
		this->points[i] = make_float4(points[i], 0);
	}
	pointsLength = length;
	initialized = false;
}
void Mesh::SetTriangles(int triangles[], int length)
{
	if(this->points == nullptr)
	{
		throw "Initialize points first";
	}
	if(length % 3 != 0)
	{
		throw "length of triangles array must be divisble by 3";
	}
	if(this->triangles != nullptr)
	{
		delete[] this->triangles;
	}
	this->triangles = new int[length];
	for(int i = 0; i < length; i++)
	{
		if(pointsLength <= triangles[i])
			throw "Incorrect triangle: points array doesn't have this many numbers";
		this->triangles[i] = triangles[i];
	}
	trianglesLength = length;
}
void Mesh::RecalculateNormals()
{
	if(this->triangles == nullptr)
	{
		throw "Initialize triangles and points first";
	}
	if(this->triangleNormals != nullptr)
	{
		delete[] this->triangleNormals;
	}
	if(this->normals != nullptr)
	{
		delete[] this->normals;
	}
	triangleNormals = new float4[trianglesLength/3];
	normals = new float4[pointsLength];
	for(int i = 0; i < pointsLength; i++)
	{
		normals[i]=make_float4(0, 0, 0, 0);
	}
	for(int i = 0; i < trianglesLength/3; i++)
	{
		float4 u = points[triangles[i*3+1]]-points[triangles[i*3]];
		float4 v = points[triangles[i*3+2]] - points[triangles[i*3]];
		triangleNormals[i] = make_float4(cross(make_float3(u), make_float3(v)),0);
		normals[triangles[i*3]] += triangleNormals[i];
		normals[triangles[i*3+1]] += triangleNormals[i];
		normals[triangles[i*3+2]] += triangleNormals[i];
	}
	for(int i = 0; i < pointsLength; i++)
	{
		normals[i] = make_float4(normalize(make_float3(normals[i])), 0);
	}
}
void Mesh::CopyToDevice()
{
	if(this->triangleNormals == nullptr)
	{
		throw "Initialize triangles, points and recalculate normals first";
	}

	cudaMalloc(&d_points,sizeof(float4)*pointsLength);
	getLastCudaError("Couldn't malloc d_points for Mesh");
	cudaMemcpy(d_points, points, sizeof(float4)*pointsLength, cudaMemcpyHostToDevice);
	getLastCudaError("Couldn't memcpy points to device for Mesh");

	cudaMalloc(&d_triangles,sizeof(int)*trianglesLength);
	getLastCudaError("Couldn't malloc d_triangles for Mesh");
	cudaMemcpy(d_triangles, triangles, sizeof(int)*trianglesLength, cudaMemcpyHostToDevice);
	getLastCudaError("Couldn't memcpy triangles to device for Mesh");

	cudaMalloc(&d_triangleNormals,sizeof(float4)*trianglesLength/3);
	getLastCudaError("Couldn't malloc d_triangleNormals for Mesh");
	cudaMemcpy(d_triangleNormals, triangleNormals, sizeof(float4)*trianglesLength/3, cudaMemcpyHostToDevice);
	getLastCudaError("Couldn't memcpy triangleNormals to device for Mesh");

	cudaMalloc(&d_normals,sizeof(float4)*pointsLength);
	getLastCudaError("Couldn't malloc d_normals for Mesh");
	cudaMemcpy(d_normals, normals, sizeof(float4)*pointsLength, cudaMemcpyHostToDevice);
	getLastCudaError("Couldn't memcpy normals to device for Mesh");
	initialized = true;
}

bool Mesh::IsInitialized() {
	return initialized;
}

DeviceMeshData* Mesh::GetDeviceMeshDataPointer() {
	DeviceMeshData hostRes;
	hostRes.d_points = d_points;
	hostRes.d_triangles = d_triangles;
	hostRes.d_normals = d_normals;
	hostRes.d_triangleNormals = d_triangleNormals;
	hostRes.pointsLength = pointsLength;
	hostRes.trianglesLength = trianglesLength;
	DeviceMeshData * res;
	cudaMalloc(&res,sizeof(DeviceMeshData));
	getLastCudaError("Couldn't malloc DeviceMeshData for Mesh");

	cudaMemcpy(res,&hostRes, sizeof(DeviceMeshData), cudaMemcpyHostToDevice);
	getLastCudaError("Couldn't memcpy DeviceMeshData to device for Mesh");

	return res;
}
