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
#include "DisplayCalculatorKernels.h"
#include "mat4x4.h"

Mesh::Mesh(bool onCPU) {
	worldMatrix = getIdentityMatrix();
	this->onCPU = onCPU;

}

Mesh::~Mesh() {
	if(points != nullptr)
	{
		delete[] points;
		if(onCPU)
			delete[] this->cpu_points_transformed;
	}
	if(triangles != nullptr)
	{
		delete[] triangles;
	}
	if(this->d_points != nullptr)
	{
		cudaFree(d_points);
	}
	if(this->d_points_transformed != nullptr)
	{
		cudaFree(d_points_transformed);
	}
	if(this->d_triangles != nullptr)
	{
		cudaFree(d_triangles);
	}

}
void Mesh::SetPoints(float3 points[], int length)
{
	if(this->points != nullptr)
	{
		delete[] this->points;
		this->points = nullptr;
		if(onCPU)
		{
			delete[] this->cpu_points_transformed;
			this->cpu_points_transformed = nullptr;
		}
	}
	this->points = new float3[length];
	if(onCPU)
		this->cpu_points_transformed = new float3[length];
	for(int i = 0; i < length; i++)
	{
		this->points[i] = points[i];
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
		this->triangles = nullptr;
	}
	this->triangles = new int[length];
	for(int i = 0; i < length; i++)
	{
		if(pointsLength <= triangles[i])
			throw "Incorrect triangle: points array doesn't have this many numbers";
		this->triangles[i] = triangles[i];
	}
	trianglesLength = length;
	if(onCPU) initialized = true;
}
void Mesh::CopyToDevice()
{
	if(onCPU)
		throw "You shouldn't do this on cpu";
	if(this->triangles == nullptr)
	{
		throw "Initialize triangles, points  first";
	}
	if(this->d_points != nullptr)
	{
		cudaFree(d_points);
		d_points = nullptr;
	}
	if(this->d_points_transformed != nullptr)
	{
		cudaFree(d_points_transformed);
		d_points_transformed = nullptr;
	}
	if(this->d_triangles != nullptr)
	{
		cudaFree(d_triangles);
		d_triangles = nullptr;
	}
	cudaMalloc(&d_points, sizeof(float3)*pointsLength);
	getLastCudaError("couldn't malloc d_points in mesh");

	cudaMalloc(&d_points_transformed, sizeof(float4)*pointsLength);
	getLastCudaError("couldn't malloc d_points_transformed in mesh");

	cudaMalloc(&d_triangles, sizeof(int)*trianglesLength);
	getLastCudaError("couldn't malloc d_triangles in mesh");

	cudaMemcpy(d_points, points, sizeof(float3)*pointsLength, cudaMemcpyHostToDevice);
	getLastCudaError("couldn't memcpy points in mesh");

	cudaMemcpy(d_triangles, triangles, sizeof(int)*trianglesLength, cudaMemcpyHostToDevice);
	getLastCudaError("couldn't memcpy triangles in mesh");

	SaveToConstantMemory();
	getLastCudaError("Couldn't save data to constant for Mesh");
	initialized = true;
}
void Mesh::UpdateMeshVertices()
{
	if(onCPU)
	{
		worldMatrix.multiplyAllCPUVectors(points, cpu_points_transformed, pointsLength);
	}
	else
	{
		worldMatrix.multiplyAllVectors(d_points, d_points_transformed, pointsLength);
	}
}

bool Mesh::IsInitialized() {
	return initialized;
}

DeviceMeshData Mesh::GetDeviceMeshData() {
	if(onCPU)
		throw "You shouldn't do this on cpu";
	DeviceMeshData res;
	res.pointsLength = pointsLength;
	res.trianglesLength = trianglesLength;
	res.d_points = d_points_transformed;
	res.d_triangles = d_triangles;

	return res;
}
void Mesh::SetWorldMatrix(mat4x4 matrix)
{
	worldMatrix = matrix;
}
