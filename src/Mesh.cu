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

}
void Mesh::SetPoints(float3 points[], int length)
{
	if(this->points != nullptr)
	{
		delete[] this->points;
	}
	this->points = new float3[length];
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
	}
	this->triangles = new short[length];
	for(int i = 0; i < length; i++)
	{
		if(pointsLength <= triangles[i])
			throw "Incorrect triangle: points array doesn't have this many numbers";
		this->triangles[i] = (short)triangles[i];
	}
	trianglesLength = length;
}
void Mesh::CopyToDevice()
{
	if(this->triangles == nullptr)
	{
		throw "Initialize triangles, points  first";
	}
	SaveToConstantMemory(points, triangles, pointsLength, trianglesLength);
	getLastCudaError("Couldn't memcpy data to constant for Mesh");
	initialized = true;
}

bool Mesh::IsInitialized() {
	return initialized;
}

DeviceMeshData Mesh::GetDeviceMeshData() {
	DeviceMeshData res;
	res.pointsLength = pointsLength;
	res.trianglesLength = trianglesLength;

	return res;
}
