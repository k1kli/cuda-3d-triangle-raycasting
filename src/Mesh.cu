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
	if(normals != nullptr)
	{
		delete[] normals;
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
void Mesh::SetNormals(float3 normals[])
{
	if(this->triangles == nullptr)
	{
		throw "Initialize triangles and points first";
	}
	if(this->normals != nullptr)
	{
		delete[] this->normals;
	}
	normals = new float3[pointsLength];
	for(int i = 0; i < pointsLength; i++)
	{
		this->normals[i]=normals[i];
	}
}
void Mesh::RecalculateNormals()
{
	if(this->triangles == nullptr)
	{
		throw "Initialize triangles and points first";
	}
	if(this->normals != nullptr)
	{
		delete[] this->normals;
	}
	normals = new float3[pointsLength];
	for(int i = 0; i < pointsLength; i++)
	{
		normals[i]=make_float3(0, 0, 0);
	}
	for(int i = 0; i < trianglesLength/3; i++)
	{
		float3 u = points[triangles[i*3+1]]-points[triangles[i*3]];
		float3 v = points[triangles[i*3+2]] - points[triangles[i*3]];
		float3 w = points[triangles[i*3+2]] - points[triangles[i*3+1]];
		float3 triangleNormal = normalize(cross(u, v));
		printf("for vertex (%f, %f, %f) (%f, %f, %f) (%f, %f, %f) face normal is (%f, %f, %f)\n",
				points[triangles[i*3]].x, points[triangles[i*3]].y, points[triangles[i*3]].z,
				points[triangles[i*3+1]].x, points[triangles[i*3+1]].y, points[triangles[i*3+1]].z,
				points[triangles[i*3+2]].x, points[triangles[i*3+2]].y, points[triangles[i*3+2]].z,
				triangleNormal.x, triangleNormal.y, triangleNormal.z);
		normals[triangles[i*3]] += triangleNormal*asin(length(cross(normalize(u),normalize(v))));
		normals[triangles[i*3+1]] += triangleNormal*asin(length(cross(normalize(-u),normalize(w))));
		normals[triangles[i*3+2]] += triangleNormal*asin(length(cross(normalize(-v),normalize(-w))));
	}
	for(int i = 0; i < pointsLength; i++)
	{
		printf("for vertex (%f, %f, %f) normal is (%f, %f, %f)\n", points[i].x, points[i].y, points[i].z,normals[i].x, normals[i].y, normals[i].z);
		normals[i] = normalize(normals[i]);
	}
	printf("after normalization\n");
	for(int i = 0; i < pointsLength; i++)
	{
		printf("for vertex (%f, %f, %f) normal is (%f, %f, %f)\n", points[i].x, points[i].y, points[i].z,normals[i].x, normals[i].y, normals[i].z);
	}
}
void Mesh::CopyToDevice()
{
	if(this->normals == nullptr)
	{
		throw "Initialize triangles, points and recalculate normals first";
	}
	SaveToConstantMemory(points, normals, triangles, pointsLength, trianglesLength);
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
