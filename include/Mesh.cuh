/*
 * Mesh.h
 *
 *  Created on: 27 lis 2019
 *      Author: karol
 */

#ifndef MESH_H_
#define MESH_H_
#include "cuda_runtime_api.h"
#include "mat4x4.cuh"
#include "Material.cuh"

struct DeviceMeshData
{
	int pointsLength;
	int trianglesLength;
	float4 * d_points;
	int * d_triangles;
};
class Mesh {
	float3 * points = nullptr;
	float3 * cpu_points_transformed = nullptr;
	int * triangles = nullptr;
	bool initialized = false;
	float3 * d_points = nullptr;
	float4 * d_points_transformed = nullptr;
	int * d_triangles = nullptr;
	mat4x4 worldMatrix;
	bool onCPU;
public:
	int pointsLength = 0;
	int trianglesLength = 0;
	Material material;
	Mesh(bool onCPU = false);
	virtual ~Mesh();
	void SetPoints(float3 * points, int length);
	void SetTriangles(int * triangles, int length);
	void CopyToDevice();
	void SetWorldMatrix(mat4x4 matrix);
	bool IsInitialized();
	void UpdateMeshVertices();
	DeviceMeshData GetDeviceMeshData();
	friend class DisplayCalculator;
};

#endif /* MESH_H_ */
