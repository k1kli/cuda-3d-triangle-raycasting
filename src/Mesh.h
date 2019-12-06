/*
 * Mesh.h
 *
 *  Created on: 27 lis 2019
 *      Author: karol
 */

#ifndef MESH_H_
#define MESH_H_
#include "cuda_runtime_api.h"
#include "mat4x4.h"

struct DeviceMeshData
{
	int pointsLength;
	int trianglesLength;
};
class Mesh {
	float3 * points;
	short * triangles;
	bool initialized = false;
	float3 * d_points;
	float3 * d_points_transformed;
	mat4x4 worldMatrix;
public:
	int pointsLength = 0;
	int trianglesLength = 0;
	Mesh();
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
