/*
 * Mesh.h
 *
 *  Created on: 27 lis 2019
 *      Author: karol
 */

#ifndef MESH_H_
#define MESH_H_
#include "cuda_runtime_api.h"

struct DeviceMeshData
{
	float4 * d_points;
	int * d_triangles;
	float4 * d_normals;
	float4 * d_triangleNormals;
	int pointsLength;
	int trianglesLength;
};
class Mesh {
	float4 * points;
	int * triangles;
	float4 * normals;
	float4 * triangleNormals;
	float4 * d_points;
	int * d_triangles;
	float4 * d_normals;
	float4 * d_triangleNormals;
	int pointsLength = 0;
	int trianglesLength = 0;
	bool initialized = false;
public:
	Mesh();
	virtual ~Mesh();
	void SetPoints(float3 * points, int length);
	void SetTriangles(int * triangles, int length);
	void RecalculateNormals();
	void CopyToDevice();
	bool IsInitialized();
	DeviceMeshData * GetDeviceMeshDataPointer();
	friend class DisplayCalculator;
};

#endif /* MESH_H_ */
