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
	int pointsLength;
	int trianglesLength;
};
class Mesh {
	float3 * points;
	short * triangles;
	float3 * normals;
	bool initialized = false;
public:
	int pointsLength = 0;
	int trianglesLength = 0;
	Mesh();
	virtual ~Mesh();
	void SetPoints(float3 * points, int length);
	void SetTriangles(int * triangles, int length);
	void SetNormals(float3 normals[]);
	void RecalculateNormals();
	void CopyToDevice();
	bool IsInitialized();
	DeviceMeshData GetDeviceMeshData();
	friend class DisplayCalculator;
};

#endif /* MESH_H_ */
