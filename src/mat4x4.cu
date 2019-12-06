/*
 * mat4x4.cpp
 *
 *  Created on: 6 gru 2019
 *      Author: karol
 */

#include "mat4x4.h"
#include "cuda_runtime.h"
#include <helper_math.h>
#include <helper_cuda.h>
#include <iostream>
#include "defines.h"
#include "mat4x4Kernels.h"

mat4x4::mat4x4(float initVal) {
	for(int i = 0; i < 4; i++)
	{
		rows[i] = make_float4(initVal, initVal, initVal, initVal);
	}
}


mat4x4::mat4x4(const float4& row1, const float4& row2, const float4& row3,
		const float4& row4) {
	rows[0] = row1;
	rows[1] = row2;
	rows[2] = row3;
	rows[3] = row4;
}

mat4x4::~mat4x4() {
}

const float4& mat4x4::operator [](int i) const {
	return rows[i];
}

float4& mat4x4::operator [](int i) {
	return rows[i];
}

void mat4x4::multiplyAllVectors(float3 * d_vectors_in, float3 * d_vectors_out, int size)
{
	SetMatrix(*this);
	getLastCudaError("setting matrix failed");
	int threads = 256;
	int blocks = DIVROUNDUP(size, threads);
	multiplyAllPointKernel<<<blocks, threads>>>(d_vectors_in, d_vectors_out, size);

	getLastCudaError("multiplyAllVectors kernel execution failed");
}

//https://stackoverflow.com/questions/1148309/inverting-a-4x4-matrix
mat4x4 mat4x4::getInverse()  const{
	float * m = vals();
	float inv[16], det;
	int i;

	inv[0] = m[5]  * m[10] * m[15] -
			 m[5]  * m[11] * m[14] -
			 m[9]  * m[6]  * m[15] +
			 m[9]  * m[7]  * m[14] +
			 m[13] * m[6]  * m[11] -
			 m[13] * m[7]  * m[10];

	inv[4] = -m[4]  * m[10] * m[15] +
			  m[4]  * m[11] * m[14] +
			  m[8]  * m[6]  * m[15] -
			  m[8]  * m[7]  * m[14] -
			  m[12] * m[6]  * m[11] +
			  m[12] * m[7]  * m[10];

	inv[8] = m[4]  * m[9] * m[15] -
			 m[4]  * m[11] * m[13] -
			 m[8]  * m[5] * m[15] +
			 m[8]  * m[7] * m[13] +
			 m[12] * m[5] * m[11] -
			 m[12] * m[7] * m[9];

	inv[12] = -m[4]  * m[9] * m[14] +
			   m[4]  * m[10] * m[13] +
			   m[8]  * m[5] * m[14] -
			   m[8]  * m[6] * m[13] -
			   m[12] * m[5] * m[10] +
			   m[12] * m[6] * m[9];

	inv[1] = -m[1]  * m[10] * m[15] +
			  m[1]  * m[11] * m[14] +
			  m[9]  * m[2] * m[15] -
			  m[9]  * m[3] * m[14] -
			  m[13] * m[2] * m[11] +
			  m[13] * m[3] * m[10];

	inv[5] = m[0]  * m[10] * m[15] -
			 m[0]  * m[11] * m[14] -
			 m[8]  * m[2] * m[15] +
			 m[8]  * m[3] * m[14] +
			 m[12] * m[2] * m[11] -
			 m[12] * m[3] * m[10];

	inv[9] = -m[0]  * m[9] * m[15] +
			  m[0]  * m[11] * m[13] +
			  m[8]  * m[1] * m[15] -
			  m[8]  * m[3] * m[13] -
			  m[12] * m[1] * m[11] +
			  m[12] * m[3] * m[9];

	inv[13] = m[0]  * m[9] * m[14] -
			  m[0]  * m[10] * m[13] -
			  m[8]  * m[1] * m[14] +
			  m[8]  * m[2] * m[13] +
			  m[12] * m[1] * m[10] -
			  m[12] * m[2] * m[9];

	inv[2] = m[1]  * m[6] * m[15] -
			 m[1]  * m[7] * m[14] -
			 m[5]  * m[2] * m[15] +
			 m[5]  * m[3] * m[14] +
			 m[13] * m[2] * m[7] -
			 m[13] * m[3] * m[6];

	inv[6] = -m[0]  * m[6] * m[15] +
			  m[0]  * m[7] * m[14] +
			  m[4]  * m[2] * m[15] -
			  m[4]  * m[3] * m[14] -
			  m[12] * m[2] * m[7] +
			  m[12] * m[3] * m[6];

	inv[10] = m[0]  * m[5] * m[15] -
			  m[0]  * m[7] * m[13] -
			  m[4]  * m[1] * m[15] +
			  m[4]  * m[3] * m[13] +
			  m[12] * m[1] * m[7] -
			  m[12] * m[3] * m[5];

	inv[14] = -m[0]  * m[5] * m[14] +
			   m[0]  * m[6] * m[13] +
			   m[4]  * m[1] * m[14] -
			   m[4]  * m[2] * m[13] -
			   m[12] * m[1] * m[6] +
			   m[12] * m[2] * m[5];

	inv[3] = -m[1] * m[6] * m[11] +
			  m[1] * m[7] * m[10] +
			  m[5] * m[2] * m[11] -
			  m[5] * m[3] * m[10] -
			  m[9] * m[2] * m[7] +
			  m[9] * m[3] * m[6];

	inv[7] = m[0] * m[6] * m[11] -
			 m[0] * m[7] * m[10] -
			 m[4] * m[2] * m[11] +
			 m[4] * m[3] * m[10] +
			 m[8] * m[2] * m[7] -
			 m[8] * m[3] * m[6];

	inv[11] = -m[0] * m[5] * m[11] +
			   m[0] * m[7] * m[9] +
			   m[4] * m[1] * m[11] -
			   m[4] * m[3] * m[9] -
			   m[8] * m[1] * m[7] +
			   m[8] * m[3] * m[5];

	inv[15] = m[0] * m[5] * m[10] -
			  m[0] * m[6] * m[9] -
			  m[4] * m[1] * m[10] +
			  m[4] * m[2] * m[9] +
			  m[8] * m[1] * m[6] -
			  m[8] * m[2] * m[5];

	det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

	if (det == 0)
		return *this;

	det = 1.0 / det;

	mat4x4 res;
	for (i = 0; i < 16; i++)
		res.vals()[i] = inv[i] * det;

	return res;
}

float * mat4x4::vals() const
{
	return (float *)rows;
}
float4 mat4x4::getColumn(int i) const
{
	float * values = vals();
	return make_float4(values[i],values[i+4],values[i+8],values[i+12]);
}


float4 operator*(const mat4x4 & matrix, const float4 & vector)
{
	float4 res;
	res.x = dot(matrix[0], vector);
	res.y = dot(matrix[1], vector);
	res.z = dot(matrix[2], vector);
	res.w = dot(matrix[3], vector);
	return res;
}
mat4x4 operator+(const mat4x4 & matrix1, const mat4x4 & matrix2)
{
	mat4x4 res;
	for(int i = 0; i < 16; i++)
	{
		res.vals()[i] =matrix1.vals()[i] + matrix2.vals()[i];
	}
	return res;
}
mat4x4 operator-(const mat4x4 & matrix1, const mat4x4 & matrix2)
{
	mat4x4 res;
	for(int i = 0; i < 16; i++)
	{
		res.vals()[i] =matrix1.vals()[i] - matrix2.vals()[i];
	}
	return res;
}
mat4x4 operator*(const mat4x4 & matrix1, const mat4x4 & matrix2)
{
	mat4x4 res;
	for(int i = 0; i < 4; i++)
	{
		res[i].x = dot(matrix1[i], matrix2.getColumn(0));
		res[i].y = dot(matrix1[i], matrix2.getColumn(1));
		res[i].z = dot(matrix1[i], matrix2.getColumn(2));
		res[i].w = dot(matrix1[i], matrix2.getColumn(3));
	}
	return res;
}

std::ostream & operator<<(std::ostream & out, const mat4x4 & matrix)
{
	for(int row = 0; row < 4; row++)
	{
		out<<matrix.rows[row].x<<"\t"<<matrix.rows[row].y<<"\t"<<matrix.rows[row].z<<"\t"<<matrix.rows[row].w<<std::endl;
	}
	return out;
}

mat4x4 getIdentityMatrix()
{
	mat4x4 res;
	res[0].x = 1.0f;
	res[1].y = 1.0f;
	res[2].z = 1.0f;
	res[3].w = 1.0f;
	return res;
}

mat4x4 getRotationMatrix(float aroundXRotation,float aroundYRotation,float aroundZRotation)
{
	mat4x4 rx = getIdentityMatrix(), ry = getIdentityMatrix(), rz = getIdentityMatrix();
	rx[1].y = rx[2].z = cos(aroundXRotation);
	rx[1].z = -sin(aroundXRotation);
	rx[2].y = -rx[1].z;

	ry[0].x = ry[2].z = cos(aroundYRotation);
	ry[2].x = -sin(aroundYRotation);
	ry[0].z = -ry[2].x;

	rz[0].x = rz[1].y = cos(aroundZRotation);
	rz[0].y = -sin(aroundZRotation);
	rz[1].x = -rz[0].y;
	return rx*ry*rz;
}
