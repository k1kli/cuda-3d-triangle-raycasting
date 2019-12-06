/*
 * mat4x4.h
 *
 *  Created on: 6 gru 2019
 *      Author: karol
 */

#ifndef MAT4X4_H_
#define MAT4X4_H_

#include "cuda_runtime.h"
#include <helper_math.h>
#include <iostream>

class mat4x4 {
private:
	float4 rows[4];
public:
	mat4x4(float initVal = 0.0f);
	mat4x4(const float4 & row1, const float4 & row2, const float4 & row3, const float4 & row4);
	virtual ~mat4x4();

	float4 & operator[](int i);
	const float4 & operator[](int i) const;
	mat4x4 getInverse() const;
	float * vals() const;
	float4 getColumn(int i) const;
	void multiplyAllVectors(float3 * d_vectors_in, float3 * d_vectors_out, int size);


	friend float4 operator*(const mat4x4 & matrix, const float4 & vector);
	friend mat4x4 operator+(const mat4x4 & matrix1, const mat4x4 & matrix2);
	friend mat4x4 operator-(const mat4x4 & matrix1, const mat4x4 & matrix2);
	friend mat4x4 operator*(const mat4x4 & matrix1, const mat4x4 & matrix2);
	friend std::ostream & operator<<(std::ostream & out, const mat4x4 & matrix);
	friend mat4x4 getRotationMatrix(float aroundXRotation,float aroundYRotation,float aroundZRotation);
};
mat4x4 getIdentityMatrix();
mat4x4 getRotationMatrix(float aroundXRotation,float aroundYRotation,float aroundZRotation);

#endif /* MAT4X4_H_ */
