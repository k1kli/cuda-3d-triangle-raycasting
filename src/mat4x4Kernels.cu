
/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include "mat4x4Kernels.h"
#include "mat4x4.h"



__device__ __constant__ float4 c_matrix[4];
void SetMatrix(const mat4x4 & h_matrix)
{
	float4 rows[4]{h_matrix[0],h_matrix[1],h_matrix[2],h_matrix[3]};

	cudaMemcpyToSymbol(c_matrix, rows, sizeof(float4)*4,0,cudaMemcpyHostToDevice);
}


__global__ void multiplyAllPointKernel(float3 * vectors_in, float4 * vectors_out, int length)
{
	const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index < length)
	{
		float4 mulVec = make_float4(vectors_in[index], 1.0f);
		vectors_out[index] = make_float4(
				dot(c_matrix[0], mulVec),
				dot(c_matrix[1], mulVec),
				dot(c_matrix[2], mulVec),
				0
			);
	}
}

