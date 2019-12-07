
/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

#ifndef MAT4X4KERNELS_H_
#define MAT4X4KERNELS_H_
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include "mat4x4.h"

void SetMatrix(const mat4x4 & h_matrix);


__global__ void multiplyAllPointKernel(float3 * vectors_in, float3 * vectors_out, int length);

#endif /* MAT4X4KERNELS_H_ */
