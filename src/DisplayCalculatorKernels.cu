#include "DisplayCalculatorKernels.h"
#include "cuda_runtime.h"
#include "Mesh.h"
#include "stdio.h"

__global__ void CastRaysOrthogonal(
		float4 cameraTopLeftCorner, float4 rayDirection, float4 up,
		float cameraFovWidth, float cameraFovHeight,
		int width, int height,
		int * colorMap, DeviceMeshData * mesh)
{
	const unsigned int threadX = threadIdx.x;
	const unsigned int threadY = threadIdx.y;
	const unsigned int blockX = blockIdx.x;
	const unsigned int blockY = blockIdx.y;

	const unsigned int mapIndexX = blockX*blockDim.x+threadX;
	const unsigned int mapIndexY = blockY*blockDim.y+threadY;
	if(mapIndexX < width && mapIndexY < height)
	{
		const unsigned int mapIndex = mapIndexY*width+mapIndexX;
		unsigned int color = 0xFF000000;
		int xVal = (int)((((float)mapIndexX) / width) * 255);
		int yVal = (int)((((float)mapIndexY) / height) * 255);
		color += xVal << 16;
		color += yVal<<8;
		colorMap[mapIndex]=color;
	}
}
