
// includes, system
#include "string.h"

#include "include/mainCPU.cuh"
#include "include/mainGPU.cuh"


int main(int argc, char **argv)
{
	if(argc > 1 && 0 == strcmp(argv[1], "CPU"))
	{
		CPU::mainCPU(argc, argv);
	}
	else
	{
		GPU::mainGPU(argc, argv);
	}
	return 0;
}
