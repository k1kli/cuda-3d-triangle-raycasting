################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/CPU/mainCPU.cpp 

OBJS += \
./src/CPU/mainCPU.o 

CPP_DEPS += \
./src/CPU/mainCPU.d 


# Each subdirectory must supply rules for building sources it contributes
src/CPU/%.o: ../src/CPU/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-10.1/bin/nvcc -I../include/cudaInclude -I../include/GL -G -g -O0 -maxrregcount 32 -gencode arch=compute_61,code=sm_61  -odir "src/CPU" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-10.1/bin/nvcc -I../include/cudaInclude -I../include/GL -G -g -O0 -maxrregcount 32 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


