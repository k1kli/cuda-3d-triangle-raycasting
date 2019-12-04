################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/DisplayCalculator.cu \
../src/DisplayCalculatorKernels.cu \
../src/Mesh.cu 

CPP_SRCS += \
../src/main.cpp 

OBJS += \
./src/DisplayCalculator.o \
./src/DisplayCalculatorKernels.o \
./src/Mesh.o \
./src/main.o 

CU_DEPS += \
./src/DisplayCalculator.d \
./src/DisplayCalculatorKernels.d \
./src/Mesh.d 

CPP_DEPS += \
./src/main.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-10.1/bin/nvcc -I../include/cudaInclude -I../include/GL -G -g -O0 -maxrregcount 32 -gencode arch=compute_61,code=sm_61  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-10.1/bin/nvcc -I../include/cudaInclude -I../include/GL -G -g -O0 -maxrregcount 32 --compile --relocatable-device-code=false -gencode arch=compute_61,code=compute_61 -gencode arch=compute_61,code=sm_61  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-10.1/bin/nvcc -I../include/cudaInclude -I../include/GL -G -g -O0 -maxrregcount 32 -gencode arch=compute_61,code=sm_61  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-10.1/bin/nvcc -I../include/cudaInclude -I../include/GL -G -g -O0 -maxrregcount 32 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


