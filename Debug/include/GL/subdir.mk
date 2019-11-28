################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../include/GL/glew.c 

OBJS += \
./include/GL/glew.o 

C_DEPS += \
./include/GL/glew.d 


# Each subdirectory must supply rules for building sources it contributes
include/GL/%.o: ../include/GL/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-10.1/bin/nvcc -I../include/cudaInclude -I../include/GL -G -g -O0 -gencode arch=compute_61,code=sm_61  -odir "include/GL" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-10.1/bin/nvcc -I../include/cudaInclude -I../include/GL -G -g -O0 --compile  -x c -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


