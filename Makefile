###########################################################

## USER SPECIFIC DIRECTORIES ##

# CUDA directory:
CUDA_ROOT_DIR=/usr/local/cuda-10.1

##########################################################

## CC COMPILER OPTIONS ##

# CC compiler options:
CC=g++
CC_FLAGS=  -DGLEW_STATIC
CC_LIBS=

##########################################################

## NVCC COMPILER OPTIONS ##

# NVCC compiler options:
NVCC=$(CUDA_ROOT_DIR)/bin/nvcc
NVCC_FLAGS= -Ilibraries/cudaInclude -Ilibraries/GL -G -g -O0 -maxrregcount 32 --generate-code arch=compute_30,code=compute_30
NVCC_LIBS=

# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart  -lGL -lGLEW -lGLU -lglut

##########################################################

## Project file structure ##

# Source file directory:
SRC_DIR = src

# Object file directory:
OBJ_DIR = bin

# Include header file diretory:
INC_DIR = include

##########################################################

## Make variables ##
all: triangle-raycasting-3d

# Target executable name:
EXE = triangle-raycasting-3d

# Object files:
OBJS = $(OBJ_DIR)/DisplayCalculator.o $(OBJ_DIR)/DisplayCalculatorKernels.o  $(OBJ_DIR)/Material.o $(OBJ_DIR)/Mesh.o $(OBJ_DIR)/SceneData.o $(OBJ_DIR)/main.o $(OBJ_DIR)/mat4x4.o  $(OBJ_DIR)/mat4x4Kernels.o  $(OBJ_DIR)/Light.o $(OBJ_DIR)/mainCPU.o  $(OBJ_DIR)/mainGPU.o 

##########################################################

## Compile ##

# Link c++ and CUDA compiled object files to target executable:
$(EXE) : $(OBJS)
	$(CC) $(CC_FLAGS) $(OBJS) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

# Compile main .cpp file to object files:
$(OBJ_DIR)/%.o : %.cpp
	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile C++ source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp include/%.h
	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile CUDA source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu $(INC_DIR)/%.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

# Clean objects in object directory.
clean:
	$(RM) bin/* *.o $(EXE)
