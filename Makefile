# Compiler settings
CC = gcc
NVCC = nvcc
MPICC = mpicc
   
# Compiler flags
CFLAGS = -Wall -O2 -I./include
CFLAGS += -I$(CUDA_PATH)/include
NVCCFLAGS = -O2 -arch=sm_75
MPICFLAGS = -Wall -O2 -I./include
  
# CUDA paths (adjust these to match your CUDA installation)
CUDA_PATH = /usr/local/cuda
CUDA_INC_PATH = $(CUDA_PATH)/include
CUDA_LIB_PATH = $(CUDA_PATH)/lib64
  
# Include and library flags
INCLUDES = -I$(CUDA_PATH)/include -I./include
LFLAGS = -L$(CUDA_PATH)/lib64 -lcudart -lm -lstdc++
NVCC = nvcc
NVCCFLAGS = $(INCLUDES)
  
# Source files
CU_SOURCES = src/logbesselk.cu
EXAMPLE_SOURCES = $(wildcard examples/*.c)
  
# Object files
CU_OBJECTS = $(CU_SOURCES:.cu=.o)
  
# Example executables
EXAMPLE_EXECUTABLES = $(patsubst examples/%.c,build/%,$(EXAMPLE_SOURCES))
  
# Executable name for main program
EXECUTABLE = refined_besselk
  
# Default target
all: directories examples
  
# Create necessary directories
directories:
	mkdir -p build obj
  
# We need a main source file for the main executable
# If there is no dedicated main.c, comment out the EXECUTABLE from the 'all' target
  
# Compiling CUDA sources
src/%.o: src/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
  
# Examples target
examples: $(EXAMPLE_EXECUTABLES)
  
# Compiling example C sources into executables
build/%: examples/%.c $(CU_OBJECTS)
	$(CC) $(CFLAGS) $< $(CU_OBJECTS) -o $@ $(LFLAGS)
  
# Clean up
clean:
	rm -f $(CU_OBJECTS) $(EXECUTABLE) $(EXAMPLE_EXECUTABLES)
	
# Phony targets
.PHONY: all clean directories examples
