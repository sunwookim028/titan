# Compilers and flags
INCLUDES = -Isrc -Iext/zlib-1.3.1 -Iext/safestringlib/include -Iext/bwa-mem2/include
CXX = g++
CXXFLAGS = -Wall -O2 -std=c++11 $(INCLUDES)
CC = gcc
CFLAGS = -Wall -Wno-unused-function -O2 $(INCLUDES) -lm -DUSE_MALLOC_WRAPPERS 
NVCC = /usr/local/cuda-12.1/bin/nvcc
CU_ARCH = sm_86
CU_COMPUTE_ARCH = $(subst sm,compute,$(CU_ARCH))

PRINT_ALL = 
PRINT_ALL += -DPRINT_SMINTV
PRINT_ALL += -DPRINT_CHINTV
PRINT_ALL += -DPRINT_CHSEED_
PRINT_ALL += -DPRINT_CHSEED
PRINT_ALL += -DPRINT_CHCHAIN
PRINT_ALL += -DPRINT_SWCHAIN
PRINT_ALL += -DPRINT_SWPAIR
PRINT_ALL += -DPRINT_SWREG_
PRINT_ALL += -DPRINT_SWREG
PRINT_ALL += -DPRINT_ANREG
PRINT_ALL += -DPRINT_ANPAIR
PRINT_ALL += -DPRINT_ANALN_
PRINT_ALL += -DPRINT_ANALN

PRINT_SCE = 
PRINT_SCE += -DPRINT_CHSEED -DPRINT_SWPAIR #-DPRINT_ANALN

PRINT_SCE+ =
#PRINT_SCE+ += -DPRINT_SMINTV -DPRINT_FLATINTV -DPRINT_CHINTV -DPRINT_CHSEED -DPRINT_CHSEED_ -DPRINT_SWPAIR #-DPRINT_ANALN
PRINT_SCE+ += -DPRINT_SMINTV -DPRINT_CHINTV 

#PRINT_FLAGS = $(PRINT_ALL)

#PRINT_FLAGS = -DPRINT_ANALN

#PRINT_FLAGS = $(PRINT_SCE)
#PRINT_FLAGS = $(PRINT_SCE+)
PRINT_FLAGS = -DPRINT_SMINTV -DPRINT_CHINTV
#PRINT_FLAGS += -DDEBUG_RESEED 

__NVFLAGS = -ccbin /usr/bin/g++-11 --gpu-architecture=$(CU_COMPUTE_ARCH) --gpu-code=$(CU_ARCH) --default-stream per-thread $(INCLUDES) -Xptxas -O4 -Xcompiler -O4
_NVFLAGS = -ccbin /usr/bin/g++-11 --device-c --gpu-architecture=$(CU_COMPUTE_ARCH) --gpu-code=$(CU_ARCH) --default-stream per-thread $(INCLUDES) 
NVFLAGS = $(_NVFLAGS) -Xptxas -O4 -Xcompiler -O4 
NVFLAGS_BASELINE = $(_NVFLAGS) -DBASELINE 
NVFLAGS_DEBUG = $(_NVFLAGS) -lineinfo -Xcompiler -Wall -Xptxas -Werror 

NVFLAGS_PRINT = $(NVFLAGS) $(PRINT_FLAGS)
NVFLAGS_BASELINE_PRINT = $(NVFLAGS_BASELINE) $(PRINT_FLAGS)


# Linker flags
LINKFLAGS = -Lext/zlib-1.3.1 -Lext/safestringlib -L/usr/local/cuda/lib64 -Lext/bwa-mem2
LIBS = -lz -lcudart -lcudadevrt -lsafestring -lreadindexele


# Archiver and flags
AR = ar
ARFLAGS = -csru


# Target executable
TARGET1 = hasher
TARGET2 = titan
TARGET2_DEBUG = titan.debug
TARGET2_BASELINE = baseline
TARGET2_PRINT = titan.print
TARGET2_BASELINE_PRINT = baseline.print

# Sources, objects and dependencies files
CPP_SOURCES = $(wildcard src/*.cpp)
C_SOURCES = $(wildcard src/*.c)
CU_SOURCES = $(wildcard src/*.cu)

CPP_OBJECTS = $(CPP_SOURCES:.cpp=.o)
CPP_OBJECTS_TARGET1 = src/buildIndex.o \
		      src/hashKMer.o \
		      src/loadKMerIndex.o
CPP_OBJECTS_TARGET2 = src/loadKMerIndex.o \
		      src/FMI_wrapper.o \
		      src/FMI_search.o
C_OBJECTS = $(C_SOURCES:.c=.o)

CU_DEBUG_OBJECTS = $(CU_SOURCES:.cu=.debug.o)
_CU_OBJECTS = $(filter-out src/bwamem_GPU.o, $(CU_SOURCES:.cu=.o))
_CU_BASELINE_OBJECTS = $(filter-out src/bwamem_GPU.baseline.o, $(CU_SOURCES:.cu=.baseline.o))

CU_OBJECTS = $(_CU_OBJECTS) src/bwamem_GPU.o
CU_BASELINE_OBJECTS = $(_CU_BASELINE_OBJECTS) src/bwamem_GPU.baseline.o
CU_PRINT_OBJECTS = $(_CU_OBJECTS) src/bwamem_GPU.print.o
CU_BASELINE_PRINT_OBJECTS = $(_CU_BASELINE_OBJECTS) src/bwamem_GPU.baseline.print.o

CU_DEBUG_OBJECTS_LINKER = src/gpu_link.debug.o
CU_OBJECTS_LINKER = src/gpu_link.o
CU_BASELINE_OBJECTS_LINKER = src/gpu_link.baseline.o
CU_PRINT_OBJECTS_LINKER = src/gpu_link.print.o
CU_BASELINE_PRINT_OBJECTS_LINKER = src/gpu_link.baseline.print.o

DEPS_FILE = .depend


# Static library file
LIB_FRONTEND = src/libbwa.a


# Default rule
all: depend $(TARGET1) $(TARGET2)


# Linking rule
$(TARGET1): $(CPP_OBJECTS_TARGET1) $(LIB_FRONTEND) -lz
	$(CXX) $(LINKFLAGS) -o $@ $^

$(TARGET2): $(CU_OBJECTS_LINKER) $(CU_OBJECTS) $(CPP_OBJECTS_TARGET2) $(LIB_FRONTEND) 
	$(CXX) $(LINKFLAGS) -L. -o $@ $^ $(LIBS) 

$(TARGET2_BASELINE): $(CU_BASELINE_OBJECTS_LINKER) $(CU_BASELINE_OBJECTS) $(CPP_OBJECTS_TARGET2) $(LIB_FRONTEND) 
	$(CXX) $(LINKFLAGS) -L. -o $@ $^ -lz -lcudart -lcudadevrt -lsafestring

$(TARGET2_PRINT): $(CU_PRINT_OBJECTS_LINKER) $(CU_PRINT_OBJECTS) $(CPP_OBJECTS_TARGET2) $(LIB_FRONTEND) 
	$(CXX) $(LINKFLAGS) -L. -o $@ $^ -lz -lcudart -lcudadevrt -lsafestring

$(TARGET2_BASELINE_PRINT): $(CU_BASELINE_PRINT_OBJECTS_LINKER) $(CU_BASELINE_PRINT_OBJECTS) $(CPP_OBJECTS_TARGET2) $(LIB_FRONTEND) 
	$(CXX) $(LINKFLAGS) -L. -o $@ $^ -lz -lcudart -lcudadevrt -lsafestring

$(TARGET2_DEBUG): $(CU_DEBUG_OBJECTS_LINKER) $(CU_DEBUG_OBJECTS) $(CPP_OBJECTS_TARGET2) $(LIB_FRONTEND) 
	$(CXX) $(LINKFLAGS) -L. -o $@ $^ -lz -lcudart -lcudadevrt -lsafestring


$(LIB_FRONTEND): $(C_OBJECTS)
	$(AR) $(ARFLAGS) -o $@ $^

$(CU_OBJECTS_LINKER): $(CU_OBJECTS)
	$(NVCC) $(__NVFLAGS) --device-link $^ --output-file $@

$(CU_PRINT_OBJECTS_LINKER): $(CU_PRINT_OBJECTS)
	$(NVCC) $(__NVFLAGS) --device-link $^ --output-file $@

$(CU_BASELINE_OBJECTS_LINKER): $(CU_BASELINE_OBJECTS)
	$(NVCC) $(__NVFLAGS) --device-link $^ --output-file $@

$(CU_BASELINE_PRINT_OBJECTS_LINKER): $(CU_BASELINE_PRINT_OBJECTS)
	$(NVCC) $(__NVFLAGS) --device-link $^ --output-file $@

$(CU_DEBUG_OBJECTS_LINKER): $(CU_DEBUG_OBJECTS)
	$(NVCC) $(__NVFLAGS) --device-link $^ --output-file $@


# Compile rules
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<
%.o: %.cu
	$(NVCC) $(NVFLAGS) -c -o $@ $<
%.debug.o: %.cu
	$(NVCC) $(NVFLAGS_DEBUG) -c -o $@ $<
%.baseline.o: %.cu
	$(NVCC) $(NVFLAGS_BASELINE) -c -o $@ $<
src/bwamem_GPU.print.o: src/bwamem_GPU.cu
	$(NVCC) $(NVFLAGS_PRINT) -c -o $@ $<
src/bwamem_GPU.baseline.print.o: src/bwamem_GPU.cu
	$(NVCC) $(NVFLAGS_BASELINE_PRINT) -c -o $@ $<

#########################################################################################
# Make rules
#########################################################################################
EXE = $(TARGET2)
#REF_EXE = bwa-mem2
#REF_TARGETS = build run

build: $(EXE)

#ref:
#	$(MAKE) -C ../bwa-mem2 $(REF_TARGETS) REF_EXE=$(REF_EXE)


# Clean rule
clean:
	rm *sam *out *err

largeclean:
	rm -f $(CU_DEBUG_OBJECTS_LINKER) $(CU_OBJECTS_LINKER) $(CU_BASELINE_OBJECTS_LINKER) $(CU_BASELINE_OBJECTS) $(CU_DEBUG_OBJECTS) $(CU_OBJECTS) $(CPP_OBJECTS) $(C_OBJECTS) $(TARGET1) $(TARGET2) $(DEPS_FILE)
	rm -f $(EVALSUBDIR)/*
	rm -f temp*
	$(MAKE) -C ../bwa-mem2 clean
	rm -f ../bwa-mem2/$(EVALSUBDIR)/*


# Depend rule
depend:
	@echo "Generating dependencies..."
	@$(CXX) -MM $(CPP_SOURCES) -Isrc > $(DEPS_FILE)
	@$(CC) -MM $(C_SOURCES) -Isrc >> $(DEPS_FILE)
	@$(NVCC) -MM $(CU_SOURCES) -Isrc >> $(DEPS_FILE)


# Phony label
.PHONY: all clean depend


# For debugging this build system 
debug_make: $(CU_DEBUG_OBJECTS)
	@echo "Compiled cuda object files."

compile: $(CPP_OBJECTS)


# Dependencies
-include $(DEPS_FILE)
#EOD
