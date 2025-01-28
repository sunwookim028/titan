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


# Intermediate results print identifiers.
# NAMES SHOULD MATCH THE DEFINITIONS IN src/printintermediates.cu
SMintv = SMintv
CHintv = CHintv
CHseed_ = CHseed_
CHseed = CHseed

CHchain = CHchain
SWchain = SWchain
SWpair = SWpair
#lower_chain = lower

SWreg_ = SWreg_
SWreg = SWreg
ANreg = ANreg
ANpair = ANpair
ANaln_ = ANaln_
ANaln = ANaln
STAGES = $(SMintv) $(CHintv) $(CHseed) $(CHchain) $(SWchain) $(SWpair) $(SWreg_) $(SWreg) $(ANreg) $(ANpair) $(ANaln_) $(ANaln)


# Targets for building and testing
EXE = $(TARGET2)
HG=76bp
SIZE=100k
INPUT_FASTA = $(HG).$(SIZE)
REF_PRINT = bwa-mem2.print
REF_EXE = $(REF_PRINT)
REF_TARGETS = build run extract-results_sce
STAGE = $(SWreg_)


# Make rules
#########################################################################################

build: $(EXE)

HG38_IDX_PREFIX = ~/ours/input/index/hg38
EVALSUBDIR=eval
run: #build
	./$(EXE) mem $(HG38_IDX_PREFIX).fa $(HG38_IDX_PREFIX).hash ~/reads/$(INPUT_FASTA) -o $(EVALSUBDIR)/$(INPUT_FASTA).$(EXE) 
gdb_run:
	cuda-gdb --args ./$(EXE) mem $(HG38_IDX_PREFIX).fa $(HG38_IDX_PREFIX).hash -o $(EVALSUBDIR)/$(INPUT_FASTA).$(EXE) 

ref:
	$(MAKE) -C ../bwa-mem2 $(REF_TARGETS) INPUT_FASTA=$(INPUT_FASTA) REF_EXE=$(REF_EXE)

help:
	@echo "example usage:"
	@echo "make prep [NUM=num_gpus] <- this prepares the experiment environment."
	@echo "make test <- this runs the experiment."
	@echo ""
	@echo "Last experiment was on extending gcube to multiple GPUs."
	@echo "See the commit log for more details."

ECOLI_IDX_PREFIX = ../input/ecoli/GCA_000005845.2_ASM584v2_genomic
#ECOLI_READS = /nfs/home/skim28/ecoliball/SRR31619093.fastq
#ECOLI_READS = /nfs/home/skim28/ecoliball/largeinput.fastq
_ECOLI_READS = ../input/ecoli/ecoli.$(SIZE)
OUTFILE=test.sam
NUM_GPUS=1
ERRBUF=std.err
OUTBUF=std.out
OPT=-l
TEST_COMMAND = ./$(EXE) mem $(HG38_IDX_PREFIX).fa $(HG38_IDX_PREFIX).hash ~/reads/$(INPUT_FASTA) -o $(OUTFILE) $(OPT) 2>> $(ERRBUF) 1>>$(OUTBUF)
SMALLTEST_COMMAND = ./$(EXE) mem $(ECOLI_IDX_PREFIX).fna $(ECOLI_IDX_PREFIX).hash $(_ECOLI_READS) -o smalltest.sam -g $(NUM_GPUS) $(OPT) 1>> $(OUTBUF) #2>> $(ERRBUF) 

test: # latest testing script
	@echo "=======================================================" >> $(ERRBUF)
	$(TEST_COMMAND)

smalltest: #
	@echo "=======================================================" >> $(ERRBUF)
	$(SMALLTEST_COMMAND)

sanitize:
	compute-sanitizer --quiet --launch-timeout 9000 $(TEST_COMMAND)

gdb:
	@make _gdb EXE=$(TARGET2_DEBUG)

_gdb:
	@cuda-gdb --args $(SMALLTEST_COMMAND)

#########################################################################################

#
# Accuracy Evaluation
#
extract-intermediate: 
	/usr/bin/grep -w $(STAGE) $(EVALSUBDIR)/$(INPUT_FASTA).$(EXE) > $(EVALSUBDIR)/$(STAGE).$(INPUT_FASTA).$(EXE)

sort-intermediate: extract-intermediate
	sort $(EVALSUBDIR)/$(STAGE).$(INPUT_FASTA).$(EXE) > $(EVALSUBDIR)/$(STAGE).$(INPUT_FASTA).$(EXE).sorted
	sort ../bwa-mem2/$(EVALSUBDIR)/$(STAGE).$(INPUT_FASTA).$(REF_EXE) > ../bwa-mem2/$(EVALSUBDIR)/$(STAGE).$(INPUT_FASTA).$(REF_EXE).sorted


sort-intv:
	$(MAKE) sort-intermediate STAGE=$(SMintv)
	$(MAKE) sort-intermediate STAGE=$(CHintv)


sort-intermediate-sce:
	$(MAKE) sort-intermediate STAGE=$(CHseed) 
	$(MAKE) sort-intermediate STAGE=$(SWpair)
	#$(MAKE) sort-intermediate STAGE=$(ANaln)

FILE=./temp
STAGE_NAME = $(STAGE)
report-accuracy: sort-intermediate
	@missed=$$(diff $(EVALSUBDIR)/$(STAGE).$(INPUT_FASTA).$(EXE).sorted ../bwa-mem2/$(EVALSUBDIR)/$(STAGE).$(INPUT_FASTA).$(REF_EXE).sorted | grep '^>' | wc -l);\
	total=$$(cat ../bwa-mem2/$(EVALSUBDIR)/$(STAGE).$(INPUT_FASTA).$(REF_EXE) | wc -l);\
	percentage=$$(awk "BEGIN {print (1 - ($$missed / $$total)) * 100}");\
	echo "$(STAGE_NAME) accuracy: $$percentage%" | tee -a $(FILE)

report-accuracy-sce:
	echo "Alignment of $(INPUT_FASTA) with $(EXE)" >> $(FILE)
	$(MAKE) report-accuracy STAGE=$(CHseed) STAGE_NAME=Seeding
	$(MAKE) report-accuracy STAGE=$(SWpair) STAGE_NAME=Chaining
	#$(MAKE) report-accuracy STAGE=$(ANaln) STAGE_NAME=Extending

report-accuracy-all:
	@for stage in $(STAGES); do\
		$(MAKE) report-accuracy STAGE=$$stage STAGE_NAME=$$stage;\
	done

aux_:
	$(MAKE) -C ../bwa-mem2 extract-intv REF_EXE=$(REF_PRINT) INPUT_FASTA=$(INPUT_FASTA)

report-intv:
	$(MAKE) build run EXE=$(TARGET2_BASELINE_PRINT)
	$(MAKE) -C ../bwa-mem2 build REF_EXE=$(REF_PRINT) 
	$(MAKE) -C ../bwa-mem2 run REF_EXE=$(REF_PRINT) INPUT_FASTA=$(INPUT_FASTA)
	$(MAKE) -C ../bwa-mem2 extract-intv REF_EXE=$(REF_PRINT) INPUT_FASTA=$(INPUT_FASTA)
	echo "Seeding_interval of $(INPUT_FASTA) with $(EXE)" >> $(FILE)
	$(MAKE) report-accuracy STAGE=$(SMintv) STAGE_NAME=SMEM_intervals
	$(MAKE) report-accuracy STAGE=$(CHintv) STAGE_NAME=ALL_intervals
	@echo ""
	@echo ""
	@echo ""
	@echo ""
	@echo ""
	@cat $(FILE)
	@bash ee


report:
	$(MAKE) build run EXE=$(TARGET2_PRINT) #PRINT_FLAGS="$(PRINT_SCE)"
	$(MAKE) -C ../bwa-mem2 build REF_EXE=$(REF_PRINT) #PRINT_FLAGS="-DG3_TESTING $(PRINT_SCE)"
	$(MAKE) -C ../bwa-mem2 run REF_EXE=$(REF_PRINT) INPUT_FASTA=$(INPUT_FASTA)
	$(MAKE) -C ../bwa-mem2 extract-intermediate-sce REF_EXE=$(REF_PRINT) INPUT_FASTA=$(INPUT_FASTA)
	$(MAKE) report-accuracy-sce
	@echo ""
	@echo ""
	@echo ""
	@echo ""
	@echo ""
	@cat $(FILE)
	#@rm $(file)


INPUTS = 76bp.100 100bp.100 152bp.100 251bp.100
report-all:
	@for input in $(INPUTS); do\
		$(MAKE) report INPUT_FASTA=$$input;\
	done;



#########################################################################################

#
# Speedup Evaluation
#
extract-times:
	/usr/bin/grep SeedingStage $(EVALSUBDIR)/$(INPUT_FASTA).$(EXE) > $(EVALSUBDIR)/time.$(INPUT_FASTA).$(EXE)
	/usr/bin/grep ChainingStage $(EVALSUBDIR)/$(INPUT_FASTA).$(EXE) >> $(EVALSUBDIR)/time.$(INPUT_FASTA).$(EXE)
	/usr/bin/grep ExtendingStage $(EVALSUBDIR)/$(INPUT_FASTA).$(EXE) >> $(EVALSUBDIR)/time.$(INPUT_FASTA).$(EXE)
	/usr/bin/grep CommunicationStage $(EVALSUBDIR)/$(INPUT_FASTA).$(EXE) >> $(EVALSUBDIR)/time.$(INPUT_FASTA).$(EXE)
	/usr/bin/grep AllStages $(EVALSUBDIR)/$(INPUT_FASTA).$(EXE) >> $(EVALSUBDIR)/time.$(INPUT_FASTA).$(EXE)

measure: 
	$(MAKE) build run extract-times EXE=titan
	$(MAKE) build run extract-times EXE=baseline
	@echo ""
	@echo "titan:"
	@cat $(EVALSUBDIR)/time.$(INPUT_FASTA).titan
	@echo "baseline:"
	@cat $(EVALSUBDIR)/time.$(INPUT_FASTA).baseline

#########################################################################################

# 
# Debugging
#

vimdiff-single:
	vimdiff $(EVALSUBDIR)/$(STAGE).$(INPUT_FASTA).$(EXE).sorted ../bwa-mem2/$(EVALSUBDIR)/$(STAGE).$(INPUT_FASTA).$(REF_EXE).sorted

vimdiff_stages = $(SMintv) $(CHintv) $(CHseed) $(SWpair)
vimdiff:
	@for stage in $(vimdiff_stages); do\
		$(MAKE) vimdiff-single STAGE=$$stage;\
	done

vimdiff-raw: 
	vimdiff $(EVALSUBDIR)/$(STAGE).$(INPUT_FASTA).$(EXE) ../bwa-mem2/$(EVALSUBDIR)/$(STAGE).$(INPUT_FASTA).$(REF_EXE)

vimdiff-sorted: 
	sort $(EVALSUBDIR)/$(STAGE).$(INPUT_FASTA).$(EXE) > $(EVALSUBDIR)/$(STAGE).$(INPUT_FASTA).$(EXE).sorted
	sort ../bwa-mem2/$(EVALSUBDIR)/$(STAGE).$(INPUT_FASTA).$(REF_EXE) > ../bwa-mem2/$(EVALSUBDIR)/$(STAGE).$(INPUT_FASTA).$(REF_EXE).sorted
	vimdiff $(EVALSUBDIR)/$(STAGE).$(INPUT_FASTA).$(EXE).sorted ../bwa-mem2/$(EVALSUBDIR)/$(STAGE).$(INPUT_FASTA).$(REF_EXE).sorted

vimdiff-sorted-all:
	@for stage in $(STAGES); do\
		$(MAKE) vimdiff-sort STAGE=$$stage;\
	done




_test:
	@for stage in $(STAGES); do\
		echo $$stage;\
		/usr/bin/grep -w $$stage $(EVALSUBDIR)/100bp.10000.titan.print | wc -l;\
	done

#########################################################################################

# Clean rule
clean:
	rm *sam *out *err

bigclean:
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

#########################################################################################
NUM=4
prep:
	srun --gres=gpu:$(NUM) -p a6000 --pty bash
