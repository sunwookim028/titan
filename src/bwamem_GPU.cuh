#ifndef _BWAMEM_CUH
#define _BWAMEM_CUH
#define CUDA_BLOCKSIZE 32

#include "bwa.h"
#include "bwt.h"
#include "bntseq.h"
#include "bwamem.h"
#include "streams.cuh"

#ifdef __cplusplus
extern "C"{
#endif
	/* align reads and return the size of SAM output */
	void bwa_align(int gpuid, process_data_t *process_data, g3_opt_t *g3_opt);
#ifdef __cplusplus
} // end extern "C"
#endif

#endif
