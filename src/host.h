#ifndef SUPERBATCH_PROCESS_H
#define SUPERBATCH_PROCESS_H

#include "bwa.h"
#include "bwamem.h"
#include "kvec.h"
#include "utils.h"
#include "bntseq.h"
#include "kseq.h"
#include <locale.h>
#include "bwamem_GPU.cuh"
#include "batch_config.h"
#include "hashKMerIndex.h"
KSEQ_DECLARE(gzFile)

#define MAX_NUM_GPUS 8

typedef struct
{
	kseq_t *ks, *ks2;
	mem_opt_t *opt;
	mem_pestat_t *pes0;
	int64_t n_processed;
	int copy_comment, actual_chunk_size;
	bwaidx_t *idx;
	kmers_bucket_t *kmerHashTab;
    fmIndex loadedIndex;
    int fd_outfile;
} ktp_aux_t;


#ifdef __cplusplus
extern "C"
{
#endif

	void main_gcube(ktp_aux_t *aux, g3_opt_t *g3_opt);

#ifdef __cplusplus
} // end extern "C"
#endif


#endif
