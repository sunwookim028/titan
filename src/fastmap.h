#ifndef FASTMAP_H
#define FASTMAP_H

#include <iostream>
#include <fstream>
#include <string>
#include "bwa.h"
#include "zlib.h"
#include "kseq.h"
KSEQ_DECLARE(gzFile)

typedef struct
{
	kseq_t *ks, *ks2;
	mem_opt_t *opt;
	mem_pestat_t *pes0;
	int64_t n_processed;
	int copy_comment;
    long long int loading_batch_size;
	bwaidx_t *idx;
	kmers_bucket_t *kmerHashTab;
    fmIndex loadedIndex;
    std::ostream *samout;
} ktp_aux_t;

void *kopen(const char *fn, int *_fd);
int kclose(void *a);

#endif
