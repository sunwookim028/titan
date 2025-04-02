#include "bwa.h"
#include "kseq_wrapper.h"

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
    int fd_outfile;
} ktp_aux_t;

