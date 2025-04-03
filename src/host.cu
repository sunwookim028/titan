#include "zlib.h"
#include "fastmap.h"
#include "kstring.h"
#include "macro.h"
#include "timer.h"
#include "streams.cuh"
#include <future>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <climits>
#include <cfloat>
#include <thread>
using namespace std;

extern void aligner(superbatch_data_t *data, process_data_t *proc[MAX_NUM_GPUS],\
        transfer_data_t *tran[MAX_NUM_GPUS], g3_opt_t *g3_opt,
        double *elapsed_ms);

extern float tprof[MAX_NUM_GPUS][MAX_NUM_STEPS];

/**
 * @brief initiate memory for a super batch
 * @return superbatch_data_t*
 */
static superbatch_data_t *newSuperBatchData()
{
    superbatch_data_t *batch = (superbatch_data_t *)malloc(sizeof(superbatch_data_t));
    batch->n_reads = 0;
    // init memory for reads in the batch
    batch->reads = (bseq1_t *)malloc(SB_MAX_COUNT * sizeof(bseq1_t));
    batch->name = (char *)malloc(SB_NAME_LIMIT);
    batch->seqs = (char *)malloc(SB_SEQ_LIMIT);
    batch->comment = (char *)malloc(SB_COMMENT_LIMIT);
    batch->qual = (char *)malloc(SB_QUAL_LIMIT);
    if (batch->reads == nullptr || batch->name == nullptr || batch->seqs == nullptr || batch->comment == nullptr || batch->qual == nullptr)
    {
        fprintf(stderr, "[M::%-25s] can't malloc superbatch\n", __func__);
        exit(1);
    }

    batch->name_size = 0;
    batch->comment_size = 0;
    batch->seqs_size = 0;
    batch->qual_size = 0;

        double nGB_allocated = (double)(SB_MAX_COUNT * sizeof(bseq1_t) + SB_NAME_LIMIT + SB_SEQ_LIMIT + SB_COMMENT_LIMIT + SB_QUAL_LIMIT) / (1024ULL * 1024ULL * 1024ULL);
        fprintf(stderr, "[M::%-25s] allocated %.2f GB on host for superbatch\n", __func__, nGB_allocated);
    return batch;
}

/**
 * @brief remove data from a superbatch data set
 */
static void resetSuperBatchData(superbatch_data_t *data)
{
    data->n_reads = 0;
    data->name_size = 0;
    data->comment_size = 0;
    data->seqs_size = 0;
    data->qual_size = 0;
}


// fastq reader

static inline void trim_readno(kstring_t *s)
{
	if (s->l > 2 && s->s[s->l-2] == '/' && isdigit(s->s[s->l-1]))
		s->l -= 2, s->s[s->l] = 0;
}


static inline void kseq2bseq2(const kseq_t *ks, bseq1_t *s, superbatch_data_t *loading, g3_opt_t *g3_opt)
{
    const uint8_t nst_nt4_table[256] = {
        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 5 /*'-'*/, 4, 4,
        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
        4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4, 
        4, 4, 4, 4,  3, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
        4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4, 
        4, 4, 4, 4,  3, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4
    };

    if (ks->seq.l > SEQ_MAXLEN){
        fprintf(stderr, "[M::%-25s] ERROR: a read has length of %ld, more than the maximum set %d \n", __func__, ks->seq.l, SEQ_MAXLEN);
        exit(1);
    }

	char* temp;
	// copy name to host's memory
	if (loading->name_size+ks->name.l+1 > SB_NAME_LIMIT){ fprintf(stderr, "[W::%s] FATAL: Memory limit exceeded for seq name.\n", __func__); exit(1); }
	temp = &(loading->name[loading->name_size]);	// position on the big chunk on host
	memcpy(temp, ks->name.s, ks->name.l); temp[ks->name.l] = '\0';
	s->name = (char*)(loading->name + loading->name_size);
	loading->name_size += ks->name.l + 1;



	// copy seq to host's memory
	if (loading->seqs_size+ks->seq.l+1 > SB_SEQ_LIMIT){ fprintf(stderr, "[W::%s] FATAL: Memory limit exceeded for seq.\n", __func__); exit(1); }
	temp = &(loading->seqs[loading->seqs_size]);
	//memcpy(temp, ks->seq.s, ks->seq.l); temp[ks->seq.l] = '\0';
    for(int j=0; j < ks->seq.l; j++)
    {
        if(g3_opt->baseline){
            temp[j] = (ks->seq.s)[j];
        } else{
            temp[j] = (char)nst_nt4_table[(int)((ks->seq.s)[j])];
        }
    }
    temp[ks->seq.l] = '\0';


	s->seq = (char*)(loading->seqs + loading->seqs_size);
	loading->seqs_size += ks->seq.l + 1;



	// copy comment if not NULL
	if (ks->comment.l == 0){
		s->comment = NULL;
	} else {
		if (loading->comment_size+ks->comment.l+1 > SB_COMMENT_LIMIT){ fprintf(stderr, "[W::%s] FATAL: Memory limit exceeded for seq comment.\n", __func__); exit(1); }
		temp = &(loading->comment[loading->comment_size]);
		memcpy(temp, ks->comment.s, ks->comment.l); temp[ks->comment.l] = '\0';
		s->comment = (char*)(loading->comment + loading->comment_size);
		loading->comment_size += ks->comment.l + 1;
	}
	// copy qual if not NULL
	if (ks->qual.l == 0){
		s->qual = NULL;
	} else {
		if (loading->qual_size+ks->qual.l+1 > SB_QUAL_LIMIT){ fprintf(stderr, "[W::%s] FATAL: Memory limit exceeded for seq qual.\n", __func__); exit(1); }
		temp = &(loading->qual[loading->qual_size]);
		memcpy(temp, ks->qual.s, ks->qual.l); temp[ks->qual.l] = '\0';
		s->qual = (char*)(loading->qual + loading->qual_size);
		loading->qual_size += ks->qual.l + 1;
	}
	s->l_seq = ks->seq.l;
	s->l_comment = ks->comment.l;
	s->l_qual = ks->qual.l;
	s->l_name = ks->name.l;
}

void bseq_read2(unsigned long long loading_batch_size, unsigned long long *n_, void *ks1_, void *ks2_, superbatch_data_t *loading, g3_opt_t *g3_opt)
{
	kseq_t *ks = (kseq_t*)ks1_, *ks2 = (kseq_t*)ks2_;

	unsigned long long n = 0;
	bseq1_t *seqs = loading->reads;

	while(kseq_read(ks) >= 0 && n < loading_batch_size){
		trim_readno(&ks->name);
		kseq2bseq2(ks, &seqs[n], loading, g3_opt);

		seqs[n].id = n;
        n++;
	}
	*n_ = n;
	return;
}

/**
 * @brief
 *
 * @param ks
 * @param ks2
 * @param chunk_size
 * @param copy_comment
 * @param transfer_data
 * @return int number of reads loaded from file
 */
static unsigned long long dataloader(
        kseq_t *ks, kseq_t *ks2, 
        unsigned long long loading_batch_size, 
        superbatch_data_t *loading, g3_opt_t *g3_opt,
        double *func_elapsed_ms)
{
    FUNC_TIMER_START;
    int64_t size = 0;
    unsigned long long n_seqs_read;
    bseq_read2(loading_batch_size, &n_seqs_read, ks, ks2, loading, g3_opt); // this will write to loading
    bseq1_t *reads = loading->reads;
    loading->n_reads = n_seqs_read;
    if(n_seqs_read == 0){
        FUNC_TIMER_END;
        return 0;
    }

    FUNC_TIMER_END;
    return n_seqs_read;
}

void filewriter(std::ostream *samout, double *walltime)
{
    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME,  &start);

    // TODO
    
    *samout << "@SQ	SN:U00096.3	LN:4641652" << std::endl;
    *samout << "@PG	ID:bwa-mem2	PN:bwa-mem2	VN:2.2.1	CL:./bwa-mem2 mem -o test.sam /datasets/bwa/ref/ecoli/GCA_000005845.2_ASM584v2_genomic.fna /datasets/bwa/reads/ecoli/SRR10211335_first4.fastq" << std::endl;

    for(long long int k = 0; k < 800000; k++){
        *samout << "SRR10211335." << k << "	16	U00096.3	2759130	60	100M	*	0	0	GGATTGATGTTTGCCGATTGAATAATCTACGTGGCCCGGTATCACTTTTCTTAATGACTCTGGCTGAATCAGGTGAACGTAAGAGTACGGTTGATAAACN	????????????????????????????????????????????????????????????????????????????????????????????????????	NM:i:1	MD:Z:99T0	AS:i:99	XS:i:0"
            << std::endl;
    }
    samout->flush();

    clock_gettime(CLOCK_REALTIME,  &end);
    *walltime= (end.tv_sec - start.tv_sec) +\
                           (end.tv_nsec - start.tv_nsec) / 1e9;
    return;
}



/*
   1. file batch
   2. align batch
   --------

   1. file batch
   - ID: global offset
   - metadata: batch size
   - data:
        * INPUT (FASTQ)
            - seq / name, qual, comment.
        * OUTPUT (SAM)
            - rid, pos, cigar / name, seq, qual, comment / scores, additionals.

        * HOW to access each data entry?
            : with local offset [0..batch size-1).
        * LIFETIME of each file batch?
            : currently aligning file batch lives UNTIL all SAMs are OUTPUT.
              next file batch is INPUT during the same period.

   2. align batch
   - ID: global offset
   - metadata: batch size
   - data:
        * PUSH (READ)
            - seq.
        * PULL (ALIGNMENT)
            - rid, pos, cigar.
 */
void pipeline(ktp_aux_t *aux, g3_opt_t *g3_opt)
{
    struct timespec start, end;
    double walltime_initialize, walltime_process, walltime_cleanup;
    double load_elapsed_ms;
    double write_elapsed_ms;
    double first_load_elapsed_ms;
    double align_elapsed_ms;

    long int num_total_reads = 0;
    int num_available_gpus;

    superbatch_data_t *loaded, *loading;
    process_data_t *proc[MAX_NUM_GPUS];
    transfer_data_t *tran[MAX_NUM_GPUS];

    // Initialize runtime timers.
    memset(tprof, 0, sizeof(float) * MAX_NUM_GPUS * MAX_NUM_STEPS);

    clock_gettime(CLOCK_REALTIME,  &start);

    // Query GPU environment.
    cudaGetDeviceCount(&num_available_gpus);
    if(num_available_gpus < g3_opt->num_use_gpus){
        std::cerr << "!! invalid request of " << g3_opt->num_use_gpus << " GPUs"
            << "where only " << num_available_gpus << " GPUs are available." << std::endl;
        exit(1);
    } else{
        std::cerr << "* using " << g3_opt->num_use_gpus << " GPUs out of " << num_available_gpus
            << " available GPUs." << std::endl;
    }

    // Double buffers to overlap processing and loading the next input batch.
    loaded = newSuperBatchData();
    loading = newSuperBatchData();

    // Double buffers per GPU to overlap processing and communicating,
    //   results of the previous batch and the next input batch (mini-batch).

    // Initialize host and device memory for processing with each GPU.
    //   Utilizes pinned & fixed memory for optimal I/O performance.
    for(int j=0; j<g3_opt->num_use_gpus; j++){
        newProcess(&proc[j], j, aux->opt, aux->pes0, aux->idx->bwt,\
                aux->idx->bns, aux->idx->pac, aux->kmerHashTab,\
                &(aux->loadedIndex), g3_opt);
        newTransfer(&tran[j], j, g3_opt);
    }

    clock_gettime(CLOCK_REALTIME,  &end);
    walltime_initialize = (end.tv_sec - start.tv_sec) +\
                           (end.tv_nsec - start.tv_nsec) / 1e9;
    tprof[0][GPU_SETUP] = (float)(walltime_initialize * 1000);


    // load the first loading batch.
    resetSuperBatchData(loaded);
    dataloader(aux->ks, aux->ks2, 
            aux->loading_batch_size, loaded, g3_opt,
            &first_load_elapsed_ms);
    num_total_reads += loaded->n_reads;
    tprof[0][FILE_INPUT_FIRST] += first_load_elapsed_ms;
    tprof[0][FILE_INPUT] += first_load_elapsed_ms;

    std::cerr << "* loading the first file batch took "
        << first_load_elapsed_ms << " ms." << std::endl;

    // Create the file writer thread & in-memory pull queues for GPUs to write to.
    std::thread t_writer(filewriter, aux->samout, &write_elapsed_ms);

    // align and load rest of the input fastq file.
    while(loaded->n_reads != 0){
        std::thread t_aligner(aligner, loaded, proc, tran, g3_opt,
                &align_elapsed_ms);//, superbatch_results[AB]);
        std::thread t_dataloader(dataloader, aux->ks, aux->ks2, 
                aux->loading_batch_size, loading, g3_opt,
                &load_elapsed_ms);

        t_aligner.join();
        t_dataloader.join();

        tprof[0][FILE_INPUT] += load_elapsed_ms;
        tprof[0][ALIGNER_TOP] += align_elapsed_ms;

        std::cerr << "* loading: " << load_elapsed_ms 
            << " ms." << std::endl;

        superbatch_data_t * tmp = loaded;
        loaded = loading;
        loading = tmp;
        resetSuperBatchData(loading);
        num_total_reads += loaded->n_reads;
    }
    t_writer.join();
    tprof[0][FILE_OUTPUT] = write_elapsed_ms;

    std::cerr << "* Loaded total " << num_total_reads << " reads "
        << "from the input fastq file." << std::endl;

    // Destroy all generated per-GPU structures.
    for(int j=0; j<g3_opt->num_use_gpus; j++){
        cudaStreamDestroy(*(cudaStream_t*)proc[j]->CUDA_stream);
        cudaStreamDestroy(*(cudaStream_t*)tran[j]->CUDA_stream);
    }
    return;
}
