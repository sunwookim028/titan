#include "zlib.h"
#include "kseq_wrapper.h"

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

/**
 * @brief compare 2 reads a and b.
 * @return int positive if a > b, negative if a < b, 0 if a == b
 */
static int compareReads(const void *a, const void *b)
{
    char *a_key = ((bseq1_t *)a)->seq;
    char *b_key = ((bseq1_t *)b)->seq;
    return strncmp(a_key, b_key, 500);
}

/**
 * @brief sort reads lexicographically
 */
static void sortReads(bseq1_t *reads, int n_reads)
{
    qsort(reads, n_reads, sizeof(bseq1_t), compareReads);
}


// fastq reader

static inline void trim_readno(kstring_t *s)
{
	if (s->l > 2 && s->s[s->l-2] == '/' && isdigit(s->s[s->l-1]))
		s->l -= 2, s->s[s->l] = 0;
}


static inline void kseq2bseq2(const kseq_t *ks, bseq1_t *s, superbatch_data_t *out, g3_opt_t *g3_opt)
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
	if (out->name_size+ks->name.l+1 > SB_NAME_LIMIT){ fprintf(stderr, "[W::%s] FATAL: Memory limit exceeded for seq name.\n", __func__); exit(1); }
	temp = &(out->name[out->name_size]);	// position on the big chunk on host
	memcpy(temp, ks->name.s, ks->name.l); temp[ks->name.l] = '\0';
	s->name = (char*)(out->name + out->name_size);
	out->name_size += ks->name.l + 1;



	// copy seq to host's memory
	if (out->seqs_size+ks->seq.l+1 > SB_SEQ_LIMIT){ fprintf(stderr, "[W::%s] FATAL: Memory limit exceeded for seq.\n", __func__); exit(1); }
	temp = &(out->seqs[out->seqs_size]);
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


	s->seq = (char*)(out->seqs + out->seqs_size);
	out->seqs_size += ks->seq.l + 1;



	// copy comment if not NULL
	if (ks->comment.l == 0){
		s->comment = NULL;
	} else {
		if (out->comment_size+ks->comment.l+1 > SB_COMMENT_LIMIT){ fprintf(stderr, "[W::%s] FATAL: Memory limit exceeded for seq comment.\n", __func__); exit(1); }
		temp = &(out->comment[out->comment_size]);
		memcpy(temp, ks->comment.s, ks->comment.l); temp[ks->comment.l] = '\0';
		s->comment = (char*)(out->comment + out->comment_size);
		out->comment_size += ks->comment.l + 1;
	}
	// copy qual if not NULL
	if (ks->qual.l == 0){
		s->qual = NULL;
	} else {
		if (out->qual_size+ks->qual.l+1 > SB_QUAL_LIMIT){ fprintf(stderr, "[W::%s] FATAL: Memory limit exceeded for seq qual.\n", __func__); exit(1); }
		temp = &(out->qual[out->qual_size]);
		memcpy(temp, ks->qual.s, ks->qual.l); temp[ks->qual.l] = '\0';
		s->qual = (char*)(out->qual + out->qual_size);
		out->qual_size += ks->qual.l + 1;
	}
	s->l_seq = ks->seq.l;
	s->l_comment = ks->comment.l;
	s->l_qual = ks->qual.l;
	s->l_name = ks->name.l;
}

/* 
	load all data to transfer_data
 */
void bseq_read2(unsigned long long loading_batch_size, unsigned long long *n_, void *ks1_, void *ks2_, superbatch_data_t *transfer_data, g3_opt_t *g3_opt)
{
	kseq_t *ks = (kseq_t*)ks1_, *ks2 = (kseq_t*)ks2_;
	//unsigned long long size = 0;
	unsigned long long n = 0;
	bseq1_t *seqs = transfer_data->reads;
    //fprintf(stderr, "* %s loading_batch_size %ld\n", __func__, loading_batch_size);
	while(kseq_read(ks) >= 0 && n < loading_batch_size){
		trim_readno(&ks->name);
		kseq2bseq2(ks, &seqs[n], transfer_data, g3_opt);

		seqs[n].id = n;
		//size += seqs[n].l_seq;
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
static unsigned long long dataloader(kseq_t *ks, kseq_t *ks2, unsigned long long loading_batch_size, int copy_comment, superbatch_data_t *transfer_data, g3_opt_t *g3_opt,
        double *func_elapsed_ms)
{
    FUNC_TIMER_START;
    int64_t size = 0;
    unsigned long long n_seqs_read;
    bseq_read2(loading_batch_size, &n_seqs_read, ks, ks2, transfer_data, g3_opt); // this will write to transfer_data
    bseq1_t *reads = transfer_data->reads;
    transfer_data->n_reads = n_seqs_read;
    if(n_seqs_read == 0){
        FUNC_TIMER_END;
        return 0;
    }
    if(copy_comment)
        for (int i = 0; i < n_seqs_read; ++i)
        {
            reads[i].comment = 0;
        }

    // sortReads(reads, n_seqs_read);
    FUNC_TIMER_END;
    return n_seqs_read;
}

/**
 * @brief process all data in fasta files using super batches
 *
 * @param aux top-level data on this program: input fasta files, indexes, mapping parameters.
 */
void main_gcube(ktp_aux_t *aux, g3_opt_t *g3_opt)
{
    struct timespec start, end;
    double walltime_initialize, walltime_process, walltime_cleanup;
    clock_gettime(CLOCK_REALTIME,  &start);

    int num_gpus;
    int num_use_gpus = g3_opt->num_use_gpus;
    cudaGetDeviceCount(&num_gpus);
    if(num_gpus < num_use_gpus){
        std::cerr << "!! invalid request of " << num_use_gpus << " GPUs"
            << "where only " << num_gpus << " GPUs are available." << std::endl;
        exit(1);
    } else{
        std::cerr << "* using " << num_use_gpus << " GPUs out of " << num_gpus
            << " available GPUs." << std::endl;
    }

    // Double buffers to overlap processing and loading the next input batch.
    superbatch_data_t *loaded = newSuperBatchData();
    superbatch_data_t *loading = newSuperBatchData();

    // Double buffers per GPU to overlap processing and communicating,
    //   results of the previous batch and the next input batch (mini-batch).
    process_data_t *proc[MAX_NUM_GPUS];
    transfer_data_t *tran[MAX_NUM_GPUS];

    // Initialize host and device memory for processing with each GPU.
    //   Utilizes pinned & fixed memory for optimal I/O performance.
    for(int j=0; j<num_use_gpus; j++){
        newProcess(&proc[j], j, aux->opt, aux->pes0, aux->idx->bwt,\
                aux->idx->bns, aux->idx->pac, aux->kmerHashTab,\
                &(aux->loadedIndex), g3_opt);
        newTransfer(&tran[j], j, g3_opt);
        tran[j]->fd_outfile = aux->fd_outfile;
    }

    clock_gettime(CLOCK_REALTIME,  &end);
    walltime_initialize = (end.tv_sec - start.tv_sec) +\
                           (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_REALTIME,  &start);
    long int num_total_reads = 0;
    memset(tprof, 0, sizeof(float) * MAX_NUM_GPUS * MAX_NUM_STEPS);


    // Load the next (global) input batch from the host storage
    //   and process the currently loaded batch using all GPUs.
    //   The aligner thread handles all communication and kernel invocation
    //   for GPUs. It also writes results to a single file in host storage.
#define A 0
#define B 1
#define toggle(ab) (1 - (ab))
    int AB = A;
    double load_elapsed_ms;
    double align_elapsed_ms;
    do { // TODO we want dataloader to be multi-threaded as well.
        std::thread t_dataloader(dataloader, aux->ks, aux->ks2, 
                aux->loading_batch_size, aux->copy_comment, loading, g3_opt,
                &load_elapsed_ms);
        std::thread t_aligner(aligner, loaded, proc, tran, g3_opt,
                &align_elapsed_ms);//, superbatch_results[AB]);
        //std::thread t_storer(storer, aux->fd_outfile, superbatch_results[toggle(AB)]); // possibly merge pulled results

        t_aligner.join();
        t_dataloader.join();

        fprintf(stderr, "* loaded %ld reads from storage in %.2f ms\n",
                loading->n_reads, load_elapsed_ms);
        fprintf(stderr, "* aligned %ld reads with %d GPUs in %.2f ms\n",
                loaded->n_reads, g3_opt->num_use_gpus, align_elapsed_ms);

        superbatch_data_t * tmp = loaded;
        loaded = loading;
        loading = tmp;
        resetSuperBatchData(loading);
        AB = toggle(AB);
        num_total_reads += loaded->n_reads;
    } while (loaded->n_reads != 0);

    clock_gettime(CLOCK_REALTIME,  &end);
    walltime_process = (end.tv_sec - start.tv_sec) +\
                           (end.tv_nsec - start.tv_nsec) / 1e9;



    // Destroy all generated per-GPU structures.
    for(int j=0; j<num_use_gpus; j++){
        cudaStreamDestroy(*(cudaStream_t*)proc[j]->CUDA_stream);
        cudaStreamDestroy(*(cudaStream_t*)tran[j]->CUDA_stream);
    }


    std::cerr << std::endl << std::endl;
    fprintf(stderr,"* Wall-clock time mem alloc & transfer: %.2lf seconds\n", walltime_initialize);
    std::cerr << std::endl;
    fprintf(stderr,"* Wall-clock time for aligning all %ld reads (including push & pull time): %.2lf seconds\n", num_total_reads, walltime_process);


    std::cerr << "* \tWall-clock time for alignment across "
        << g3_opt->num_use_gpus << " GPUs for each stage (avg, min, max):"
        << std::endl;

    // Runtime profiling stats
    //  0: min, 1: avg, 2: max.
    float walltime_seeding[3], walltime_chaining[3], walltime_extending[3];
    //  0: seeding, 1: chaining, 2: extending.
    float tim, min_tim[3], max_tim[3], sum_tim[3];
    for(int k=0; k<3; k++){
        min_tim[k] = FLT_MAX; max_tim[k] = FLT_MIN; sum_tim[k] = 0;
    }
    float *tims;
    for(int gpuid = 0; gpuid < g3_opt->num_use_gpus; gpuid++){
        tims = tprof[gpuid];

        tim = tims[S_SMEM] + tims[S_R2] + tims[S_R3];
        if(tim < min_tim[0]) min_tim[0] = tim;
        if(tim > max_tim[0]) max_tim[0] = tim;
        sum_tim[0] += tim;

        tim = tims[C_SAL] + tims[C_SORT_SEEDS] + tims[C_CHAIN]
            + tims[C_SORT_CHAINS] + tims[C_FILTER];
        if(tim < min_tim[1]) min_tim[1] = tim;
        if(tim > max_tim[1]) max_tim[1] = tim;
        sum_tim[1] += tim;

        tim = tims[E_PAIRGEN] + tims[E_EXTEND] + tims[E_FILTER_MARK]
            + tims[E_SORT_ALNS] + tims[E_T_PAIRGEN] + tims[E_TRACEBACK]
            + tims[E_FINALIZE];
        if(tim < min_tim[2]) min_tim[2] = tim;
        if(tim > max_tim[2]) max_tim[2] = tim;
        sum_tim[2] += tim;
    }
    std::cerr << "\t\t\t\t\t- seeding: (";
    std::cerr << std::fixed << std::setprecision(2)
        << sum_tim[0] / g3_opt->num_use_gpus / 1000 << ", "
        << min_tim[0] / 1000 << ", "
        << max_tim[0] / 1000 << ") seconds" << std::endl;

    std::cerr << "\t\t\t\t\t- chaining: (";
    std::cerr << std::fixed << std::setprecision(2)
        << sum_tim[1] / g3_opt->num_use_gpus / 1000 << ", "
        << min_tim[1] / 1000 << ", "
        << max_tim[1] / 1000 << ") seconds" << std::endl;

    std::cerr << "\t\t\t\t\t- extending: (";
    std::cerr << std::fixed << std::setprecision(2)
        << sum_tim[2] / g3_opt->num_use_gpus / 1000 << ", "
        << min_tim[2] / 1000 << ", "
        << max_tim[2] / 1000 << ") seconds" << std::endl;


    for(int gpuid = 0; gpuid < g3_opt->num_use_gpus; gpuid++){
        std::cerr << std::endl
            << "* \tWall-clock time, GPU #" << gpuid
            << " total alignment Sum: ";
        std::cerr << std::fixed << std::setprecision(2) 
            << tprof[gpuid][COMPUTE_TOTAL] / 1000 << " seconds" << std::endl;
    }
}
