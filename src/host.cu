#include "host.h"
#include "batch_config.h"
#include "offloader.h"
#include "streams.cuh"
#include <future>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <iostream>
#include <climits>
#include <thread>
using namespace std;

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

    if (bwa_verbose >= 3)
    {
        double nGB_allocated = (double)(SB_MAX_COUNT * sizeof(bseq1_t) + SB_NAME_LIMIT + SB_SEQ_LIMIT + SB_COMMENT_LIMIT + SB_QUAL_LIMIT) / (1024ULL * 1024ULL * 1024ULL);
        fprintf(stderr, "[M::%-25s] allocated %.2f GB on host for superbatch\n", __func__, nGB_allocated);
    }
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

/**
 * @brief
 *
 * @param ks
 * @param ks2
 * @param actual_chunk_size
 * @param copy_comment
 * @param transfer_data
 * @return int number of reads loaded from file
 */
static unsigned long long dataloader(kseq_t *ks, kseq_t *ks2, unsigned long long actual_chunk_size, int copy_comment, superbatch_data_t *transfer_data, g3_opt_t *g3_opt)
{
    int64_t size = 0;
    unsigned long long n_seqs_read;
    bseq_read2(actual_chunk_size, &n_seqs_read, ks, ks2, transfer_data, g3_opt); // this will write to transfer_data
    bseq1_t *reads = transfer_data->reads;
    transfer_data->n_reads = n_seqs_read;
    if(n_seqs_read == 0){
        return 0;
    }
    if(copy_comment)
        for (int i = 0; i < n_seqs_read; ++i)
        {
            reads[i].comment = 0;
        }

    // sortReads(reads, n_seqs_read);
    return n_seqs_read;
}

/**
 * @brief process all data in fasta files using super batches
 *
 * @param aux top-level data on this program: input fasta files, indexes, mapping parameters.
 */
void main_gcube(ktp_aux_t *aux, g3_opt_t *g3_opt)
{
    int num_gpus;
    int num_use_gpus = g3_opt->num_use_gpus;
    cudaGetDeviceCount(&num_gpus);
    std::cerr << "PROCESSING WITH " << num_use_gpus << " GPUS, OUT OF "\
        << num_gpus << " GPUS." << std::endl;
    if(num_gpus < num_use_gpus){
        std::cerr << "INVALID REQUEST" << std::endl;
        exit(1);
    }

    superbatch_data_t *loaded = newSuperBatchData();
    superbatch_data_t *loading = newSuperBatchData();

    process_data_t *proc[MAX_NUM_GPUS];
    transfer_data_t *tran[MAX_NUM_GPUS];

    // initialize pinned or fixed memory on both host and device 
    // and transfer index data to each GPU device.
    for(int j=0; j<num_use_gpus; j++){
        newProcess(&proc[j], j, aux->opt, aux->pes0, aux->idx->bwt,\
                aux->idx->bns, aux->idx->pac, aux->kmerHashTab,\
                &(aux->loadedIndex));
        newTransfer(&tran[j], j);
        tran[j]->fd_outfile = aux->fd_outfile;
    }

    struct timespec start, end;
    double walltime;

    //dataloader(aux->ks, aux->ks2, INT_MAX, aux->copy_comment, loaded, g3_opt);
    //offloader(loaded, proc, tran, g3_opt);//, superbatch_results[iter]);

    long int num_total_reads = 0;
    clock_gettime(CLOCK_REALTIME,  &start);

#define A 0
#define B 1
#define toggle(ab) (1 - (ab))
    int AB = A;
    do {
        std::thread t_dataloader(dataloader, aux->ks, aux->ks2, INT_MAX, aux->copy_comment, loading, g3_opt);
        std::thread t_offloader(offloader, loaded, proc, tran, g3_opt);//, superbatch_results[AB]);
        //std::thread t_storer(storer, aux->fd_outfile, superbatch_results[toggle(AB)]); // possibly merge pulled results

        t_offloader.join();
        t_dataloader.join();

        superbatch_data_t * tmp = loaded;
        loaded = loading;
        loading = tmp;
        resetSuperBatchData(loading);
        AB = toggle(AB);
        fprintf(stderr, "main_gcube: loaded->n_reads %d\n", loaded->n_reads);
        num_total_reads += loaded->n_reads;
    } while (loaded->n_reads != 0);

    clock_gettime(CLOCK_REALTIME,  &end);
    walltime = (end.tv_sec - start.tv_sec) +\
                           (end.tv_nsec - start.tv_nsec) / 1e9;
    fprintf(stderr,"\n\nWall-clock time for processing all %ld reads: %.6lf seconds\n\n\n", num_total_reads, walltime);

    for(int j=0; j<num_use_gpus; j++){
        cudaStreamDestroy(*(cudaStream_t*)proc[j]->CUDA_stream);
        cudaStreamDestroy(*(cudaStream_t*)tran[j]->CUDA_stream);
    }
}
