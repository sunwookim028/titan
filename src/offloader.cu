#include "offloader.h"
#include "bwa.h"
#include <locale.h>
#include "bwamem_GPU.cuh"
#include "batch_config.h"
#include "streams.cuh"
#include <future>
#include <iostream>
#include <atomic>
#include <mutex>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <iostream>
#include <ostream>
#include <thread>


/**
 * @brief convert current host addresses on a minibatch's transfer_data to their (future) addresses on GPU
 * assuming name, seq, comment, qual pointers on trasnfer_data still points to host memory
 *
 * @param seqs
 * @param n_seqs
 * @param transfer_data transfer_data_t object where these reads reside
 */
void convert2DevAddr(transfer_data_t *transfer_data)
{
	auto reads = transfer_data->h_seqs;
	auto n_reads = transfer_data->n_seqs;
	auto first_read = reads[0];
	for (int i = 0; i < n_reads; i++)
	{
		reads[i].name = reads[i].name - first_read.name + transfer_data->d_seq_name_ptr;
		reads[i].seq = reads[i].seq - first_read.seq + transfer_data->d_seq_seq_ptr;
		reads[i].comment = reads[i].comment - first_read.comment + transfer_data->d_seq_comment_ptr;
		reads[i].qual = reads[i].qual - first_read.qual + transfer_data->d_seq_qual_ptr;
	}
}

/**
 * @brief copy a minibatch of n_reads from superbatch to transfer_data minibatch's pinned memory, starting from firstReadId. 
 * Read info are contiguous, but name, comment, seq, qual are not
 * 
 * @param superbatch_data
 * @param transfer_data 
 * @param firstReadId 
 * @param n_reads 
 */
void copyReads2PinnedMem(superbatch_data_t *superbatch_data, transfer_data_t *transfer_data, int firstReadId, int n_reads){
	int lastReadId = firstReadId + n_reads - 1; 
	// copy name, comment, seq, qual one by one
	for (int i = firstReadId; i <= lastReadId; i++){
		bseq1_t *read = &(superbatch_data->reads[i]);
		char *toAddr;

		toAddr = transfer_data->h_seq_seq_ptr + transfer_data->h_seq_seq_size;
		memcpy(toAddr, read->seq, read->l_seq + 1); // size + 1 for null-terminating char
		read->seq = toAddr;
		transfer_data->h_seq_seq_size += read->l_seq + 1;
	}
	// copy read info
	memcpy(transfer_data->h_seqs, &superbatch_data->reads[firstReadId], n_reads * sizeof(bseq1_t));
}

/**
 * @brief load a small batch from superbatch to transfer_data, up to MB_MAX_COUNT. 
 * Return number of reads loaded into transfer_data->n_seqs. return 0 if no read loaded
 * after loading, translate reads' addresses to GPU and transfer to GPU,
 * @param transfer_data
 * @param superbatch_data
 * @param num_loaded number of reads loaded from this superbatch before this minibatch
 * @return int number of reads loaded into transfer_data->n_seqs
 */
static void push(transfer_data_t *tran, superbatch_data_t *loadedinput, int *push_counter, std::mutex *push_m)
{
    int push_count;
    int push_offset;

    push_m->lock();
    push_offset = *push_counter;
    if(*push_counter <= loadedinput->n_reads - 0.9*MB_MAX_COUNT){
        push_count = 0.9*MB_MAX_COUNT;
    } else if(*push_counter < loadedinput->n_reads){
        push_count = loadedinput->n_reads - *push_counter;
    } else{
        push_count = 0;
    }
    *push_counter += push_count;
    push_m->unlock();

#ifdef VERBOSE
    if(push_count > 0){
        std::cerr << "Pushing input no. " << push_offset << " ~ no. "
            << push_offset + push_count - 1 << std::endl;
    }
#endif

    if(push_count == 0){
        tran->n_seqs = 0;
        return;
    }

    // Push inputs to GPU device. ASSUMED MEMCPY DOES NOT FAIL
	resetTransfer(tran);
    tran->n_seqs = push_count;

	copyReads2PinnedMem(loadedinput, tran, push_offset, push_count);
	// at this point, all pointers on tran still point to name, seq, comment, qual addresses on loaded
	// translate reads' addresses to GPU addresses
	convert2DevAddr(tran);
	CUDATransferSeqsIn(tran);
}


/**
 * @brief output the previous batch of reads.
 * first transfer device's seqio to host's seq_io.
 * then write from host's seq_io to output
 * 
 * @param first_batch 
 * @param transfer_data 
 */

static void pull(transfer_data_t *tran, int *pull_counter, std::mutex *pull_m)
{
    int k = 0;
    int pull_offset;
    int pull_count = tran->n_seqs;

    if (pull_count==0) return;

    pull_m->lock();
    pull_offset = *pull_counter;
    *pull_counter += pull_count;
    pull_m->unlock();

#ifdef VERBOSE
    std::cerr << "Pulling input no. " << pull_offset << " ~ no. "
        << pull_offset + pull_count - 1 << std::endl;
#endif

	//CUDATransferSamOut(tran); // FIXME

	// write from host's seq_io to output
#define BUFLEN 1024
	//bseq1_t *seqs = tran->h_seqs;
	for (int i = 0; i < pull_count; ++i){ // aggregate from memory then write to file
		//if (seqs[i].sam){
            //err_fputs(seqs[i].sam, stdout); // FIXME
            //pwrite(tran->fd_outfile, "Hello!", BUFLEN, pull_offset * BUFLEN);
            //write(tran->fd_outfile, "spider-man", 10);
        //} 
    }
}

static void deviceoffloader(int gpuid, superbatch_data_t *loadedinput,\
        process_data_t *batch_A, transfer_data_t *batch_B,\
        int *pull_counter, int *push_counter,\
        std::mutex *pull_m, std::mutex *push_m, g3_opt_t *g3_opt)
{
    if(cudaSetDevice(gpuid) != cudaSuccess){
       std::cerr << "Offloader for GPU no. " << gpuid 
           << " : cudaSetDevice failed" << std::endl;
        return;
    } 

    while(true){
        std::thread t_align(bwa_align, gpuid, batch_A, g3_opt);
        pull(batch_B, pull_counter, pull_m);
        push(batch_B, loadedinput, push_counter, push_m);
        t_align.join();

        swapData(batch_A, batch_B);
        if(batch_A->n_seqs == 0){
            pull(batch_B, pull_counter, pull_m);
            break;
        }
    }

    resetTransfer(batch_B);
	return;
}

void offloader(superbatch_data_t *loadedinput, process_data_t *proc[MAX_NUM_GPUS],\
        transfer_data_t *tran[MAX_NUM_GPUS], g3_opt_t *g3_opt)
{
    std::cerr << "CPU Batch: Processing # " << loadedinput->n_reads
        << "reads" << std::endl;
    std::thread perGPU[MAX_NUM_GPUS];
    int pull_counter = 0;
    int push_counter = 0;
    std::mutex push_m, pull_m;

    int num_use_gpus = g3_opt->num_use_gpus;

    for(int j=0; j<num_use_gpus; j++){
        perGPU[j] = std::thread(deviceoffloader, j, loadedinput, proc[j], tran[j],\
                &pull_counter, &push_counter, &pull_m, &push_m, g3_opt);
    }
    for(int j=0; j<num_use_gpus; j++){
        perGPU[j].join();
    }
}
