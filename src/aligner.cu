#include "timer.h"
#include "macro.h"
#include "bwa.h"
#include <locale.h>
#include "bwamem_GPU.cuh"
#include "streams.cuh"
#include <future>
#include <iostream>
#include <iomanip>
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

extern float tprof[MAX_NUM_GPUS][MAX_NUM_STEPS];

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
static void push(transfer_data_t *tran, superbatch_data_t *loaded_input,
        int *push_counter, std::mutex *push_m, g3_opt_t *g3_opt,
        int *actual_push_count,
        double *func_elapsed_ms)
{
    FUNC_TIMER_START;
    int push_count;
    int push_offset;

    push_m->lock();
    push_offset = *push_counter;
    tran->batch_offset = loaded_input->loading_batch_offset + push_offset;

    if(push_offset >= loaded_input->n_reads){
        push_count = 0;
    } else{
        if(loaded_input->n_reads < g3_opt->batch_size){
            push_count = loaded_input->n_reads;
        } else{
            if(push_offset <= loaded_input->n_reads - g3_opt->batch_size){
                push_count = g3_opt->batch_size;
            } else if(push_offset < loaded_input->n_reads){
                push_count = loaded_input->n_reads - push_offset;
            } 
        }
    }

    *push_counter += push_count;
    push_m->unlock();


    if(push_count == 0){
        tran->n_seqs = 0;
        *actual_push_count = 0;
        FUNC_TIMER_END;
        return;
    }

    // Push inputs to GPU device. ASSUMED MEMCPY DOES NOT FAIL
	resetTransfer(tran);
    tran->n_seqs = push_count;

	copyReads2PinnedMem(loaded_input, tran, push_offset, push_count);
	// at this point, all pointers on tran still point to name, seq, comment, qual addresses on loaded
	// translate reads' addresses to GPU addresses
	convert2DevAddr(tran);
	CUDATransferSeqsIn(tran);

    *actual_push_count = push_count;
    FUNC_TIMER_END;
    return;
}


/**
 * @brief output the previous batch of reads.
 * first transfer device's seqio to host's seq_io.
 * then write from host's seq_io to output
 * 
 * @param first_batch 
 * @param transfer_data 
 */

static void pull(transfer_data_t *tran, double *func_elapsed_ms)
{
    FUNC_TIMER_START;
    if (tran->n_seqs==0){
        FUNC_TIMER_END;
        return;
    }

    // Pull alignment results of the previous batch.
    int total_alns_num;
    cudaMemcpy(&total_alns_num, tran->d_total_alns_num, sizeof(int),
            cudaMemcpyDeviceToHost);

    int *alnID_offset = (int*)malloc(sizeof(int) * total_alns_num);
    int *alnID_num = (int*)malloc(sizeof(int) * total_alns_num);
    cudaMemcpy(alnID_offset, tran->d_alns_offset, sizeof(int) * total_alns_num,
            cudaMemcpyDeviceToHost);
    cudaMemcpy(alnID_num, tran->d_alns_num, sizeof(int) * total_alns_num,
            cudaMemcpyDeviceToHost);


    int *rid = (int*)malloc(sizeof(int) * total_alns_num);
    int64_t *pos = (int64_t*)malloc(sizeof(int64_t) * total_alns_num);

    int total_cigar_len;
    cudaMemcpy(&total_cigar_len, tran->d_total_cigar_len, sizeof(int),
            cudaMemcpyDeviceToHost);
    int *cigar = (int*)malloc(sizeof(int) * total_cigar_len);
    cudaMemcpy(cigar, tran->d_alns_cigar, sizeof(int) * total_cigar_len,
            cudaMemcpyDeviceToHost);

    int *cigar_len = (int*)malloc(sizeof(int) * total_alns_num);
    cudaMemcpy(cigar_len, tran->d_alns_cigar_len, sizeof(int) * total_alns_num,
            cudaMemcpyDeviceToHost);
    uint32_t *cigar_offset = (uint32_t*)malloc(sizeof(uint32_t) * total_alns_num);
    cudaMemcpy(cigar_offset, tran->d_alns_cigar_offset, sizeof(uint32_t) * total_alns_num,
            cudaMemcpyDeviceToHost);


    // TODO Generate SAM strings.

    FUNC_TIMER_END;
    return;

    // TODO Sequentially write to the output file. -> delegate it to the host.cu level.
}

static void offloader(int gpuid, superbatch_data_t *loadedinput,
        process_data_t *batch_A, transfer_data_t *batch_B,
        int *push_counter, std::mutex *push_m, g3_opt_t *g3_opt)
{
    int this_gpuid;
    if(cudaSetDevice(gpuid) != cudaSuccess){
       std::cerr << "Offloader for GPU no. " << gpuid 
           << " : cudaSetDevice failed" << std::endl;
        return;
    }
    cudaGetDevice(&this_gpuid);
    if(this_gpuid != gpuid){
       std::cerr << "Offloader for GPU no. " << gpuid 
           << " : cudaSetDevice failed" << std::endl;
        return;
    } 

    double compute_ms, pull_ms, push_ms;
    int push_count, pull_count;
    while(true){
        std::thread t_compute(bwa_align, gpuid, batch_A, g3_opt, &compute_ms);
        pull_count = batch_B->n_seqs;
        pull(batch_B, &pull_ms);
        push(batch_B, loadedinput, push_counter, push_m, g3_opt, 
                &push_count, &push_ms);
        t_compute.join();
        tprof[gpuid][COMPUTE_TOTAL] += (float)compute_ms;
        tprof[gpuid][PULL_TOTAL] += (float)pull_ms;
        tprof[gpuid][PUSH_TOTAL] += (float)push_ms;

        /*
        std::cerr << "* GPU #" << gpuid << " | ";
        std::cerr << std::fixed << std::setprecision(2) << std::setw(8) 
            << compute_ms;
        std::cerr << "ms | GPU-computed " << batch_A->n_seqs << " reads" << std::endl;

        std::cerr << "* GPU #" << gpuid << " | ";
        std::cerr << std::fixed << std::setprecision(2) << std::setw(8) 
            << pull_ms;
        std::cerr << "ms | pulled " << pull_count << " reads (prev. batch)"
            << std::endl;

        std::cerr << "* GPU #" << gpuid << " | ";
        std::cerr << std::fixed << std::setprecision(2) << std::setw(8) 
            << push_ms;
        std::cerr << "ms | pushed " << push_count << " reads (next batch)"
            << std::endl;
            */

        swapData(batch_A, batch_B);
        if(batch_A->n_seqs == 0){
            pull(batch_B, &pull_ms);
            tprof[gpuid][PULL_TOTAL] += (float)pull_ms;

            /*
            std::cerr << "* GPU #" << gpuid << " | ";
            std::cerr << std::fixed << std::setprecision(2) << std::setw(8) 
                << pull_ms;
            std::cerr << "ms | pulled " << pull_count
                << " reads (prev. FINAL batch)" << std::endl;
                */
            break;
        }
    }

    resetTransfer(batch_B);
	return;
}

void aligner(superbatch_data_t *loadedinput, process_data_t *proc[MAX_NUM_GPUS],
        transfer_data_t *tran[MAX_NUM_GPUS], g3_opt_t *g3_opt,
        double *func_elapsed_ms)
{
    FUNC_TIMER_START;
    std::cerr << "* aligning " << loadedinput->n_reads << " reads with " 
        << g3_opt->num_use_gpus << " GPUs" << std::endl;
    if(loadedinput->n_reads == 0){
        FUNC_TIMER_END;
        return;
    }
    std::thread t_offloader[MAX_NUM_GPUS];
    int push_counter = 0;
    std::mutex push_m;


    int gpuid;
    for(gpuid = 0; gpuid < g3_opt->num_use_gpus; gpuid++){
        t_offloader[gpuid] = std::thread(offloader, gpuid, loadedinput,
                proc[gpuid], tran[gpuid], &push_counter, &push_m, g3_opt);
    }
    for(gpuid = 0; gpuid < g3_opt->num_use_gpus; gpuid++){
        t_offloader[gpuid].join();
    }

    FUNC_TIMER_END;
    return;
}
