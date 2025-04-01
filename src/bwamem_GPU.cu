#include "utils_CUDA.cuh"
#include "macro.h"
#include "timer.h"
#include "errHandler.cuh"
#include "bwa.h"
#include "bwamem_GPU.cuh"
#include "CUDAKernel_memmgnt.cuh"
#include "bwt_CUDA.cuh"
#include "bntseq_CUDA.cuh"
#include "bwa_CUDA.cuh"
#include <string.h>
#include "streams.cuh"
#include "batch_config.h"
#include "hashKMerIndex.h"
#include "seed.cuh"
#include "preprocessing.cuh"
#include <fstream>
#include <chrono>
using namespace std::chrono;
#include <iostream>
using namespace std;

#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "aux.cuh"
#include "host.h"

#define PRINT(LABEL) \
    g3_opt->print_mask & BIT(LABEL)

#define CudaEventSynchronize(event)\
{\
    cudaError_t err;\
    err = cudaEventSynchronize(event);\
    if (err != cudaSuccess) {\
         std::cerr << "GPU " << gpuid << "  " << "CUDA error: " << cudaGetErrorString(err) << std::endl;\
    }\
}

#define TIMER_INIT \
    cudaEvent_t timer_event_start, timer_event_stop;\
    cudaEventCreate(&timer_event_start);\
    cudaEventCreate(&timer_event_stop);

#define TIMER_DESTROY \
    cudaEventDestroy(timer_event_start);\
    cudaEventDestroy(timer_event_stop);

#define TIMER_START(lap) \
    lap = 0;\
    cudaEventRecord(timer_event_start, *(cudaStream_t*)proc->CUDA_stream);\

#define TIMER_END(lap) \
    cudaEventRecord(timer_event_stop, *(cudaStream_t*)proc->CUDA_stream);\
    CudaEventSynchronize(timer_event_stop);\
    cudaEventElapsedTime(&lap, timer_event_start, timer_event_stop);\

#define LAUNCH_CHK(stream)\
{\
    cudaStreamSynchronize(stream);\
    cudaError_t err;\
    err = cudaGetLastError();\
    if(err != cudaSuccess)\
    {\
      fprintf(stderr,"GPU %d cudaGetLastError(): %s %s %d\n", gpuid, cudaGetErrorString(err), __FILE__, __LINE__);\
    }\
}


// should match the strings in utils_CUDA.cu
const char * const h_phasename[] = {
    /* _DETAIL     */ "dummy",
    /* _STAGE_TIME */ "dummy",
    /* _STEP_TIME  */ "dummy",
    /* _SMEM       */ "smem",
    /* _INTV       */ "intv",
    /* _SEED       */ "seed",
    /* _STSEED     */ "stseed",
    /* _CHAIN      */ "chain",
    /* _STCHAIN    */ "stchain",
    /* _FTCHAIN    */ "ftchain",
    /* _EXPAIR     */ "expair",
    /* _REGION     */ "region",
    /* _FTREGION   */ "ftregion",
    /* _STREGION   */ "stregion",
    /* _TBPAIR     */ "tbpair",
    /* _ALIGNMENT  */ "alignment",
    /* _RESULT     */ "result",
};

extern float tprof[MAX_NUM_GPUS][MAX_NUM_STEPS];

/*  main function for bwamem in GPU 
 *  return to seqs.sam
 *
 *
 * Stage    |   Substage        |   Step          |   note
 * ---------------------------------------------------------------------------
 * Seeding  |   SMEM seeding    |   seed          |   seeding core
 *          |   Reseeding       |   r2            |
 *          |                   |   r3            |
 * Chaining |   B-tree chaining |   sal           |
 *          |                   |   sort_seeds    |
 *          |                   |   chain         |   chaining core
 *          |                   |   sort_chains   |
 *          |                   |   filter        |
 * Extending|   Local extending |   pairgen       |
 *          |                   |   extend        |   extending core
 *          |                   |   filter_mark   |
 *          |                   |   sort_alns     |
 *          |   Traceback       |   pairgen       |
 *          |                   |   traceback     |   traceback core (w/ NM test)
 *          |                   |   finalize      |
 */
void bwa_align(int gpuid, process_data_t *proc, g3_opt_t *g3_opt,
        double *func_elapsed_ms)
{
    FUNC_TIMER_START;
    float step_lap;

    void *d_temp_storage;
    size_t temp_storage_size;
    int n_seqs, num_seeds_to_extend;

    // Configure the GPU to use.
    int this_gpuid;
    if(cudaSetDevice(gpuid) != cudaSuccess){
        std::cerr << "bwa_align for GPU no. " << gpuid 
            << " : cudaSetDevice failed" << std::endl;
        FUNC_TIMER_END;
        return;
    }
    cudaGetDevice(&this_gpuid);
    if(this_gpuid != gpuid){
        std::cerr << "bwa_align for GPU no. " << gpuid 
            << " : cudaSetDevice failed" << std::endl;
        FUNC_TIMER_END;
        return;
    } 

    // Initialize variables for aligning this batch of read sequences.
    if((n_seqs = proc->n_seqs) == 0){
        FUNC_TIMER_END;
        return;
    }
    if(PRINT(_DETAIL)){
        std::cerr << "GPU " << gpuid << "  " << "Aligning " << n_seqs << " seqs" << std::endl;
    }
    CUDAResetBufferPool(proc->d_buffer_pools, g3_opt->batch_size);
    gpuErrchk( cudaMemset(proc->d_Nseeds, 0, sizeof(int)) );

    TIMER_INIT; // This should be placed after setting GPU.

    // (1/3) Seeding

    // SMEM seeding (Seeding 1/2)
    TIMER_START(step_lap);
    if(g3_opt->baseline){
        PREPROCESS_convert_bit_encoding_kernel 
            <<< n_seqs, 32, 0, *(cudaStream_t*)proc->CUDA_stream >>> (proc->d_seqs);
        LAUNCH_CHK(*(cudaStream_t*)proc->CUDA_stream);

        MEMFINDING_collect_intv_kernel 
            <<< n_seqs, 320, 512, *(cudaStream_t*)proc->CUDA_stream >>> (
                    proc->d_opt, proc->d_bwt, proc->d_seqs,
                    proc->d_aux,	// output
                    proc->d_kmerHashTab,
                    proc->d_buffer_pools);
        LAUNCH_CHK(*(cudaStream_t*)proc->CUDA_stream);

        filterSeeds <<< n_seqs, WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream >>>(
                proc->d_opt, proc->d_aux);
        LAUNCH_CHK(*(cudaStream_t*)proc->CUDA_stream);
    } else{
        preseedAndFilterV2 <<< n_seqs, 320, 0, *(cudaStream_t*)proc->CUDA_stream >>> (
                proc->d_fmIndex,
                proc->d_opt, proc->d_seqs, proc->d_aux, proc->d_kmerHashTab, proc->d_buffer_pools);
        LAUNCH_CHK(*(cudaStream_t*)proc->CUDA_stream);
    }
    TIMER_END(step_lap);
    tprof[gpuid][S_SMEM] += step_lap;

    if(PRINT(_SMEM) || PRINT(_ALL_SEEDING)){
        for(int readID = 0; readID < n_seqs; readID++)
        {
            printIntv<<<1, WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream>>>(proc->d_aux, readID, _SMEM);
            LAUNCH_CHK(*(cudaStream_t*)proc->CUDA_stream);
        }
    }

    // Reseeding (Seeding 2/2)
    TIMER_START(step_lap);
    reseedV2 <<< ceil((float)n_seqs / WARPSIZE) == 0 ? 1 
        : ceil((float)n_seqs / WARPSIZE), WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream >>>(
                proc->d_fmIndex, proc->d_opt, proc->d_seqs, proc->d_aux, proc->d_kmerHashTab, 
                proc->d_buffer_pools, n_seqs);
    LAUNCH_CHK(*(cudaStream_t*)proc->CUDA_stream);

    TIMER_END(step_lap);
    tprof[gpuid][S_R2] += step_lap;

    TIMER_START(step_lap);
    reseedLastRound <<< ceil((float)n_seqs / WARPSIZE) == 0 ? 1 
        : ceil((float)n_seqs / WARPSIZE), WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream >>>(
                proc->d_fmIndex, proc->d_opt, proc->d_seqs, proc->d_aux, proc->d_kmerHashTab, n_seqs);
    LAUNCH_CHK(*(cudaStream_t*)proc->CUDA_stream);

    TIMER_END(step_lap);
    tprof[gpuid][S_R3] += step_lap;

    if(PRINT(_STEP_TIME)){
        std::cerr << "GPU " << gpuid << "  " << "Reseeding: " << step_lap 
            << " ms" << std::endl;
    }

    if(PRINT(_INTV) || PRINT(_ALL_SEEDING)){
        for(int readID = 0; readID < n_seqs; readID++)
        {
            printIntv<<<1, WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream>>>(proc->d_aux, readID, _INTV);
            LAUNCH_CHK(*(cudaStream_t*)proc->CUDA_stream);
        }
    }


    // (2/3) Chaining

    // B-tree chaining (Chaining 1/1)
    TIMER_START(step_lap);
    saLookup <<< n_seqs, 128, 0, *(cudaStream_t*)proc->CUDA_stream >>> (
            proc->d_opt, proc->d_bwt, proc->d_bns, proc->d_seqs, proc->d_aux,
            proc->d_seq_seeds,
            proc->d_buffer_pools);
    LAUNCH_CHK(*(cudaStream_t*)proc->CUDA_stream);

    TIMER_END(step_lap);
    tprof[gpuid][C_SAL] += step_lap;

    if(PRINT(_SEED) || PRINT(_ALL_CHAINING)){
        for(int readID = 0; readID < n_seqs; readID++)
        {
            printSeed<<<1, WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream>>>(proc->d_seq_seeds, readID, _SEED);
            LAUNCH_CHK(*(cudaStream_t*)proc->CUDA_stream);
        }
    }


    TIMER_START(step_lap);
    sortSeedsLowDim 
        <<< n_seqs, SORTSEEDSLOW_BLOCKDIMX, 0, *(cudaStream_t*)proc->CUDA_stream >>> (
                proc->d_seq_seeds,
                proc->d_buffer_pools);
    LAUNCH_CHK(*(cudaStream_t*)proc->CUDA_stream);

    sortSeedsHighDim 
        <<< n_seqs, SORTSEEDSHIGH_BLOCKDIMX, 0, *(cudaStream_t*)proc->CUDA_stream >>> (
                proc->d_seq_seeds,
                proc->d_buffer_pools);
    LAUNCH_CHK(*(cudaStream_t*)proc->CUDA_stream);

    TIMER_END(step_lap);
    tprof[gpuid][C_SORT_SEEDS] += step_lap;

    if(PRINT(_STSEED) || PRINT(_ALL_CHAINING)){
        for(int readID = 0; readID < n_seqs; readID++)
        {
            printSeed<<<1, WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream>>>(proc->d_seq_seeds, readID, _STSEED);
            LAUNCH_CHK(*(cudaStream_t*)proc->CUDA_stream);
        }
    }

    TIMER_START(step_lap);
    if(g3_opt->baseline){
#if 0
#define SEEDCHAINING_CHAIN_BLOCKDIMX 256
        SEEDCHAINING_chain_kernel <<< n_seqs, SEEDCHAINING_CHAIN_BLOCKDIMX, 0, *(cudaStream_t*)proc->CUDA_stream >>> (
                proc->d_opt, proc->d_bns, proc->d_seqs, proc->d_seq_seeds,
                proc->d_chains,	// output
                proc->d_buffer_pools);
#else
        BTreeChaining 
            <<< ceil((float)n_seqs / WARPSIZE) == 0 ? 1 : ceil((float)n_seqs / WARPSIZE), WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream >>>(
                    proc->d_opt, proc->d_bns, proc->d_seqs, proc->d_seq_seeds,
                    proc->d_chains,	// output
                    proc->d_buffer_pools);
#endif
    } else{
        BTreeChaining 
            <<< ceil((float)n_seqs / WARPSIZE) == 0 ? 1 : ceil((float)n_seqs / WARPSIZE), WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream >>>(

                    proc->d_opt, proc->d_bns, proc->d_seqs, proc->d_seq_seeds,
                    proc->d_chains,	// output
                    proc->d_buffer_pools);
    }
    LAUNCH_CHK(*(cudaStream_t*)proc->CUDA_stream);

    TIMER_END(step_lap);
    tprof[gpuid][C_CHAIN] += step_lap;

    if(PRINT(_CHAIN) || PRINT(_ALL_CHAINING)){
        for(int readID = 0; readID < n_seqs; readID++){
            printChain<<<1, WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream>>>(proc->d_chains, readID, _CHAIN);
            LAUNCH_CHK(*(cudaStream_t*)proc->CUDA_stream);
        }
    }

    TIMER_START(step_lap);
    sortChainsDecreasingWeight 
        <<< n_seqs, SORTCHAIN_BLOCKDIMX, 
        MAX_N_CHAIN*2*sizeof(uint16_t)+sizeof(mem_chain_t**), *(cudaStream_t*)proc->CUDA_stream>>> 
            (proc->d_chains, proc->d_buffer_pools);
    LAUNCH_CHK(*(cudaStream_t*)proc->CUDA_stream);

    TIMER_END(step_lap);
    tprof[gpuid][C_SORT_CHAINS] += step_lap;

    if(PRINT(_STCHAIN) || PRINT(_ALL_CHAINING)){
        for(int readID = 0; readID < n_seqs; readID++)
        {
            printChain<<<1, WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream>>>(proc->d_chains, readID, _STCHAIN);
            LAUNCH_CHK(*(cudaStream_t*)proc->CUDA_stream);
        }
    }

    TIMER_START(step_lap);
    CHAINFILTERING_filter_kernel <<< n_seqs, CHAIN_FLT_BLOCKSIZE, MAX_N_CHAIN*(3*sizeof(uint16_t)+sizeof(uint8_t)), *(cudaStream_t*)proc->CUDA_stream >>> (
            proc->d_opt, 
            proc->d_chains, 	// input and output
            proc->d_buffer_pools);
    LAUNCH_CHK(*(cudaStream_t*)proc->CUDA_stream);

    TIMER_END(step_lap);
    tprof[gpuid][C_FILTER] += step_lap;

#if 0 // this code is anyway dead for short reads of our concern (<700bp).
    TIMER_START(lap);
    CHAINFILTERING_flt_chained_seeds_kernel <<< ceil((float)n_seqs / WARPSIZE) == 0 ? 1 : ceil((float)n_seqs / WARPSIZE), WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream >>>(
            proc->d_opt, proc->d_bns, proc->d_pac,
            proc->d_seqs, proc->d_chains, 	// input and output
            n_seqs, proc->d_buffer_pools);
    TIMER_END(lap);
    step_lap += lap;
#endif

    if(PRINT(_FTCHAIN) || PRINT(_ALL_CHAINING)){
        for(int readID = 0; readID < n_seqs; readID++)
        {
            printChain<<<1, WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream>>>(proc->d_chains, readID, _FTCHAIN);
            LAUNCH_CHK(*(cudaStream_t*)proc->CUDA_stream);
        }
    }


    // (3/3) Extending

    // Extending -> Local extending (1/2)

    TIMER_START(step_lap);
    SWSeed <<< ceil((float)n_seqs / WARPSIZE) == 0 ? 1 : ceil((float)n_seqs / WARPSIZE), WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream >>>(
            proc->d_chains, proc->d_regs, proc->d_seed_records, proc->d_Nseeds, n_seqs,
            proc->d_buffer_pools
            );
    LAUNCH_CHK(*(cudaStream_t*)proc->CUDA_stream);

    gpuErrchk2( cudaMemcpy(&num_seeds_to_extend, proc->d_Nseeds, sizeof(int), cudaMemcpyDeviceToHost));

    if(num_seeds_to_extend==0){
        proc->n_processed += proc->n_seqs;
        //gpuErrchk( cudaStreamSynchronize(*(cudaStream_t*)proc->CUDA_stream) );
        return;
    }

    ExtendingPairGenerate <<< ceil((float)num_seeds_to_extend/32.0), 32, 0, *(cudaStream_t*)proc->CUDA_stream >>> (
            proc->d_opt, proc->d_bns, proc->d_pac, proc->d_seqs,
            proc->d_chains, proc->d_regs, proc->d_seed_records, proc->d_Nseeds,
            n_seqs, proc->d_buffer_pools
            );
    LAUNCH_CHK(*(cudaStream_t*)proc->CUDA_stream);

    TIMER_END(step_lap);
    tprof[gpuid][E_PAIRGEN] += step_lap;

    if(PRINT(_DETAIL)){
        std::cerr << "GPU " << gpuid << "  " << "# local extending pairs: " << num_seeds_to_extend << std::endl;
    }

    if(PRINT(_EXPAIR) || PRINT(_ALL_EXTENDING)){
        printPair<<<1, WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream>>>(proc->d_seed_records, num_seeds_to_extend, _EXPAIR);
        LAUNCH_CHK(*(cudaStream_t*)proc->CUDA_stream);
    }


    TIMER_START(step_lap);
    if(g3_opt->baseline){
        localExtending_baseline <<< num_seeds_to_extend, WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream >>> (
                proc->d_opt,
                proc->d_chains, 		// input chains
                proc->d_seed_records,
                proc->d_regs,		// output array
                proc->d_Nseeds);
    } else{
        localExtending <<< num_seeds_to_extend, WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream >>> (
                proc->d_opt,
                proc->d_chains, 		// input chains
                proc->d_seed_records,
                proc->d_regs,		// output array
                proc->d_Nseeds);
        // TODO adopt agatha
    }
    LAUNCH_CHK(*(cudaStream_t*)proc->CUDA_stream);

    TIMER_END(step_lap);
    tprof[gpuid][E_EXTEND] += step_lap;

    if(PRINT(_REGION) || PRINT(_ALL_EXTENDING)){
        for(int readID = 0; readID < n_seqs; readID++)
        {
            printReg<<<1, WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream>>>(proc->d_regs, readID, _REGION);
            LAUNCH_CHK(*(cudaStream_t*)proc->CUDA_stream);
        }
    }

    // remove duplicates
    TIMER_START(step_lap);
    filterRegions <<< n_seqs, 320, 0, *(cudaStream_t*)proc->CUDA_stream >>> (
            proc->d_opt, proc->d_bns,
            proc->d_chains, proc->d_regs, proc->d_buffer_pools
            );
    LAUNCH_CHK(*(cudaStream_t*)proc->CUDA_stream);

    TIMER_END(step_lap);
    tprof[gpuid][E_FILTER_MARK] += step_lap;

    if(PRINT(_FTREGION) || PRINT(_ALL_EXTENDING)){
        for(int readID = 0; readID < n_seqs; readID++)
        {
            printReg<<<1, WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream>>>(proc->d_regs, readID, _FTREGION);
            LAUNCH_CHK(*(cudaStream_t*)proc->CUDA_stream);
        }
    }


    TIMER_START(step_lap);
    sortRegions <<< ceil((float)n_seqs / WARPSIZE) == 0 ? 1 : ceil((float)n_seqs / WARPSIZE), WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream >>>(
            proc->d_regs, n_seqs, proc->d_buffer_pools);
    LAUNCH_CHK(*(cudaStream_t*)proc->CUDA_stream);


    TIMER_END(step_lap);
    tprof[gpuid][E_SORT_ALNS] += step_lap;

    if(PRINT(_STREGION) || PRINT(_ALL_EXTENDING)){
        for(int readID = 0; readID < n_seqs; readID++)
        {
            printReg<<<1, WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream>>>(proc->d_regs, readID, _STREGION);
            LAUNCH_CHK(*(cudaStream_t*)proc->CUDA_stream);
        }
    }


    // Extending -> Traceback (2/2)

    TIMER_START(step_lap);
    gpuErrchk2( cudaMemset(proc->d_Nseeds, 0, sizeof(int)));

    FINALIZEALN_preprocessing1_kernel <<< ceil((float)n_seqs / WARPSIZE) == 0 ? 1 : ceil((float)n_seqs / WARPSIZE), WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream >>>(
            proc->d_regs, proc->d_alns, proc->d_seed_records, proc->d_Nseeds,
            proc->d_buffer_pools);
    LAUNCH_CHK(*(cudaStream_t*)proc->CUDA_stream);

    gpuErrchk2( cudaMemcpy(&num_seeds_to_extend, proc->d_Nseeds, sizeof(int), cudaMemcpyDeviceToHost));

    if(num_seeds_to_extend==0){
        proc->n_processed += proc->n_seqs;
        //gpuErrchk( cudaStreamSynchronize(*(cudaStream_t*)proc->CUDA_stream) );
        FUNC_TIMER_END;
        return;
    }

    FINALIZEALN_preprocessing2_kernel <<< ceil((float)num_seeds_to_extend / WARPSIZE) == 0 ? 1 : ceil((float)num_seeds_to_extend / WARPSIZE), WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream >>>(
            proc->d_opt, proc->d_seqs,
            proc->d_pac, proc->d_bns,
            proc->d_regs, proc->d_alns, proc->d_seed_records, num_seeds_to_extend,
            proc->d_sortkeys_in,	// sortkeys_in = bandwidth * rlen
            proc->d_seqIDs_in,
            proc->d_buffer_pools);
    LAUNCH_CHK(*(cudaStream_t*)proc->CUDA_stream);


    // reverse query and target if aln position is on reverse strand
    FINALIZEALN_reverseSeq_kernel <<< num_seeds_to_extend, WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream >>> (proc->d_seed_records, proc->d_alns, proc->d_buffer_pools);
    LAUNCH_CHK(*(cudaStream_t*)proc->CUDA_stream);



    d_temp_storage = NULL;
    temp_storage_size = 0;
    // determine temporary storage requirement
    gpuErrchk2( cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_size, proc->d_sortkeys_in, proc->d_sortkeys_out, proc->d_seqIDs_in, proc->d_seqIDs_out, num_seeds_to_extend, 0, 8*sizeof(int), *(cudaStream_t*)proc->CUDA_stream) );
    // Allocate temporary storage
    gpuErrchk2( cudaMalloc(&d_temp_storage, temp_storage_size) );
    // perform radix sort
    gpuErrchk2( cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_size, proc->d_sortkeys_in, proc->d_sortkeys_out, proc->d_seqIDs_in, proc->d_seqIDs_out, num_seeds_to_extend, 0, 8*sizeof(int), *(cudaStream_t*)proc->CUDA_stream) );
    cudaFree(d_temp_storage);

    TIMER_END(step_lap);
    tprof[gpuid][E_T_PAIRGEN] += step_lap;


    if(PRINT(_TBPAIR) || PRINT(_ALL_EXTENDING)){
        printPair<<<1, WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream>>>(proc->d_seed_records, num_seeds_to_extend, _TBPAIR);
        LAUNCH_CHK(*(cudaStream_t*)proc->CUDA_stream);
    }

    if(PRINT(_DETAIL)){
        std::cerr << "GPU " << gpuid << "  " << "# traceback pairs: " << num_seeds_to_extend << std::endl;
    }


    TIMER_START(step_lap);
    if(g3_opt->baseline){
        traceback_baseline <<< ceil((float)num_seeds_to_extend / WARPSIZE) == 0 ? 1 : ceil((float)num_seeds_to_extend / WARPSIZE), WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream >>>(
                proc->d_opt,
                proc->d_seed_records, num_seeds_to_extend, proc->d_alns, proc->d_seqIDs_out,
                proc->d_buffer_pools
                );
    } else{
        traceback<<< ceil((float)num_seeds_to_extend / WARPSIZE) == 0 ? 1 : ceil((float)num_seeds_to_extend / WARPSIZE), WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream >>>(
                proc->d_opt,
                proc->d_seed_records, num_seeds_to_extend, proc->d_alns, proc->d_seqIDs_out,
                proc->d_buffer_pools
                );
    }
    LAUNCH_CHK(*(cudaStream_t*)proc->CUDA_stream);

    TIMER_END(step_lap);
    tprof[gpuid][E_TRACEBACK] += step_lap;

    if(PRINT(_ALIGNMENT) || PRINT(_ALL_EXTENDING)){
        for(int readID = 0; readID < n_seqs; readID++)
        {
            printAln<<<1, WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream>>>(proc->d_bns, proc->d_alns, readID, _ALIGNMENT);
            LAUNCH_CHK(*(cudaStream_t*)proc->CUDA_stream);
        }
    }

    TIMER_START(step_lap);
    finalize<<< ceil((float)num_seeds_to_extend / WARPSIZE) == 0 ? 1 : ceil((float)num_seeds_to_extend / WARPSIZE), WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream >>>(
            proc->d_opt, proc->d_bns, proc->d_seqs,
            proc->d_regs, proc->d_alns, proc->d_seed_records, num_seeds_to_extend,
            proc->d_buffer_pools
            );
    LAUNCH_CHK(*(cudaStream_t*)proc->CUDA_stream);

    TIMER_END(step_lap);
    tprof[gpuid][E_FINALIZE] += step_lap;

    if(PRINT(_RESULT) || PRINT(_ALL_EXTENDING)){
        for(int readID = 0; readID < n_seqs; readID++)
        {
            printAln<<<1, WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream>>>(proc->d_bns, proc->d_alns, readID, _RESULT);
            LAUNCH_CHK(*(cudaStream_t*)proc->CUDA_stream);
        }
    }


    // pack alns for transferring.
    //TODO

    TIMER_DESTROY;
    if(PRINT(_BUFFER_USAGE)){
        printBufferInfoHost(proc->d_buffer_pools);
    }

    FUNC_TIMER_END;
    return;
};
