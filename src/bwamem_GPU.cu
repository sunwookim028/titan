#include "utils_CUDA.cuh"
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
    cudaEvent_t step_start, step_stop;\
    cudaEventCreate(&step_start);\
    cudaEventCreate(&step_stop);

#define TIMER_DESTROY \
    cudaEventDestroy(step_start);\
    cudaEventDestroy(step_stop);

#define TIMER_START(lap) \
    if(PRINT(_STEP_TIME) || PRINT(_STAGE_TIME)){\
        cudaEventRecord(step_start, *(cudaStream_t*)proc->CUDA_stream);\
    }

#define TIMER_END(lap) \
    if(PRINT(_STEP_TIME) || PRINT(_STAGE_TIME)){\
        cudaEventRecord(step_stop, *(cudaStream_t*)proc->CUDA_stream);\
        CudaEventSynchronize(step_stop);\
        cudaEventElapsedTime(&lap, step_start, step_stop);\
    }

#define LAUNCH_CHK(name, stream)\
{\
    cudaStreamSynchronize(stream);\
    cudaError_t err;\
    err = cudaGetLastError();\
    if(err != cudaSuccess)\
    {\
        fprintf(stderr, "CUDA error in kernel for computing %s: %s\n", name, cudaGetErrorString(err));\
        exit(EXIT_FAILURE);\
    }\
}

#define LAUNCH_CHK_PRINTING(name, stream)\
{\
    cudaStreamSynchronize(stream);\
    cudaError_t err;\
    err = cudaGetLastError();\
    if(err != cudaSuccess)\
    {\
        fprintf(stderr, "CUDA error in kernel for printing %s: %s\n", name, cudaGetErrorString(err));\
        exit(EXIT_FAILURE);\
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

/*  main function for bwamem in GPU 
    return to seqs.sam
 */
void bwa_align(int gpuid, process_data_t *proc, g3_opt_t *g3_opt)
{
    float stage_lap;
    float step_lap;
    float lap;

    void *d_temp_storage;
    size_t temp_storage_size;
    int n_seqs, num_seeds_to_extend;

    // Configure the GPU to use.
    int current;
    if(cudaSetDevice(gpuid) != cudaSuccess){
         std::cerr << "GPU " << gpuid << "  " << "bwa_align: cudaSetDevice failed" << std::endl;
        return;
    } 
    cudaGetDevice(&current);
    if(current != gpuid){
         std::cerr << "GPU " << gpuid << "  " << "bwa_align: cudaSetDevice is wrong" << std::endl;
        return;
    }

    // Initialize variables for aligning this batch of read sequences.
    if((n_seqs = proc->n_seqs) == 0){
        gpuErrchk( cudaStreamSynchronize(*(cudaStream_t*)proc->CUDA_stream) );
        return;
    }
    if(PRINT(_DETAIL)){
         std::cerr << "GPU " << gpuid << "  " << "Aligning " << n_seqs << " seqs" << std::endl;
    }
    CUDAResetBufferPool(proc->d_buffer_pools);
    gpuErrchk( cudaMemset(proc->d_Nseeds, 0, sizeof(int)) );

    TIMER_INIT; // This should be placed after setting GPU.

    // (1/3) Seeding
    stage_lap = 0;

    // SMEM seeding (Seeding 1/2)
    if(g3_opt->baseline){
        step_lap = 0;
        TIMER_START(lap);
        PREPROCESS_convert_bit_encoding_kernel 
            <<< n_seqs, 32, 0, *(cudaStream_t*)proc->CUDA_stream >>> (proc->d_seqs);
        GET_LAST_ERR();
        TIMER_END(lap);
        LAUNCH_CHK("S preproc", *(cudaStream_t*)proc->CUDA_stream);
        step_lap += lap;

        TIMER_START(lap);
        MEMFINDING_collect_intv_kernel 
            <<< n_seqs, 320, 512, *(cudaStream_t*)proc->CUDA_stream >>> (
                    proc->d_opt, proc->d_bwt, proc->d_seqs,
                    proc->d_aux,	// output
                    proc->d_kmerHashTab,
                    proc->d_buffer_pools);
        GET_LAST_ERR();
        TIMER_END(lap);
        LAUNCH_CHK("S smem preseeding", *(cudaStream_t*)proc->CUDA_stream);
        step_lap += lap;

        TIMER_START(lap);
        filterSeeds <<< n_seqs, WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream >>>(
                proc->d_opt, proc->d_aux);
        GET_LAST_ERR();
        TIMER_END(lap);
        LAUNCH_CHK("S smem filtering", *(cudaStream_t*)proc->CUDA_stream);
        step_lap += lap;
    } else{
        TIMER_START(lap);
        preseedAndFilterV2 <<< n_seqs, 320, 0, *(cudaStream_t*)proc->CUDA_stream >>> (
                proc->d_fmIndex,
                proc->d_opt, proc->d_seqs, proc->d_aux, proc->d_kmerHashTab, proc->d_buffer_pools);
        GET_LAST_ERR();
        TIMER_END(lap);
        LAUNCH_CHK("S smem", *(cudaStream_t*)proc->CUDA_stream);
        step_lap = lap;
    }

    if(PRINT(_STEP_TIME)){
         std::cerr << "GPU " << gpuid << "  " << "SMEM seeding: " << step_lap 
            << " ms" << std::endl;
    }
    stage_lap += step_lap;

    if(PRINT(_SMEM) || PRINT(_ALL_SEEDING)){
        for(int readID = 0; readID < n_seqs; readID++)
        {
            printIntv<<<1, WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream>>>(proc->d_aux, readID, _SMEM);
            GET_LAST_ERR();
            LAUNCH_CHK_PRINTING(h_phasename[_SMEM], *(cudaStream_t*)proc->CUDA_stream);
        }
    }

    // Reseeding (Seeding 2/2)
    step_lap = 0;
    TIMER_START(lap);
    reseedV2 <<< ceil((float)n_seqs / WARPSIZE) == 0 ? 1 
        : ceil((float)n_seqs / WARPSIZE), WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream >>>(
                proc->d_fmIndex, proc->d_opt, proc->d_seqs, proc->d_aux, proc->d_kmerHashTab, 
                proc->d_buffer_pools, n_seqs);
        GET_LAST_ERR();
    LAUNCH_CHK("S reseed", *(cudaStream_t*)proc->CUDA_stream);
    TIMER_END(lap);
    step_lap = lap;

    if(PRINT(_STEP_TIME)){
         std::cerr << "GPU " << gpuid << "  " << "SMEM seeding: " << step_lap 
            << " ms" << std::endl;
    }
    stage_lap += step_lap;

    if(PRINT(_STEP_TIME)){
         std::cerr << "GPU " << gpuid << "  " << "Reseeding (round 2): " << lap 
            << " ms" << std::endl;
    }
    step_lap += lap;


    TIMER_START(lap);
    reseedLastRound <<< ceil((float)n_seqs / WARPSIZE) == 0 ? 1 
        : ceil((float)n_seqs / WARPSIZE), WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream >>>(
                proc->d_fmIndex, proc->d_opt, proc->d_seqs, proc->d_aux, proc->d_kmerHashTab, n_seqs);
        GET_LAST_ERR();
    TIMER_END(lap);
    LAUNCH_CHK("S reseed r3", *(cudaStream_t*)proc->CUDA_stream);
    if(PRINT(_STEP_TIME)){
         std::cerr << "GPU " << gpuid << "  " << "Reseeding (round 3): " << lap 
            << " ms" << std::endl;
    }
    step_lap += lap;

    if(PRINT(_STEP_TIME)){
         std::cerr << "GPU " << gpuid << "  " << "Reseeding: " << step_lap 
            << " ms" << std::endl;
    }
    stage_lap += step_lap;

    if(PRINT(_INTV) || PRINT(_ALL_SEEDING)){
        for(int readID = 0; readID < n_seqs; readID++)
        {
            printIntv<<<1, WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream>>>(proc->d_aux, readID, _INTV);
        GET_LAST_ERR();
            LAUNCH_CHK_PRINTING(h_phasename[_INTV], *(cudaStream_t*)proc->CUDA_stream);
        }
    }

    if(PRINT(_STAGE_TIME)){
         std::cerr << "GPU " << gpuid << "  " << "Seeding (SMEM seeding + Reseeding): " << stage_lap 
            << " ms" << std::endl;
    }


    // (2/3) Chaining
    stage_lap = 0;

    // B-tree chaining (Chaining 1/1)
    step_lap = 0;

    TIMER_START(lap);
    saLookup <<< n_seqs, 128, 0, *(cudaStream_t*)proc->CUDA_stream >>> (
            proc->d_opt, proc->d_bwt, proc->d_bns, proc->d_seqs, proc->d_aux,
            proc->d_seq_seeds,
            proc->d_buffer_pools);
        GET_LAST_ERR();
    TIMER_END(lap);
    LAUNCH_CHK("C sal", *(cudaStream_t*)proc->CUDA_stream);
    if(PRINT(_STEP_TIME)){
         std::cerr << "GPU " << gpuid << "  " << "B-tree chaining (SA lookup): " << lap 
            << " ms" << std::endl;
    }
    step_lap += lap;

    if(PRINT(_SEED) || PRINT(_ALL_CHAINING)){
        for(int readID = 0; readID < n_seqs; readID++)
        {
            printSeed<<<1, WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream>>>(proc->d_seq_seeds, readID, _SEED);
        GET_LAST_ERR();
            LAUNCH_CHK_PRINTING(h_phasename[_SEED], *(cudaStream_t*)proc->CUDA_stream);
        }
    }


    TIMER_START(lap);
    sortSeedsLowDim 
        <<< n_seqs, SORTSEEDSLOW_BLOCKDIMX, 0, *(cudaStream_t*)proc->CUDA_stream >>> (
                proc->d_seq_seeds,
                proc->d_buffer_pools);
        GET_LAST_ERR();
    // FIXME remove below before measuring time
    LAUNCH_CHK("C sort low", *(cudaStream_t*)proc->CUDA_stream);

    sortSeedsHighDim 
        <<< n_seqs, SORTSEEDSHIGH_BLOCKDIMX, 0, *(cudaStream_t*)proc->CUDA_stream >>> (
                proc->d_seq_seeds,
                proc->d_buffer_pools);
        GET_LAST_ERR();
    TIMER_END(lap);
    LAUNCH_CHK("C sort high", *(cudaStream_t*)proc->CUDA_stream);

    if(PRINT(_STEP_TIME)){
         std::cerr << "GPU " << gpuid << "  " << "B-tree chaining (sort seeds): " << lap 
            << " ms" << std::endl;
    }
    step_lap += lap;

    if(PRINT(_STSEED) || PRINT(_ALL_CHAINING)){
        for(int readID = 0; readID < n_seqs; readID++)
        {
            printSeed<<<1, WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream>>>(proc->d_seq_seeds, readID, _STSEED);
        GET_LAST_ERR();
            LAUNCH_CHK_PRINTING(h_phasename[_STSEED], *(cudaStream_t*)proc->CUDA_stream);
        }
    }

    TIMER_START(lap);
    if(g3_opt->baseline){
#if 0
#define SEEDCHAINING_CHAIN_BLOCKDIMX 256
        SEEDCHAINING_chain_kernel <<< n_seqs, SEEDCHAINING_CHAIN_BLOCKDIMX, 0, *(cudaStream_t*)proc->CUDA_stream >>> (
                proc->d_opt, proc->d_bns, proc->d_seqs, proc->d_seq_seeds,
                proc->d_chains,	// output
                proc->d_buffer_pools);
        GET_LAST_ERR();
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
        GET_LAST_ERR();
    TIMER_END(lap);
    LAUNCH_CHK("C chain", *(cudaStream_t*)proc->CUDA_stream);


    if(PRINT(_STEP_TIME)){
         std::cerr << "GPU " << gpuid << "  " << "B-tree chaining (core): " << lap
            << " ms" << std::endl;
    }
    step_lap += lap;

    if(PRINT(_CHAIN) || PRINT(_ALL_CHAINING)){
        for(int readID = 0; readID < n_seqs; readID++){
            printChain<<<1, WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream>>>(proc->d_chains, readID, _CHAIN);
        GET_LAST_ERR();
            LAUNCH_CHK_PRINTING(h_phasename[_CHAIN], *(cudaStream_t*)proc->CUDA_stream);
        }
    }

    TIMER_START(lap);
    sortChainsDecreasingWeight 
        <<< n_seqs, SORTCHAIN_BLOCKDIMX, 
        MAX_N_CHAIN*2*sizeof(uint16_t)+sizeof(mem_chain_t**), *(cudaStream_t*)proc->CUDA_stream>>> 
            (proc->d_chains, proc->d_buffer_pools);
        GET_LAST_ERR();
    TIMER_END(lap);
    LAUNCH_CHK("C sort chain", *(cudaStream_t*)proc->CUDA_stream);

    if(PRINT(_STEP_TIME)){
         std::cerr << "GPU " << gpuid << "  " << "B-tree chaining (sort): " << lap 
            << " ms" << std::endl;
    }
    step_lap += lap;

    if(PRINT(_STCHAIN) || PRINT(_ALL_CHAINING)){
        for(int readID = 0; readID < n_seqs; readID++)
        {
            printChain<<<1, WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream>>>(proc->d_chains, readID, _STCHAIN);
        GET_LAST_ERR();
            LAUNCH_CHK_PRINTING(h_phasename[_STCHAIN], *(cudaStream_t*)proc->CUDA_stream);
        }
    }

    TIMER_START(lap);
    CHAINFILTERING_filter_kernel <<< n_seqs, CHAIN_FLT_BLOCKSIZE, MAX_N_CHAIN*(3*sizeof(uint16_t)+sizeof(uint8_t)), *(cudaStream_t*)proc->CUDA_stream >>> (
            proc->d_opt, 
            proc->d_chains, 	// input and output
            proc->d_buffer_pools);
        GET_LAST_ERR();
    TIMER_END(lap);
    LAUNCH_CHK("C filter chain", *(cudaStream_t*)proc->CUDA_stream);

    if(PRINT(_STEP_TIME)){
         std::cerr << "GPU " << gpuid << "  " << "B-tree chaining (filter): " << lap 
            << " ms" << std::endl;
    }
    step_lap += lap;

#if 0 // this code is anyway dead for short reads of our concern (<700bp).
    TIMER_START(lap);
    CHAINFILTERING_flt_chained_seeds_kernel <<< ceil((float)n_seqs / WARPSIZE) == 0 ? 1 : ceil((float)n_seqs / WARPSIZE), WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream >>>(
            proc->d_opt, proc->d_bns, proc->d_pac,
            proc->d_seqs, proc->d_chains, 	// input and output
            n_seqs, proc->d_buffer_pools);
        GET_LAST_ERR();
    TIMER_END(lap);
    step_lap += lap;
#endif

    if(PRINT(_FTCHAIN) || PRINT(_ALL_CHAINING)){
        for(int readID = 0; readID < n_seqs; readID++)
        {
            printChain<<<1, WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream>>>(proc->d_chains, readID, _FTCHAIN);
        GET_LAST_ERR();
            LAUNCH_CHK_PRINTING(h_phasename[_FTCHAIN], *(cudaStream_t*)proc->CUDA_stream);
        }
    }

    if(PRINT(_STEP_TIME)){
         std::cerr << "GPU " << gpuid << "  " << "B-tree chaining: " << step_lap 
            << " ms" << std::endl;
    }

    stage_lap = step_lap;
    if(PRINT(_STAGE_TIME)){
         std::cerr << "GPU " << gpuid << "  " << "Chaining (B-tree chaining): " << stage_lap 
            << " ms" << std::endl;
    }

#if 0

    // (3/3) Extending
    stage_lap = 0;

    // Extending -> Local extending (1/2)
    step_lap = 0;


    TIMER_START(lap);
    SWSeed <<< ceil((float)n_seqs / WARPSIZE) == 0 ? 1 : ceil((float)n_seqs / WARPSIZE), WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream >>>(
            proc->d_chains, proc->d_regs, proc->d_seed_records, proc->d_Nseeds, n_seqs,
            proc->d_buffer_pools
            );
        GET_LAST_ERR();
    //FIXME remove below line for measuring time
    LAUNCH_CHK("E pairgen 1", *(cudaStream_t*)proc->CUDA_stream);

    gpuErrchk2( cudaMemcpy(&num_seeds_to_extend, proc->d_Nseeds, sizeof(int), cudaMemcpyDeviceToHost));

    if(num_seeds_to_extend==0){
        proc->n_processed += proc->n_seqs;
        gpuErrchk( cudaStreamSynchronize(*(cudaStream_t*)proc->CUDA_stream) );
        return;
    }

    ExtendingPairGenerate <<< ceil((float)num_seeds_to_extend/32.0), 32, 0, *(cudaStream_t*)proc->CUDA_stream >>> (
            proc->d_opt, proc->d_bns, proc->d_pac, proc->d_seqs,
            proc->d_chains, proc->d_regs, proc->d_seed_records, proc->d_Nseeds,
            n_seqs, proc->d_buffer_pools
            );
        GET_LAST_ERR();
    LAUNCH_CHK("E pairgen 2", *(cudaStream_t*)proc->CUDA_stream);

    TIMER_END(lap);

    if(PRINT(_STEP_TIME)){
         std::cerr << "GPU " << gpuid << "  " << "Local extending (pairgen): " << lap 
            << " ms" << std::endl;
    }
    step_lap += lap;

    if(PRINT(_DETAIL)){
         std::cerr << "GPU " << gpuid << "  " << "# local extending pairs: " << num_seeds_to_extend << std::endl;
    }

    if(PRINT(_EXPAIR) || PRINT(_ALL_EXTENDING)){
        printPair<<<1, WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream>>>(proc->d_seed_records, num_seeds_to_extend, _EXPAIR);
        LAUNCH_CHK_PRINTING(h_phasename[_EXPAIR], *(cudaStream_t*)proc->CUDA_stream);
    }


    TIMER_START(lap);
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
        GET_LAST_ERR();
    //FIXME remove below line for measuring time
    LAUNCH_CHK("E localExtend", *(cudaStream_t*)proc->CUDA_stream);

    TIMER_END(lap);
    if(PRINT(_STEP_TIME)){
         std::cerr << "GPU " << gpuid << "  " << "Local extending (core): " << lap 
            << " ms" << std::endl;
    }
    step_lap += lap;

    if(PRINT(_REGION) || PRINT(_ALL_EXTENDING)){
        for(int readID = 0; readID < n_seqs; readID++)
        {
            printReg<<<1, WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream>>>(proc->d_regs, readID, _REGION);
        GET_LAST_ERR();
            LAUNCH_CHK_PRINTING(h_phasename[_REGION], *(cudaStream_t*)proc->CUDA_stream);
        }
    }


#if 0
    // remove duplicates
    TIMER_START(lap);
    filterRegions <<< n_seqs, 320, 0, *(cudaStream_t*)proc->CUDA_stream >>> (
            proc->d_opt, proc->d_bns,
            proc->d_chains, proc->d_regs, proc->d_buffer_pools
            );
        GET_LAST_ERR();
    TIMER_END(lap);
    LAUNCH_CHK("E filter region", *(cudaStream_t*)proc->CUDA_stream);


    if(PRINT(_STEP_TIME)){
         std::cerr << "GPU " << gpuid << "  " << "Local extending (filter & mark primary regions): " << lap 
            << " ms" << std::endl;
    }
    step_lap += lap;

    if(PRINT(_FTREGION) || PRINT(_ALL_EXTENDING)){
        for(int readID = 0; readID < n_seqs; readID++)
        {
            printReg<<<1, WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream>>>(proc->d_regs, readID, _FTREGION);
        GET_LAST_ERR();
            LAUNCH_CHK_PRINTING(h_phasename[_FTREGION], *(cudaStream_t*)proc->CUDA_stream);
        }
    }


    TIMER_START(lap);
    sortRegions <<< ceil((float)n_seqs / WARPSIZE) == 0 ? 1 : ceil((float)n_seqs / WARPSIZE), WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream >>>(
            proc->d_regs, n_seqs, proc->d_buffer_pools);
        GET_LAST_ERR();
    TIMER_END(lap);
    LAUNCH_CHK("E sort region", *(cudaStream_t*)proc->CUDA_stream);


    if(PRINT(_STEP_TIME)){
         std::cerr << "GPU " << gpuid << "  " << "Local extending (sort): " << lap 
            << " ms" << std::endl;
    }
    step_lap += lap;

    if(PRINT(_STREGION) || PRINT(_ALL_EXTENDING)){
        for(int readID = 0; readID < n_seqs; readID++)
        {
            printReg<<<1, WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream>>>(proc->d_regs, readID, _STREGION);
        GET_LAST_ERR();
            LAUNCH_CHK_PRINTING(h_phasename[_STREGION], *(cudaStream_t*)proc->CUDA_stream);
        }
    }

    if(PRINT(_STEP_TIME)){
         std::cerr << "GPU " << gpuid << "  " << "Local extending: " << step_lap 
            << " ms" << std::endl;
    }
    stage_lap += step_lap;


    // Extending -> Traceback (2/2)
    step_lap = 0;

#endif
    TIMER_START(lap);
    gpuErrchk2( cudaMemset(proc->d_Nseeds, 0, sizeof(int)));

    FINALIZEALN_preprocessing1_kernel <<< ceil((float)n_seqs / WARPSIZE) == 0 ? 1 : ceil((float)n_seqs / WARPSIZE), WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream >>>(
            proc->d_regs, proc->d_alns, proc->d_seed_records, proc->d_Nseeds,
            proc->d_buffer_pools);
        GET_LAST_ERR();
    LAUNCH_CHK("E traceback pairgen 1", *(cudaStream_t*)proc->CUDA_stream);

    gpuErrchk2( cudaMemcpy(&num_seeds_to_extend, proc->d_Nseeds, sizeof(int), cudaMemcpyDeviceToHost));

    if(num_seeds_to_extend==0){
        proc->n_processed += proc->n_seqs;
        gpuErrchk( cudaStreamSynchronize(*(cudaStream_t*)proc->CUDA_stream) );
        return;
    }

    FINALIZEALN_preprocessing2_kernel <<< ceil((float)num_seeds_to_extend / WARPSIZE) == 0 ? 1 : ceil((float)num_seeds_to_extend / WARPSIZE), WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream >>>(
            proc->d_opt, proc->d_seqs,
            proc->d_pac, proc->d_bns,
            proc->d_regs, proc->d_alns, proc->d_seed_records, num_seeds_to_extend,
            proc->d_sortkeys_in,	// sortkeys_in = bandwidth * rlen
            proc->d_seqIDs_in,
            proc->d_buffer_pools);
        GET_LAST_ERR();
    LAUNCH_CHK("E traceback pairgen 2", *(cudaStream_t*)proc->CUDA_stream);


    // reverse query and target if aln position is on reverse strand
    FINALIZEALN_reverseSeq_kernel <<< num_seeds_to_extend, WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream >>> (proc->d_seed_records, proc->d_alns, proc->d_buffer_pools);
        GET_LAST_ERR();
    LAUNCH_CHK("E traceback pairgen 2", *(cudaStream_t*)proc->CUDA_stream);



    d_temp_storage = NULL;
    temp_storage_size = 0;
    // determine temporary storage requirement
    gpuErrchk2( cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_size, proc->d_sortkeys_in, proc->d_sortkeys_out, proc->d_seqIDs_in, proc->d_seqIDs_out, num_seeds_to_extend, 0, 8*sizeof(int), *(cudaStream_t*)proc->CUDA_stream) );
    // Allocate temporary storage
    gpuErrchk2( cudaMalloc(&d_temp_storage, temp_storage_size) );
    // perform radix sort
    gpuErrchk2( cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_size, proc->d_sortkeys_in, proc->d_sortkeys_out, proc->d_seqIDs_in, proc->d_seqIDs_out, num_seeds_to_extend, 0, 8*sizeof(int), *(cudaStream_t*)proc->CUDA_stream) );
    cudaFree(d_temp_storage);

    TIMER_END(lap);

    if(PRINT(_STEP_TIME)){
         std::cerr << "GPU " << gpuid << "  " << "Traceback (pairgen): " << lap 
            << " ms" << std::endl;
    }
    step_lap += lap;

    if(PRINT(_TBPAIR) || PRINT(_ALL_EXTENDING)){
        printPair<<<1, WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream>>>(proc->d_seed_records, num_seeds_to_extend, _TBPAIR);
        GET_LAST_ERR();
        LAUNCH_CHK_PRINTING(h_phasename[_TBPAIR], *(cudaStream_t*)proc->CUDA_stream);
    }

    if(PRINT(_DETAIL)){
         std::cerr << "GPU " << gpuid << "  " << "# traceback pairs: " << num_seeds_to_extend << std::endl;
    }


    TIMER_START(lap);
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
        GET_LAST_ERR();
    TIMER_END(lap);
    LAUNCH_CHK("E traceback", *(cudaStream_t*)proc->CUDA_stream);

    if(PRINT(_STEP_TIME)){
         std::cerr << "GPU " << gpuid << "  " << "Traceback (core): " << lap 
            << " ms" << std::endl;
    }
    step_lap += lap;

    if(PRINT(_ALIGNMENT) || PRINT(_ALL_EXTENDING)){
        for(int readID = 0; readID < n_seqs; readID++)
        {
            printAln<<<1, WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream>>>(proc->d_bns, proc->d_alns, readID, _ALIGNMENT);
        GET_LAST_ERR();
            LAUNCH_CHK_PRINTING(h_phasename[_ALIGNMENT], *(cudaStream_t*)proc->CUDA_stream);
        }
    }


    TIMER_START(lap);
    finalize<<< ceil((float)num_seeds_to_extend / WARPSIZE) == 0 ? 1 : ceil((float)num_seeds_to_extend / WARPSIZE), WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream >>>(
            proc->d_opt, proc->d_bns, proc->d_seqs,
            proc->d_regs, proc->d_alns, proc->d_seed_records, num_seeds_to_extend,
            proc->d_buffer_pools
            );
        GET_LAST_ERR();
    TIMER_END(lap);
    LAUNCH_CHK("E finalize", *(cudaStream_t*)proc->CUDA_stream);

    if(PRINT(_STEP_TIME)){
         std::cerr << "GPU " << gpuid << "  " << "Traceback (finalizing alns): " << lap 
            << " ms" << std::endl;
    }
    step_lap += lap;

    if(PRINT(_STEP_TIME)){
         std::cerr << "GPU " << gpuid << "  " << "Traceback: " << step_lap 
            << " ms" << std::endl;
    }
    stage_lap += step_lap;

    if(PRINT(_RESULT) || PRINT(_ALL_EXTENDING)){
        for(int readID = 0; readID < n_seqs; readID++)
        {
            printAln<<<1, WARPSIZE, 0, *(cudaStream_t*)proc->CUDA_stream>>>(proc->d_bns, proc->d_alns, readID, _RESULT);
        GET_LAST_ERR();
            LAUNCH_CHK_PRINTING(h_phasename[_RESULT], *(cudaStream_t*)proc->CUDA_stream);
        }
    }

    if(PRINT(_STAGE_TIME)){
         std::cerr << "GPU " << gpuid << "  " << "Extending (Local extending + Traceback): " << stage_lap 
            << " ms" << std::endl;
    }


    // pack alns for transferring.
    //TODO

#endif

    TIMER_DESTROY;
    if(PRINT(_BUFFER_USAGE)){
        printBufferInfoHost(proc->d_buffer_pools);
    }
};
