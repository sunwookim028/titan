#include "bwa.h"
#include "macro.h"
#include <cub/cub.cuh>

__global__
void project_aln_cnts(int batch_size, mem_aln_v *d_alnvecs,
        int *d_aln_cnts)
{
    int read_id = blockDim.x * blockIdx.x + threadIdx.x;
    if(read_id >= batch_size) return;

    d_aln_cnts[read_id] = d_alnvecs[read_id].n;
}


__global__
void project_cigar_lens(int batch_size, mem_aln_v *d_alnvecs,
        int *d_aln_cnts,
        int *d_aln_offsets,
        int *d_cigar_lens)
{
    int read_id = blockDim.x * blockIdx.x + threadIdx.x;
    if(read_id >= batch_size) return;

    int aln_cnt = d_aln_cnts[read_id];
    int aln_offset = d_aln_offsets[read_id];

    mem_aln_t *alns = d_alnvecs[read_id].a;
    int aln_no;
    for(aln_no = 0; aln_no < aln_cnt; aln_no++){
        d_cigar_lens[aln_offset + aln_no] = alns[aln_no].n_cigar;
    }
}


// Compute offsets for each aln and cigar in the batch using parallel scan.
// (cigar is a part of aln but lengths are variable, 
//  hence we compute separate offsets)
void final_pack_compute_offsets(int batch_size, mem_aln_v *d_alnvecs,
        int batch_aln_cnt,
        int *d_aln_offsets, int *d_cigar_offsets, int *batch_cigar_len)
{
    int *d_aln_cnts;  // indexed with read_id
    int *d_cigar_lens;  // indexed with aln_id
    CUDA_CHECK(cudaMalloc(&d_aln_cnts, sizeof(int) * batch_size)); // FIXME already computed in bwa_align to determine total number of alns.
    CUDA_CHECK(cudaMalloc(&d_cigar_lens, sizeof(int) * batch_aln_cnt));


    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes;

    // compute aln offsets
    // step 1. project embedded cnts into a flat array.
    project_aln_cnts<<<NUM_BLOCKS, BLOCKDIM>>>(batch_size, d_alnvecs,
            d_aln_cnts);
    // step 2. determine scan storage requirements.
    cub::DeviceScan::ExclusiveSum(
            d_temp_storage, temp_storage_bytes,
            d_aln_cnts, d_aln_offsets, batch_size);
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    // step 3. actual scan to compute offsets.
    cub::DeviceScan::ExclusiveSum(
            d_temp_storage, temp_storage_bytes,
            d_aln_cnts, d_aln_offsets, batch_size);
    CUDA_CHECK(cudaFree(d_temp_storage));


    // compute cigar offsets.
    project_cigar_lens<<<NUM_BLOCKS, BLOCKDIM>>>(batch_size, d_alnvecs,
            d_aln_cnts, d_aln_offsets, d_cigar_lens);
    cub::DeviceScan::ExclusiveSum(
            d_temp_storage, temp_storage_bytes,
            d_cigar_lens, d_cigar_offsets, batch_aln_cnt);
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cub::DeviceScan::ExclusiveSum(
            d_temp_storage, temp_storage_bytes,
            d_cigar_lens, d_cigar_offsets, batch_aln_cnt);
    CUDA_CHECK(cudaFree(d_temp_storage));

    int last_cigar_len;
    CUDA_CHECK(cudaMemcpy(batch_cigar_len, d_cigar_offsets + (batch_aln_cnt - 1),
                sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&last_cigar_len, d_cigar_lens + (batch_aln_cnt - 1),
                sizeof(int), cudaMemcpyDeviceToHost));
    *batch_cigar_len += last_cigar_len;

    CUDA_CHECK(cudaFree(d_aln_cnts));
    CUDA_CHECK(cudaFree(d_cigar_lens));
}

// Pack each aln (rid, pos and cigar) into a continuous struct each,
// using the computed offsets. This is to copy to the host.
__global__
void final_pack_compact(int batch_size, mem_aln_v *d_alnvecs,
        int *d_aln_offsets, int *d_cigar_offsets,
        int *d_rids, uint64_t *d_pos, uint32_t *cigars)
{
    int read_id = blockDim.x * blockIdx.x + threadIdx.x;
    if(read_id >= batch_size) return;

    int aln_cnt = d_alnvecs[read_id].n;
    mem_aln_t *alns = d_alnvecs[read_id].a;

    int aln_offset = d_aln_offsets[read_id];
    int cigar_offset = d_cigar_offsets[aln_offset];

    mem_aln_t *aln = &alns[0];
    while(aln_cnt--){
        d_rids[aln_offset] = aln->rid;
        d_pos[aln_offset] = aln->pos;
        for(int k = 0; k < aln->n_cigar; k++){
            cigars[cigar_offset++] = aln->cigar[k];
        }
        aln_offset++;
    }
}
