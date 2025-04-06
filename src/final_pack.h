#ifndef FINAL_PACK_H
#define FINAL_PACK_H

#include "bwa.h"
#include "macro.h"

// Compute offsets for each aln and cigar in the batch using parallel scan.
// (cigar is a part of aln but lengths are variable, 
//  hence we compute separate offsets)
void final_pack_compute_offsets(int batch_size, mem_aln_v *d_alnvecs,
        int batch_aln_cnt,
        int *d_aln_offsets, int *d_cigar_offsets, int *batch_cigar_len);

// Pack each aln (rid, pos and cigar) into a continuous struct each,
// using the computed offsets. This is to copy to the host.
__global__
void final_pack_compact(int batch_size, mem_aln_v *d_alnvecs,
        int *d_aln_offsets, int *d_cigar_offsets,
        int *d_rids, uint64_t *d_pos, uint32_t *cigars);

__global__
void final_pack(int batch_size, int* d_alns_offset, int* d_rid, uint64_t* d_pos);

#endif
