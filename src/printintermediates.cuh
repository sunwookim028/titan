#ifndef PRINTINTERMEDIATES_CUH
#define PRINTINTERMEDIATES_CUH
#include "bwa.h"

#define BIT(n) (1LL << (n))

// Seeding stage.
#define SMINTV      0   /* bwtintv_t,   smem_aux_t */
#define CHINTV      1   /* bwtintv_t,   smem_aux_t */
#define CHSEED_     2   /* mem_seed_t,  mem_seed_v */
#define CHSEED      3   /* mem_seed_t,  mem_seed_v */
// Chaining stage.
#define CHCHAIN     4   /* mem_chain_t, mem_chain_v */
#define SWCHAIN     5   /* mem_chain_t, mem_chain_v */
#define SWPAIR      6   /* seed_record_t, batch-wide 1D array */
// Extending stage.
#define SWREG_      7   /* mem_alnreg_t, mem_alnreg_v */
#define SWREG       8   /* mem_alnreg_t, mem_alnreg_v */
#define ANREG       9   /* mem_alnreg_t, mem_alnreg_v */
#define ANPAIR      10  /* seed_record_t, batch-wide 1D array */
#define ANALN_      11  /* mem_aln_t, mem_aln_v */
#define ANALN       12  /* mem_aln_t, mem_aln_v */

#define FLATINTV      13   /* bwtintv_t,   smem_aux_t */

#define STEP_TIME 50
#define STAGE_TIME 51
#define ADDITIONAL 60

// format: [ID readID] qbeg qend num_hits sa_k
__global__ void printIntv(smem_aux_t *d_intvvecs, int readID, int type);

// format: [ID readID] rbeg len qbeg
__global__ void printSeed(mem_seed_v *d_seedvecs, int readID, int type);

// format: [ID readID] rpos weight num_seeds
__global__ void printChain(mem_chain_v *d_chainvecs, int readID, int type);

// format: [ID readID] q_left r_left q_right r_right
__global__ void printPair(seed_record_t *d_pairs, int num_records, int type);

// format: [ID readID] rb re qb qe score w seedcov frac_rep seedlen0
__global__ void printReg(mem_alnreg_v *d_regvecs, int readID, int type);

// format: [ID readID] rid rpos cigarstring
__global__ void printAln(mem_aln_v *d_alnvecs, int readID, int type);
#endif
