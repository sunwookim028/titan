#ifndef SEED_CONSTANTS_
#define SEED_CONSTANTS_

#include "batch_config.h"

// Runtime profiling macros
#define S_SMEM 0
#define S_R2    1
#define S_R3    2
#define C_SAL   3
#define C_SORT_SEEDS    4
#define C_CHAIN     5
#define C_SORT_CHAINS   6
#define C_FILTER    7
#define E_PAIRGEN   8
#define E_EXTEND    9
#define E_FILTER_MARK   10
#define E_SORT_ALNS 11
#define E_T_PAIRGEN 12
#define E_TRACEBACK 13
#define E_FINALIZE  14

#define ALIGN_TOTAL 15



//
// Compile options
//

//
// Index building & Seeding options
//#define FORWARD // preseeding option // THIS MUST NOT BE DEFINED
#define STRIDED // THIS MUST BE DEFINED

#define SA_COMPRESSION
//#define INDEX_ONLY
#define STUDYING_HG38
//#define FORWARD_STRAND_ONLY

//
#define BACKWARD_EXT(seed, base)\
        backwardExt(sentinelIndex, &seed, base, &seed, oneHot, cpOcc, count);

// that is, in the "base1 <- base0 <- seed" direction.
#define BACKWARD_EXT2(seed, base0, base1)\
        backwardExt2(sentinelIndex, &seed, base0, base1, &seed, oneHot, cpOcc, count, cpOcc2, count2, &firstBase);

#define BACKWARD_EXT_B(seed, base)\
        backwardExtBackward(sentinelIndex, &seed, base, &seed, oneHot, cpOcc, count);

// that is, in the "base1 <- base0 <- seed" direction.
#define BACKWARD_EXT2_B(seed, base0, base1)\
        backwardExt2Backward(sentinelIndex, &seed, base0, base1, &seed, oneHot, cpOcc, count, cpOcc2, count2, &firstBase);

#define FORWARD_EXT(seed, base);\
            {uint64_t temp = seed.x[0];\
            seed.x[0] = seed.x[1];\
            seed.x[1] = temp;\
            BACKWARD_EXT(seed, 3 - (base));\
            temp = seed.x[0];\
            seed.x[0] = seed.x[1];\
            seed.x[1] = temp;}

#define FORWARD_EXT2(seed, base0, base1);\
            {uint64_t temp = seed.x[0];\
            seed.x[0] = seed.x[1];\
            seed.x[1] = temp;\
            BACKWARD_EXT2(seed, 3 - (base0), 3 - (base1));\
            temp = seed.x[0];\
            seed.x[0] = seed.x[1];\
            seed.x[1] = temp;}
    
#define LEN(seed) ((uint32_t)((seed).info) - (uint32_t)(((seed).info)>>32))
#define N(seed) ((uint32_t)((seed).info) - 1)
#define M(seed) ((uint32_t)((seed).info >> 32))
#define INFO(m, n) ((((uint64_t)(m)) << 32) | (uint64_t)(n + 1))
#define PLEN(seed) ((uint32_t)(seed->info) - (uint32_t)((seed->info)>>32))
#define PN(seed) ((uint32_t)(seed->info) - 1)
#define PM(seed) ((uint32_t)(seed->info >> 32))
#endif // end of #ifndef SEED_CONSTANTS_
