#ifndef _MACRO_H
#define _MACRO_H

#define MAX_NUM_GPUS 8
#define MAX_NUM_STEPS 24

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

#define COMPUTE_TOTAL 15
#define PULL_TOTAL 16
#define PUSH_TOTAL 17

#define GPU_SETUP 18
#define FILE_INPUT 19
#define FILE_OUTPUT 20
#define ALIGNER_TOP 21
#define FILE_INPUT_FIRST 22

// Decoding CIGAR from bam encoding.
#define BAM2LEN(bam)    ((int)(bam>>4))
#define BAM2OP(bam)     ((char)("MIDSH"[(int)bam&0xf])) 

// constants
#define WARPSIZE 32

#define SB_MAX_COUNT 1000000                   // max number of reads

#define NBUFFERPOOLS 32 	// number of buffer pools


// Constant values
#define MAX_LEN_READ 320 // max. length of input short read
#define SEQ_MAXLEN MAX_LEN_READ// max length of a seq we want to process
#define AVG_NUM_SEEDS 8



#define MAX_NUM_SW_SEEDS 500 // max. number of seed counts in all chains of a read, including dups.

// super-batch config
#define SB_NAME_LIMIT (unsigned long)SB_MAX_COUNT * 100       // chunk size of name
#define SB_COMMENT_LIMIT (unsigned long)SB_MAX_COUNT * 100    // chunk size of comment
#define SB_SEQ_LIMIT (unsigned long)SB_MAX_COUNT *SEQ_MAXLEN  // chunk size of seq
#define SB_QUAL_LIMIT (unsigned long)SB_MAX_COUNT *SEQ_MAXLEN // chunk size of qual


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

#endif 
