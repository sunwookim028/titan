#ifndef BATCH_CONFIG_H
#define BATCH_CONFIG_H

#define NBUFFERPOOLS 32 	// number of buffer pools
#define POOLSIZE 80000000	// size of each buffer pool

#include "macro.h"

#ifdef BASELINE
#define USE_DEFAULT_PREPROCESSING
#define USE_DEFAULT_SEEDING
#define USE_DEFAULT_CHAINING
#define USE_DEFAULT_EXTEND
#define USE_DEFAULT_TRACEBACK
#endif


// Constant values
#define MAX_LEN_READ 320 // max. length of input short read
#define MB_MAX_COUNT 64000 // larger batch (e.g. 131072 doesn't seem clearly beneficial)       // max number of reads
#define AVG_NUM_SEEDS 8


#define SEQ_MAXLEN MAX_LEN_READ// max length of a seq we want to process
// mini-batch config // DO NOT CHANGE UNTIL WE GET EVALUATION DATA
#define MB_NAME_LIMIT MB_MAX_COUNT * 1       // chunk size of name
#define MB_COMMENT_LIMIT MB_MAX_COUNT * 1    // chunk size of comment
#define MB_SEQ_LIMIT MB_MAX_COUNT *SEQ_MAXLEN  // chunk size of seq
#define MB_QUAL_LIMIT MB_MAX_COUNT *SEQ_MAXLEN // chunk size of qual
#define MB_SAM_LIMIT MB_MAX_COUNT * 1       // chunk size of sam output

#define MAX_NUM_SW_SEEDS 500 // max. number of seed counts in all chains of a read, including dups.

// super-batch config
#define SB_MAX_COUNT 10000000                   // max number of reads
#define SB_NAME_LIMIT (unsigned long)SB_MAX_COUNT * 100       // chunk size of name
#define SB_COMMENT_LIMIT (unsigned long)SB_MAX_COUNT * 100    // chunk size of comment
#define SB_SEQ_LIMIT (unsigned long)SB_MAX_COUNT *SEQ_MAXLEN  // chunk size of seq
#define SB_QUAL_LIMIT (unsigned long)SB_MAX_COUNT *SEQ_MAXLEN // chunk size of qual

#endif
