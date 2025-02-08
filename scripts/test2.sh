#!/bin/bash

# WE TEST WITH ECOLI SINCE IT IS FASTER.
# AFTER ACHIVING ACCURACY, MOVE ON TO HG38.

READS_SIZE=100000
GPUS_COUNT=1
PRINT_TIMES="" # comment out to not print times

SAMFILE="test.sam"
OUTFILE="test.out" # redirect all stdout output to here
ERRFILE="test.err" # redirect all stderr output to here

# DO NOT CHANGE FROM HERE.
ECOLI_IDX_PREFIX="../input/ecoli/GCA_000005845.2_ASM584v2_genomic"
HG38_IDX_PREFIX="~/ours/input/index/hg38"

IDX="$IDX_PREFIX.fna"
HASH="$IDX_PREFIX.hash"
READS="../input/ecoli/ecoli.$READS_SIZE"

# test command
echo "./titan mem $IDX $HASH $READS -o $SAMFILE 2> >(tee $ERRFILE >&2) 1>> $OUTFILE -g $GPUS_COUNT $PRINT_TIMES"
#./titan mem $IDX $HASH $READS -o $SAMFILE 2> >(tee $ERRFILE >&2) 1>> $OUTFILE -g $GPUS_COUNT $PRINT_TIMES

IDX="$HG38_IDX_PREFIX.fa"
HASH="$HG38_IDX_PREFIX.hash"
READS="~/reads/100bp.10000"
./titan mem $IDX $HASH $READS -o $SAMFILE 2> >(tee $ERRFILE >&2) 1>> $OUTFILE -g $GPUS_COUNT $PRINT_TIMES
