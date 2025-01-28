#!/bin/bash

# File to collect the last lines of stderr
#output_file="out.txt"

# Clear the file at the start
#> "$output_file"

# Run the script multiple times
nums=("1" "2" "3" "4")
HG=${1:-76bp}
SIZE=${2:-10000}
trials=${3:-3}
for num_gpus in "${nums[@]}"; do
    for((i=1; i<=$trials; i++)); do
        echo "Running with $num_gpus GPUs, trial $i"
        make test NUM_GPUS=$num_gpus ERRBUF=$HG.$SIZE.err OUTBUF=$HG.$SIZE.out
    done
done
