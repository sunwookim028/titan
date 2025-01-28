#!/bin/bash

# $1 == SIZE

# File to collect the last lines of stderr
output_file="$1.times"

# Clear the file at the start
> "$output_file"

# Run the script multiple times
nums=("1" "2" "3" "4")
size=${1:-40m}
trials=${2:-3}
ERRBUF=$1.err
for num_gpus in "${nums[@]}"; do
    for((i=1; i<=$trials; i++)); do
        echo "Running with $num_gpus GPUs, trial $i"
        echo "Running with $num_gpus GPUs, trial $i" >> "$output_file"
        make smalltest SIZE=$size NUM_GPUS=$num_gpus ERRBUF=$1.err OUTBUF=$1.out
        tail $ERRBUF -n 1 >> "$output_file"
    done
done

echo "Collected last lines of stderr:"
cat "$output_file"
echo "open $ERRBUF for stderr output"
