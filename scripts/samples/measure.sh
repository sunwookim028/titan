#!/bin/bash


#####
#####   CLEAN THIS FILE BEFORE USAGE.
#####
#####
#####
#####




# File to collect the last lines of stderr
output_file="9pm.txt"

# Clear the file at the start
> "$output_file"

# Run the script multiple times
nums=("1" "2" "3" "4")
size=${1:-4g}
trials=${2:-2}
ERRBUF=stderr.log
for num_gpus in "${nums[@]}"; do
    for((i=1; i<=$trials; i++)); do
        echo "Running with $num_gpus GPUs, trial $i"
        echo "Running with $num_gpus GPUs, trial $i" >> "$output_file"
        make smalltest SIZE=$size NUM_GPUS=$num_gpus 
        tail $ERRBUF -n 1 >> "$output_file"
    done
done

echo "Collected last lines of stderr:"
cat "$output_file"
echo "open $ERRBUF for stderr output"
