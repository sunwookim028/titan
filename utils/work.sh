#!/bin/bash
# work.sh - organizes testing & debugging workflows.

# script args
argc="$#"
script="$0"
arg1="$1"
arg2="$2"
arg3="$3"
arg4="$4"
arg5="$5"
cmd="$arg1"
id_first_printid="6"

# global usage
usage() {
    echo "usage: $0 {slurm|prep|run|debug|sanitize|profile|help} {ecoli|76bp|100bp|152bp} {100k|1m|...|full} {1|2|3|4} {400000|800000|...} {printing selection}"
    echo ""
    echo "Commands:"
    echo "  slurm       Execute $0/../_slurm.sh through the slurm system."
    echo "  prep        Request GPUs. Options: {a6000|rtx4090} {1|2|3|4}."
    echo "  run         Execute."
    echo "  clean       Remove NAME.sam, .err, .out files."
    echo "  debug       Launch in GDB."
    echo "  sanitize    Check memory errors with compute-sanitizer."
    echo "  profile    Profile the execution with ncu."
    echo "  help        Display this message."
    echo ""
    echo "Options:"
    echo "  ecoli       101bp (submission data)"
    echo "  76bp        hg38 (submission data)"
    echo "  100bp       hg38 (submission data)"
    echo "  152bp       hg38 (submission data)"
    echo ""
    echo "  100k|1m|...|full    Number of queries"
    echo ""
    echo "  1|2|3|4     Number of GPUs to use"
    echo ""
    echo "  400000...   Batch size in numbers"
    echo ""
    echo ""
    echo "Example:"
    echo "  work.sh run 76bp 1m 4 800000"
    echo "  : this BWA-MEM aligns 1m 76bp reads on 4 GPUs with each B=800k."
    echo "    (assumes reads located at ~/reads/76bp.1m)"
    echo ""
}

usage_printflags() {
    echo "Printing selection: select multiple ids from below to print out, separate with whitespaces."
    echo "    -1: silent"
    echo "    0: runtime details, 1: stagewise time, 2: stepwise time"
    echo "    30: buffer usage per batch"
    echo ""
    echo "    3: smem, 4: intv (+reseed)"
    echo "    5: seed (SA looked up), 6: stseed (sorted)"
    echo "    7: chain, 8: stchain, 9: ftchain (filtered)"
    echo "    10: expair, 11: region, 12: ftregion, 13: stregion"
    echo "    14: tbpair, 15: alignment, 16: result"
    echo ""
    echo "Example:"
    echo " work.sh run 76bp 1m 3 2 30 7"
    echo "                       ^ from this $id_first_printid-th arg"
    echo " : prints out chain results, stagewise execution time"
    echo "   and buffer usage after each aligning batch,"
    echo "   WHILE aligning 1m 76bp reads with 3 GPUs."
    echo ""
}


# generate program args
if [ $argc -lt 1 ]; then
        usage
        exit 1
fi
if [ "$cmd" = "run" ] && [ $argc -lt $id_first_printid ]; then
    if [ $argc -gt $(($id_first_printid - 2)) ]; then
        usage_printflags
    else
        usage
        usage_printflags
    fi
    exit 1
else
    print_mask=0
    for ((i=$id_first_printid; i<=$#; i++)); do
        num=${!i}
        if [[ "$num" =~ ^-?[0-9]+$ ]]; then
            print_mask=$((print_mask + $((2 ** num))))
        else
            echo "Invalid input: '$num' is not a valid integer."
            exit 1
        fi
    done
    printopt="-l $print_mask"
fi

# preset args
PROG="./g3"
NAME="$arg2_$arg3_$arg5"
ARGS=""
if [ "$arg1" != "profile" ]; then
    if [ "$arg2" = "ecoli" ]; then
        ARGS="mem\
            -g $arg4\
            -Z $arg5\
            $printopt\
            -o $arg2.sam\
            /datasets/bwa/ref/ecoli/GCA_000005845.2_ASM584v2_genomic.fna\
            /datasets/bwa/ref/ecoli/GCA_000005845.2_ASM584v2_genomic.hash\
            /datasets/bwa/reads/ecoli/ecoli.$arg3"

    else
        ARGS="mem\
            -g $arg4\
            -Z $arg5\
            -b\
             $printopt\
            -o $arg2.sam\
            /datasets/bwa/ref/hg38/hg38.fa)\
            /datasets/bwa/ref/hg38/hg38.hash)\
            /datasets/bwa/reads/hg38/$arg2.$arg3)"
    fi
fi
#"g3 additional options:"
#"		-l: print times"
#"		-b: use baseline"
#"		-g [NUM]: num of GPUs to use"
#"		-v [NUM]: verbosity level"


slurm() {
    sbatch scripts/_slurm.sh
}


prep() {
    if [ $argc -lt 3 ]; then
        echo "usage: $0 $cmd {a6000|rtx4090} {1|2|3|4}"
        exit 1
    fi
    GPU="$arg2"
    NUM="$arg3"
    echo "srun --gres=gpu:$NUM -p $GPU --pty bash"
    srun --gres=gpu:$NUM -p $GPU --pty bash
}


run() {
    if [ $argc -lt 2 ]; then
        usage
        exit 1;
    fi
    if [ $argc -lt 3 ]; then
        echo "usage: $script $arg1 $arg2 {0..}"
        usage_printflags
        exit 1;
    fi
    echo "$PROG $ARGS"
    ${PROG} ${ARGS} 2> >(tee $NAME.err >&2) 1>> $NAME.out
}

ask_to_confirm() {
    while true; do
        read -p "Proceed? (y/n): " yn
        case $yn in
            [Yy]* ) break;;  # If 'y' or 'Y', proceed
            [Nn]* ) exit;;      # If 'n' or 'N', exit
            * ) echo "invalid input";;  # If invalid input, ask again
        esac
    done
}

clean() {
    echo "rm $(ls $NAME.*)"
    ask_to_confirm
    rm $(ls $NAME.*)
}


debug() {
    if [ $argc -lt 3 ]; then
        echo "usage: $0 $1 $2 {0..}"
        usage_printflags
        exit 1;
    fi
    echo "cuda-gdb --args $PROG $ARGS"
    cuda-gdb set detach-on-fork off --args $PROG $ARGS
}


sanitize() {
    compute-sanitizer --quiet --launch-timeout 9000 $PROG $ARGS
}


profile_run() {
    if [ "$arg2" = "ecoli" ]; then
        ARGS="mem\
            -g $arg4\
            -Z $arg5\
            $printopt\
            -o $arg2.sam\
            /datasets/bwa/ref/ecoli/GCA_000005845.2_ASM584v2_genomic.fna\
            /datasets/bwa/ref/ecoli/GCA_000005845.2_ASM584v2_genomic.hash\
            /datasets/bwa/reads/ecoli/ecoli.$arg3"

    else
        ARGS="mem\
            -g $arg4\
            -Z $arg5\
            $printopt\
            -o $arg2.sam\
            /datasets/bwa/ref/hg38/hg38.fa\
            /datasets/bwa/ref/hg38/hg38.hash\
            /datasets/bwa/reads/hg38/$arg2.$arg3"
    fi

    echo "$PROG $ARGS"
    ${PROG} ${ARGS} 2> >(tee $NAME.err >&2) 1>> $NAME.out
}

profile_run_base() {
    if [ "$arg2" = "ecoli" ]; then
        ARGS="mem\
            -g $arg4\
            -Z $arg5\
            -b\
            $printopt\
            -o $arg2.sam\
            /datasets/bwa/ref/ecoli/GCA_000005845.2_ASM584v2_genomic.fna\
            /datasets/bwa/ref/ecoli/GCA_000005845.2_ASM584v2_genomic.hash\
            /datasets/bwa/reads/ecoli/ecoli.$arg3"

    else
        ARGS="mem\
            -g $arg4\
            -Z $arg5\
            -b\
            $printopt\
            -o $arg2.sam\
            /datasets/bwa/ref/hg38/hg38.fa\
            /datasets/bwa/ref/hg38/hg38.hash\
            /datasets/bwa/reads/hg38/$arg2.$arg3"
    fi

    echo "$PROG $ARGS"
    ${PROG} ${ARGS} 2> >(tee $NAME.err >&2) 1>> $NAME.out
}


profile() {
    CUDA_VISIBLE_DEVICES=2 ncu --export $NAME.profile --target-processes all -f --set full bash utils/_ncu.sh
}

case "$cmd" in
    help) 
        usage
        exit 0;;

    slurm) 
        slurm
        exit 0;;

    prep) 
        prep
        exit 0;;

    run) 
        run
        exit 0;;

    clean) 
        clean
        exit 0;;

    debug)
        debug
        exit 0;;

    sanitize)
        sanitize
        exit 0;;

    profile)
        profile
        exit 0;;

    profile_run)
        profile_run
        exit 0;;

    profile_run_base)
        profile_run_base
        exit 0;;

    *) 
        usage
        exit 1;;
esac
