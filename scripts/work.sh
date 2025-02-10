#!/bin/bash
# work.sh - organizes testing & debugging workflows.

# global usage
usage() {
    echo "usage: $0 {slurm|prep|run|debug|sanitize|profile|help} {ecoli|76bp|100bp|152bp}"
    echo ""
    echo "commands:"
    echo "  slurm       Execute $0/../_slurm.sh through the slurm system."
    echo "  prep        Request GPUs."
    echo "  run         Execute."
    echo "  debug       Launch in GDB."
    echo "  sanitize    Check memory errors with compute-sanitizer."
    echo "  sanitize    Profile the execution with ncu."
    echo "  help        Display this message."
    echo ""
    echo "options:"
    echo "  ecoli       101bp 100k (submission data)"
    echo "  76bp        100k (submission data)"
    echo "  100bp       100k (submission data)"
    echo "  152bp       100k (submission data)"
}


# script args
if [ $# -lt 1 ]; then
        usage
            exit 1
fi
cmd="$1"
opt="$2"

# preset args
PROG="./titan"
NAME=""
ARGS=""
case "$opt" in
    ecoli)
        NAME="ecoli";
        ARGS="mem\
            -g 1\
            -o $NAME.sam\
            ../input/ecoli/GCA_000005845.2_ASM584v2_genomic.fna\
            ../input/ecoli/GCA_000005845.2_ASM584v2_genomic.hash\
            ../input/ecoli/ecoli.100k";;

    76bp)
        NAME="76bp";
        ARGS="mem\
            -g 1\
            -o $NAME.sam\
            ~/ours/input/index/hg38.fa\
            ~/ours/input/index/hg38.hash\
            ~/reads/76bp.100k";;

    100bp)
        NAME="100bp";
        ARGS="mem\
            -g 1\
            -o $NAME.sam\
            ~/ours/input/index/hg38.fa\
            ~/ours/input/index/hg38.hash\
            ~/reads/100bp.100k";;

    152bp)
        NAME="152bp";
        ARGS="mem\
            -g 1\
            -o $NAME.sam\
            ~/ours/input/index/hg38.fa\
            ~/ours/input/index/hg38.hash\
            ~/reads/152bp.100k";;

    *) 
        usage
        exit 1;;
esac
#"g3 additional options:"
#"		-l: print times"
#"		-b: use baseline"
#"		-g [NUM]: num of GPUs to use"
#"		-v [NUM]: verbosity level"


slurm() {
    sbatch scripts/_slurm.sh
}


prep() {
    NUM="1" # FIXME for CLI control
    GPU="rtx4090"
    srun --gres=gpu:$NUM -p $GPU --pty bash
}


run() {
    echo "$PROG $ARGS"
    ${PROG} ${ARGS} 2> >(tee $NAME.err >&2) 1>> $NAME.out
}


debug() {
    echo "cuda-gdb --args $PROG $ARGS"
    cuda-gdb --args $PROG $ARGS
}


sanitize() {
    compute-sanitizer --quiet --launch-timeout 9000 $PROG $ARGS
}

profile() {
    # FIXME wrong command (fix script name)
    CUDA_VISIBLE_DEVICES=2 ncu --export $NAME.profile --target-processes all -f --set full bash bwa-mem-gpu.300bp.2k.sh
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

    debug)
        debug
        exit 0;;

    sanitize)
        sanitize
        exit 0;;

    profile)
        profile
        exit 0;;

    *) 
        usage
        exit 1;;
esac
