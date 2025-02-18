#!/bin/bash
# work.sh - organizes testing & debugging workflows.

# global usage
usage() {
    echo "usage: $0 {slurm|prep|run|debug|sanitize|profile|help} {ecoli|76bp|100bp|152bp}"
    echo ""
    echo "commands:"
    echo "  slurm       Execute $0/../_slurm.sh through the slurm system."
    echo "  prep        Request GPUs. Options: {a6000|rtx4090} {1|2|3|4}."
    echo "  run         Execute."
    echo "  clean       Remove NAME.sam, .err, .out files."
    echo "  debug       Launch in GDB."
    echo "  sanitize    Check memory errors with compute-sanitizer."
    echo "  profile    Profile the execution with ncu."
    echo "  help        Display this message."
    echo ""
    echo "options:"
    echo "  ecoli       101bp 100k (submission data)"
    echo "  76bp        100k (submission data)"
    echo "  100bp       100k (submission data)"
    echo "  152bp       100k (submission data)"
}


# script args
argc="$#"
script="$0"
arg1="$1"
arg2="$2"
arg3="$3"
cmd="$arg1"
if [ $argc -lt 1 ]; then
        usage
            exit 1
fi
if [ $argc -lt 3 ]; then
    printopt=""
else
    printopt="-l $3"
fi

# preset args
PROG="./titan"
NAME=""
ARGS=""
case "$arg2" in
    ecoli)
        NAME="ecoli";
        ARGS="mem\
            -g 1\
            $printopt\
            -b\
            -o $NAME.sam\
            ../input/ecoli/GCA_000005845.2_ASM584v2_genomic.fna\
            ../input/ecoli/GCA_000005845.2_ASM584v2_genomic.hash\
            ../input/ecoli/ecoli.UUT";;

    76bp)
        NAME="76bp";
        ARGS="mem\
            -g 1\
            $printopt\
            -o $NAME.sam\
            $(realpath ~/ours/input/index/hg38.fa)\
            $(realpath ~/ours/input/index/hg38.hash)\
            $(realpath ~/reads/76bp.100k)";;

    100bp)
        NAME="100bp";
        ARGS="mem\
            -g 1\
            $printopt\
            -o $NAME.sam\
            $(realpath ~/ours/input/index/hg38.fa)\
            $(realpath ~/ours/input/index/hg38.hash)\
            $(realpath ~/reads/100bp.100k)";;

    152bp)
        NAME="152bp";
        ARGS="mem\
            -g 1\
            $printopt\
            -o $NAME.sam\
            $(realpath ~/ours/input/index/hg38.fa)\
            $(realpath ~/ours/input/index/hg38.hash)\
            $(realpath ~/reads/152bp.100k)";;
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
    echo "    0.. = printing flag bitmask"
    echo "            0: silent"
    echo "            +64: additional information (e.g. # extension inputs)"
    echo "            +32: final rpos & cigar results"
    echo "            +1: stagewise running time"
    echo "            +2: stepwise running time"
    echo "            +4: step results from seeding stage"
    echo "                  : Seed, Reseed intervals."
    echo "            +8: step results from chaining stage"
    echo "                  : Sampled seeds, sorted -, chains, sorted -, filtered -."
    echo "            +16: step results from extending stage"
    echo "                  : Ext. pairs, ext. regions, filtered -, sorted -,"
    echo "                    global ext. pairs, g. ext. regions,"
    echo "                    rpos & cigar results."
    exit 1;
fi
    echo "$PROG $ARGS"
    ${PROG} ${ARGS} 2> >(tee $NAME.err >&2) 1>> $NAME.out
}


clean() {
    echo "rm $(ls $NAME.*)"
    rm $(ls $NAME.*)
}


debug() {
if [ $argc -lt 3 ]; then
    echo "usage: $0 $1 $2 {0..63}"
    echo "    0..63 = printing flag bitmask"
    echo "            0: silent"
    echo "            +32: additional information (e.g. # extension inputs)"
    echo "            +1: stagewise running time"
    echo "            +2: stepwise running time"
    echo "            +4: step results from seeding stage"
    echo "            +8: step results from chaining stage"
    echo "            +16: step results from extending stage"
    exit 1;
fi
    echo "cuda-gdb --args $PROG $ARGS"
    cuda-gdb set detach-on-fork off --args $PROG $ARGS
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

    *) 
        usage
        exit 1;;
esac
