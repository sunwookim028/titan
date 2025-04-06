len=$1 
k_cnt=$2
rm -f com_$len;
compute-sanitizer --tool memcheck --target-processes all --quiet --launch-timeout 9000 bash $len $k_cnt > com_$len
