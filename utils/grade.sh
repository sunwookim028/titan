awk 'NR==FNR { ref[NR] = /usr/bin/bash; next }
     { total++; if (/usr/bin/bash == ref[FNR]) hit++ }
     END { printf(Hit ratio: %.2f%%n, hit/total*100) }' ../bwa-mem2/gold.proc3 test.proc5
