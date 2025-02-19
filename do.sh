#!/usr/bin/bash

argc="$#"
if [ $argc -lt 2 ]; then
    echo "usage: $0 {ecoli|76bp|100bp|152bp} {chchain|swchain|...}"
    exit 1;
fi
name="$1"
phase="$2"


# work
work.sh run ${name} 1024


# do
lines=$(grep -n "new batch" ${name}.out | cut -d: -f1 | tail -n 2)

const1=1
linebegA=$(($(echo "${lines}" | head -n 1) + const1))
lineendA=$(($(echo "${lines}" | tail -n 1) - const1))
linebegB=$(($(echo "${lines}" | tail -n 1) + const1))
lineendB=$(wc -l ${name}.out | awk '{print $1}')

sed -n "$linebegA,$lineendA p" ${name}.out | grep "^${phase} " | sed "s/^${phase} //g" > ${name}.${phase}
sed -n "$linebegB,$lineendB p" ${name}.out | grep "^${phase} " | sed "s/^${phase} //g" | awk '{$1=$1+57600; print}' >> ${name}.${phase}


# grade
grade.py ${name} ${phase}
