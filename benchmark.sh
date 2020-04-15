#!/bin/sh

for file in ./benchmarks/*; do
  python3 Sim.py ${file} $1 $2 $3 ${file/benchmarks/output}_${4}.txt &
done
exit
