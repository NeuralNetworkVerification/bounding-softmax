#!/usr/bin/env bash

# Assume Marabou has been installed:  https://github.com/anwu1219/Marabou branch softmax-bound
SCRIPT=/home/haozewu/Projects/softmax/Marabou/resources/runMarabou.py

NETWORK=./self-attention-small-sim.onnx

for i in {0..5}
#for i in {0..499}
do
    for e in 0.008 0.01 0.012 0.014 0.016
    do
        for t in linear lse1 lse2 er
        do
            sFile="$i"_"$e"_"$t".summary
            $SCRIPT $NETWORK --dataset mnist -i $i -e $e --softmax-bound $t --summary-file="$sFile"
            if grep -q unsat "$sFile"; then
                echo $i $e $t unsat >> results.csv
            else
                echo $i $e $t unknown >> results.csv
            fi
        done
    done
done
