#!/usr/bin/env bash

# Assume Marabou has been installed:  https://github.com/anwu1219/Marabou branch softmax-bound
SCRIPT=/home/haozewu/Projects/softmax/Marabou/resources/runMarabou.py

NETWORK=./self-attention-small-sim.onnx

for t in linear lse1 lse2 er
do
    for e in 0.008 0.012 0.016 0.02
    do
        #for i in {0..2}
        for i in {0..499}
        do
            sFile="$i"_"$e"_"$t".summary
            $SCRIPT $NETWORK --dataset mnist -i $i -e $e --softmax-bound $t --summary-file="$sFile" --verbosity=0 --num-workers=8
            if [ ! -f $sFile ]; then
                echo $i $e $t misclassify >> results.csv
            elif grep -q unsat "$sFile"; then
                echo $i $e $t unsat >> results.csv
                rm $sFile
            else
                echo $i $e $t unknown >> results.csv
                rm $sFile
            fi
        done
    done
done
