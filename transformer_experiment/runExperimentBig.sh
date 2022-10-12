#!/usr/bin/env bash

# Assume Marabou has been installed:  https://github.com/anwu1219/Marabou branch softmax-bound
SCRIPT=/home/haozewu/Projects/softmax/Marabou/resources/runMarabou.py

NETWORK=./self-attention-big-sim.onnx

for t in linear lse1 lse2 lseh er
do
    for e in 0.008 0.012 0.016
    do
        #for i in {0..2}
        for i in {0..499}
        do
            sFile="$i"_"$e"_"$t"_big.summary
            $SCRIPT $NETWORK --dataset mnist -i $i -e $e --softmax-bound $t --summary-file="$sFile" --verbosity=0
            if [ ! -f $sFile ]; then
                echo $i $e $t misclassify >> results_big.csv
            elif grep -q unsat "$sFile"; then
                echo $i $e $t unsat >> results_big.csv
                rm $sFile
            else
                echo $i $e $t unknown >> results_big.csv
                rm $sFile
            fi
        done
    done
done
