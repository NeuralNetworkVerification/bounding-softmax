#!/usr/bin/env bash

cd /barrett/scratch/haozewu/softmax/bounding-softmax/transformer_experiment/
source ../py37/bin/activate


network=$1
ind=$2
eps=$3
t=$4

if [[ "$t" == lin0 ]]; then
    ./Marabou_iclr/resources/runMarabou.py $network --dataset=mnist -e $eps -i $ind --verbosity=0 
else
    ./Marabou/resources/runMarabou.py $network --dataset=mnist -e $eps -i $ind --softmax-bound $t --verbosity=0  --dump-bounds
fi
