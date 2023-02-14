#!/usr/bin/env bash

cd /barrett/scratch/haozewu/softmax/bounding-softmax/transformer_experiment/
source ../py37/bin/activate


network=$1
ind=$2
eps=$3
t=$4

./Marabou/resources/runMarabou.py training/$network --dataset=mnist -e $eps -i $ind --softmax-bound $t --verbosity=0  --dump-bounds
