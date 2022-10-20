#!/usr/bin/env bash

cd /barrett/scratch/haozewu/softmax/bounding-softmax/cvx_experiment/
source ../py37/bin/activate

network=$1
eps=$2
index=$3
lb=$4
ub=$5
score=$6

python UQ_Verif_Deep_Ensemble_cluster.py --network $network --eps $eps --index $index --lb $lb --ub $ub --scoring $score 

#--lb LB            ER, or LSE
#--ub UB            LSE
#--scoring SCORING  NLL or Brier

