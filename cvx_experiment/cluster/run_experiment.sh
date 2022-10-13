#!/usr/bin/env bash

cd /barrett/scratch/haozewu/softmax/bounding-softmax/cvx_experiment/
source ../py37/bin/activate

eps=$1
index=$2
lb=$3
ub=$4
score=$5

python UQ_Verif_Deep_Ensemble_cluster.py --eps $eps --index $index --lb $lb --ub $ub --scoring $score 

#--lb LB            ER, or LSE
#--ub UB            LSE
#--scoring SCORING  NLL or Brier

