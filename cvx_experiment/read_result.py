import pickle
import sys
import os
import sys
import numpy as np

INDEX = int(sys.argv[1])
EPS = float(sys.argv[2])
LB = sys.argv[3]
SCORE = sys.argv[4]

objs = []
logitss = []
probss = []

for filename in os.listdir("results"):
    ind, eps, lb, ub, score, _ = filename.split("_")
    ind = int(ind[3:])
    eps = float(eps[3:])
    lb = lb[2:]
    ub = ub[2:]
    score = score[5:]
    if ind == INDEX and eps == EPS and lb == LB and score == SCORE:
        with open(f'results/{filename}', 'rb') as f:
            obj, logits, probs = pickle.load(f)
            objs.append(obj)
            logitss.append(logits)
            probss.append(probs)

assert(len(objs) == 100)
