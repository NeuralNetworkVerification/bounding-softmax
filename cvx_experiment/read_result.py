import pickle
import sys
import os
import sys
import numpy as np

EPS = float(sys.argv[1])
LB = sys.argv[2]
SCORE = sys.argv[3]

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
    if eps == EPS and lb == LB and score == SCORE:
        with open(f'results/{filename}', 'rb') as f:
            obj, logits, probs = pickle.load(f)
            objs.append(obj)
            logitss.append(logits)
            probss.append(probs)

print(objs)
            
assert(len(objs) == 100)
