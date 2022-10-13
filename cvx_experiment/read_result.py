import pickle
import sys
import os

for filename in os.listdir("results"):
    ind, eps, lb, ub, score, _ = filename.split("_")
    ind = int(ind[3:])
    eps = float(eps[3:])
    lb = lb[2:]
    ub = ub[2:]
    score = score[5:]
    with open(f'results/{filename}', 'rb') as f:
        obj, logits, probs = pickle.load(f)
        print(ind, eps, eps, lb, ub, obj, logits, probs)
    
