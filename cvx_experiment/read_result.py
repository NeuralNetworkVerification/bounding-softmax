import pickle
import sys
import os
import numpy as np
from tensorflow.keras.datasets import mnist

EPS = [0.008, 0.012, 0.016]
LB = ['lin', 'ER', 'LSE']
NETWORK = sys.argv[1]
SCORE = sys.argv[2]

objs = np.zeros((100, len(LB), len(EPS)))
logitss = []
probss = []

for filename in os.listdir("results"):
    if filename[:len(NETWORK)] != NETWORK:
        continue
    ind, eps, lb, ub, score, _ = filename.split("_")
    ind = int(ind[3:])
    eps = float(eps[3:])
    lb = lb[2:]
    ub = ub[2:]
    score = score[5:]
    if score == SCORE:
        with open(f'results/{filename}', 'rb') as f:
            obj, logits, probs = pickle.load(f)
            e = EPS.index(eps)
            b = LB.index(lb)
            objs[ind, b, e] = obj
            logitss.append(logits)
            probss.append(probs)

if SCORE == 'NLL':
    # Negative log of (lower bound on) probability
    objs = -np.log(-objs)

# Average over instances
print(objs.mean(axis=0))
            
#assert(len(objs) == 100)

# FOR NLL, READ LOWER BOUNDS ON PROBABILITY FROM BOUND FILES INSTEAD
if SCORE == 'NLL':
    # Load true labels for test set
    (X, y), (Xt, yt) = mnist.load_data()
    
    M = 5
    pStarLB = np.empty((100, M, len(EPS)))
    # Iterate over test images
    for i in range(100):
        for e, eps in enumerate(EPS):
            # Iterate over models
            for m in range(M):
                with open(f"./bounds/bounds_net{m}_ind{i}_eps{eps}.pickle", 'rb') as fp:
                    bounds = pickle.load(fp)
                lbs = bounds["lbs"]
                ubs = bounds["ubs"]
    
                # Lower bound on probability of true label
                pStarLB[i, m, e] = lbs[9][yt[i]]
    
    # Average over models, take negative log, and average over instances
    print(-np.log(pStarLB.mean(axis=1)).mean(axis=0))
