import pickle
import sys
import os
import numpy as np
from tensorflow.keras.datasets import mnist

EPS = [0.008, 0.012, 0.016, 0.02, 0.024]
LB = ['lin', 'ER', 'LSE']
SCORE = sys.argv[1]
NETWORK = sys.argv[2]

objs = np.zeros((100, len(LB), len(EPS)))
logitss = []
probss = []

for filename in os.listdir("results"):
    filenameSplit = filename.split("_")
    if len(filenameSplit) == 6:
        network = 'mnist'
    elif len(filenameSplit) == 7:
        network = filenameSplit[0]
        filenameSplit = filenameSplit[1:]
    elif len(filenameSplit) == 8:
        network = filenameSplit[0] + '_' + filenameSplit[1]
        filenameSplit = filenameSplit[2:]
    ind, eps, lb, ub, score, _ = filenameSplit
    ind = int(ind[3:])
    eps = float(eps[3:])
    lb = lb[2:]
    ub = ub[2:]
    score = score[5:]
    if score == SCORE:
        if network == NETWORK:
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
    if NETWORK == 'mnist':
        suffix = ''
    elif NETWORK == 'mnist-large':
        suffix = 'large-'
    elif NETWORK == 'robust_mnist-large':
        suffix = 'robust-large-'

    # Load true labels for test set
    (X, y), (Xt, yt) = mnist.load_data()
    
    M = 5
    pStarLB = np.empty((100, M, len(EPS)))
    # Iterate over test images
    for i in range(100):
        for e, eps in enumerate(EPS):
            # Iterate over models
            for m in range(M):
                with open(f"./bounds/bounds_net{suffix}{m}_ind{i}_eps{eps}.pickle", 'rb') as fp:
                    bounds = pickle.load(fp)
                lbs = bounds["lbs"]
                ubs = bounds["ubs"]
    
                # Lower bound on probability of true label
                pStarLB[i, m, e] = lbs[-1][yt[i]]
    
    # Average over models, take negative log, and average over instances
    print(-np.log(pStarLB.mean(axis=1)).mean(axis=0))
