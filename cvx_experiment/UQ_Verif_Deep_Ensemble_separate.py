#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robustness verification of uncertainty estimates
"""

import pickle
import numpy as np
from scipy.special import expit, logsumexp, softmax
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar10
import cvxpy as cvx
import argparse

parser = argparse.ArgumentParser(description='Example.')
parser.add_argument('--eps', type=float, default=0.008, help="0.008 0.012 0.016")
parser.add_argument('--index', type=int, default=0, help="< 100")
parser.add_argument('--lb', type=str, default="ER", help="lin or ER or LSE")
parser.add_argument('--ub', type=str, default="LSE", help="lin or LSE")
parser.add_argument('--scoring', type=str, default="NLL", help="NLL or Brier")
parser.add_argument('--network', type=str, default="mnist", help="mnist or mnist-large or robust_mnist-large or cifar10-large or robust_cifar10-large")
args = parser.parse_args()


if __name__ == '__main__':
    # PARAMETERS TO BE VARIED
    # Perturbation radius
    eps = args.eps #0.008
    # Softmax bound types
    LBtype = args.lb #'ER'  # 'lin' or 'ER' or 'LSE' currently
    UBtype = args.ub #'LSE'  # 'lin' or 'LSE' currently, 'ER' DOESN'T WORK
    # Scoring function
    scoring = args.scoring #'NLL'     # 'NLL' or 'Brier'
    # Network
    network = args.network
    if network == 'mnist':
        suffix = ''
    elif network == 'mnist-large':
        suffix = 'large-'
    elif network == 'robust_mnist-large':
        suffix = 'robust-large-'
    elif network == 'robust_cifar10':
        suffix = 'robust_cifar10-'
    elif network == 'robust_cifar10-large':
        suffix = 'robust_cifar10-large-'

    # ENSEMBLE DIMENSIONS
    # Number of models in ensemble
    M = 5
    # Number of hidden layers
    if network == 'mnist':
        L = 2
    elif network in ['mnist-large', 'robust_mnist-large', 'robust_cifar10', 'robust_cifar10-large']:
        L = 3
    # Number of classes
    d = 10
    
    # LOAD MNIST TEST IMAGES AND BOUNDS
    if "mnist" in network:
        (X, y), (Xt, yt) = mnist.load_data()
    elif "cifar10" in network:
        (X, y), (Xt, yt) = cifar10.load_data()
    # Index of test image to use
    i = args.index
    
    # Results arrays
    obj = np.zeros(M)
    logits = np.zeros((M, d))
    probs = np.zeros((M, d))
    
    # LOAD NEURON BOUNDS
    
    lbs = []
    ubs = []
    # Bounds on logit differences
    diffs_l = np.empty((M, d, d-1))
    diffs_u = np.empty((M, d, d-1))
    # Bounds on probabilities
    sm_l = np.empty((M, d))
    sm_u = np.empty((M, d))
    if UBtype == 'LSE':
        lsm_l = np.empty((M, d))
        lsm_u = np.empty((M, d))
    
    # Iterate over models
    for m in range(M):
        with open(f"./bounds/bounds_net{suffix}{m}_ind{i}_eps{eps}.pickle", 'rb') as fp:
            bounds = pickle.load(fp)
        lbs.append(bounds["lbs"])
        ubs.append(bounds["ubs"])

        # Iterate over classes
        for j in range(d):
            # Bounds on logit differences z_{j'} - z_j
            # NEEDS TO BE MODEL-SPECIFIC FOR DEEP ENSEMBLE 
            diffs_l[m, j] = np.array(lbs[m][2*(L+1)])[(d-1)*j:(d-1)*(j+1)]
            diffs_u[m, j] = np.array(ubs[m][2*(L+1)])[(d-1)*j:(d-1)*(j+1)]
            # Constant bounds on softmax output
            sm_l[m, j] = expit(-logsumexp(diffs_u[m, j]))
            sm_u[m, j] = expit(-logsumexp(diffs_l[m, j]))
            if UBtype == 'LSE':
                # Also need log of constant bounds
                lsm_l[m, j], lsm_u[m, j] = np.log(sm_l[m, j]), np.log(sm_u[m, j])
    
    # INPUT LAYER
    mid0 = Xt[i].flatten() / 255
    # Input variables
    D = len(mid0)
    x0 = cvx.Variable(D)
    # Input bounds
    l0 = np.maximum(mid0 - eps, 0)
    u0 = np.minimum(mid0 + eps, 1)

    # ITERATE OVER MODELS
    
    for m in range(M):

        # Initialize list of constraints
        cons = [x0 >= l0, x0 <= u0]

        # LOAD NN WEIGHTS
        W = np.load(f'networks/{network}-{m}.npz')
        # Put weights into list of arrays
        w = [None] * (L + 1)
        b = [None] * (L + 1)
        for ll in range(L + 1):
            w[ll] = W['w' + str(ll + 1)]
            b[ll] = W['b' + str(ll + 1)]

        # HIDDEN LAYERS
    
        # Initialize lists of quantities over layers
        z = [None] * (L + 1)
        xa = [None] * L
        xu = [None] * L
        l = [None] * (L + 1)
        u = [None] * (L + 1)
        inactive = [None] * L
        active = [None] * L
        unstable = [None] * L
        
        # Iterate over hidden layers
        for ll in range(L):
            # Bounds on pre-activation neurons
            # NEEDS TO BE MADE MODEL-SPECIFIC FOR DEEP ENSEMBLE
            l[ll] = np.array(lbs[m][2*ll+1])
            u[ll] = np.array(ubs[m][2*ll+1])
    
            # Drop inactive neurons
            inactive[ll] = u[ll] <= 0
            w[ll] = w[ll][:, ~inactive[ll]]
            b[ll] = b[ll][~inactive[ll]]
            l[ll] = l[ll][~inactive[ll]]
            u[ll] = u[ll][~inactive[ll]]
            w[ll+1] = w[ll+1][~inactive[ll], :]
    
            # Find unstable and active neurons
            unstable[ll] = l[ll] < 0
            active[ll] = ~unstable[ll]
    
            # Affine transformation
            if ll == 0:
                # Input layer
                z[ll] = x0 @ w[ll] + b[ll]
            else:
                nUnstable = unstable[ll-1].sum()
                if nUnstable == unstable[ll-1].shape[0]:
                    z[ll] = xu[ll-1] @ w[ll][unstable[ll-1], :] + b[ll]
                else:
                    z[ll] = xa[ll-1] @ w[ll][active[ll-1], :] + xu[ll-1] @ w[ll][unstable[ll-1], :] + b[ll]
            
            # Define active neurons
            xa[ll] = z[ll][active[ll]]
    
            # Define and constrain unstable neurons
            nUnstable = unstable[ll].sum()
            if nUnstable:
                xu[ll] = cvx.Variable(nUnstable, nonneg=True)
                cons.append(xu[ll] >= z[ll][unstable[ll]])
                g = u[ll][unstable[ll]] / (u[ll][unstable[ll]] - l[ll][unstable[ll]])
                cons.append(xu[ll] <= cvx.multiply(g, z[ll][unstable[ll]] - l[ll][unstable[ll]]))
            else:
                # No unstable neurons
                xu[ll] = np.zeros(nUnstable)
    
        # LOGITS
    
        nUnstable = unstable[L-1].sum()
        if nUnstable == unstable[L-1].shape[0]:
            z[L] = xu[L-1] @ w[L][unstable[L-1], :] + b[L]
        else:
            z[L] = xa[L-1] @ w[L][active[L-1], :] + xu[L-1] @ w[L][unstable[L-1], :] + b[L]
        # Bounds on logits
        # NEEDS TO BE MODEL-SPECIFIC FOR DEEP ENSEMBLE
        l[L] = np.array(lbs[m][2*L+1])
        u[L] = np.array(ubs[m][2*L+1])
        # Index of largest logit in terms of midpoint
        jmax = (l[L] + u[L]).argmax()

        # SOFTMAX
        
        # Output probabilities
        p = cvx.Variable(d, nonneg=True)
    
        # Constrain softmax outputs
        # Iterate over classes
        for j in range(d):

            if LBtype == 'lin' or UBtype == 'lin':
                # Pre-compute some quantities for linear bounds
                # Tangent points for bounding exponentials from below
                diffs_t = np.minimum(np.log((np.exp(diffs_u[m, j]) - np.exp(diffs_l[m, j])) / (diffs_u[m, j] - diffs_l[m, j])), diffs_l[m, j] + 1)
                # Lower and upper bounds on denominator of softmax
                den_l = 1 + np.dot(np.exp(diffs_t), diffs_l[m, j] - diffs_t + 1)
                den_u = 1 + np.exp(diffs_u[m, j]).sum()
                # Tangent point for bounding reciprocal from below
                den_t = max(np.sqrt(den_l * den_u), den_u / 2)
            
            # Add constraint on jth softmax output
            if j == yt[i]:
                # Correct class, need lower bound on probability
                if LBtype == 'lin':
                    # Linear ER bound
                    a_lin_l = np.zeros(d)
                    others = np.arange(d) != j
                    a_lin_l[others] = -(np.exp(diffs_u[m, j]) - np.exp(diffs_l[m, j])) / (diffs_u[m, j] - diffs_l[m, j]) / den_t**2
                    a_lin_l[j] = -a_lin_l[others].sum()
                    b_lin_l = ((diffs_u[m, j] * np.exp(diffs_l[m, j]) - diffs_l[m, j] * np.exp(diffs_u[m, j])) / (diffs_u[m, j] - diffs_l[m, j])).sum()
                    b_lin_l = 1 / den_t * (2 - 1 / den_t * (1 + b_lin_l))
                    bnd = p[j] >= a_lin_l @ z[L] + b_lin_l
                elif LBtype == 'ER' or j == jmax:
                    # LSE2 bound same as ER bound when j = jmax
                    # Differences with z_j
                    others = np.arange(d) != j
                    diffs = z[L][others] - z[L][j]
                    # Bounding log probability avoids having to use a second-order cone
                    bnd = cvx.log(p[j]) >= -cvx.log(1 + cvx.sum((cvx.multiply(np.exp(diffs_l[m, j]), diffs_u[m, j] - diffs) + cvx.multiply(np.exp(diffs_u[m, j]), diffs - diffs_l[m, j])) / (diffs_u[m, j] - diffs_l[m, j])))
                elif LBtype == 'LSE':
                    # LSE2 bound for j != jmax
                    # Differences with z_{jmax}
                    others = np.arange(d) != jmax
                    diffs = z[L] - z[L][jmax]
                    bnd = cvx.log(p[j]) >= diffs[j] - cvx.log(1 + cvx.sum((cvx.multiply(np.exp(diffs_l[m, jmax]), diffs_u[m, jmax] - diffs[others]) + cvx.multiply(np.exp(diffs_u[m, jmax]), diffs[others] - diffs_l[m, jmax])) / (diffs_u[m, jmax] - diffs_l[m, jmax])))
                    # STILL NEED TO IMPLEMENT LSE1 BOUND
                # Add constant bound
                bnd2 = p[j] >= sm_l[m, j]
            else:
                # Incorrect class, need upper bound on probability
                if UBtype == 'lin':
                    # Linear ER bound
                    a_lin_u = np.zeros(d)
                    others = np.arange(d) != j
                    a_lin_u[others] = -np.exp(diffs_t) / (den_l * den_u)
                    a_lin_u[j] = -a_lin_u[others].sum()
                    b_lin_u = 1 / den_l + 1 / den_u - (1 + np.dot(np.exp(diffs_t), 1 - diffs_t)) / (den_l * den_u)
                    bnd = p[j] <= a_lin_u @ z[L] + b_lin_u
                elif UBtype == 'ER':
                    # CAN'T USE: HAVING MORE THAN 2 OF THESE BOUNDS PREVENTS CONVERGENCE 
                    #bnd = p[j] <= sm_l[j] + sm_u[j] - sm_l[j] * sm_u[j] * (1 + cvx.sum(cvx.exp(diffs[j])))
                    bnd = p[j] <= sm_l[m, j] + sm_u[m, j] - sm_l[m, j] * sm_u[m, j] * cvx.exp(cvx.log_sum_exp(z[L]) - z[L][j])
                elif UBtype == 'LSE':
                    bnd = p[j] <= (lsm_u[m, j] * sm_l[m, j] - lsm_l[m, j] * sm_u[m, j] - (sm_u[m, j] - sm_l[m, j]) * (cvx.log_sum_exp(z[L]) - z[L][j])) / (lsm_u[m, j] - lsm_l[m, j])
                # Add constant bound
                bnd2 = p[j] <= sm_u[m, j]
            cons.append(bnd2)
            cons.append(bnd)
    
        # Sum to 1 constraint
        cons.append(cvx.sum(p) == 1)
    
        # SCORING FUNCTION
        
        if scoring == 'NLL':
            # Negative log-likelihood: equivalently minimize probability of correct class
            score_ub = -p[yt[i]]
        elif scoring == 'Brier':
            # Brier score: linear upper bound
            others = np.arange(d) != yt[i]
            score_ub = -(2 - sm_l[:, yt[i]].mean() - sm_u[:, yt[i]].mean()) * p[yt[i]] + (sm_l[:, others].mean(axis=0) + sm_u[:, others].mean(axis=0)) @ p[others] - np.dot(sm_l.mean(axis=0), sm_u.mean(axis=0)) + 1 
            
        # SOLVE PROBLEM
        
        prob = cvx.Problem(cvx.Maximize(score_ub), cons)
        if network == 'mnist-large':
            prob.solve(solver='SCS', verbose=True, acceleration_lookback=0, max_iters=int(1e5))
        else:
            prob.solve(solver='SCS', verbose=True, acceleration_lookback=0, max_iters=int(1e5))

        print(f'Status: {prob.status}')
        print(f'Objective value = {prob.value}')
        
        # SAVE RESULTS
        obj[m] = prob.value
        logits[m] = z[L].value
        probs[m] = p.value
        
    obj = obj.mean()
    print(f'Mean objective value = {obj}')
    
    data = [obj, logits, probs]
    with open(f'results/{network}_ind{i}_eps{eps}_lb{LBtype}_ub{UBtype}_score{scoring}_separate_results.pickle', 'wb') as f:
        pickle.dump(data, f)
