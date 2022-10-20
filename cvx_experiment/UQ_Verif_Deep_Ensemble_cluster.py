#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robustness verification of uncertainty estimates
"""

import pickle
import numpy as np
from scipy.special import expit, logsumexp, softmax
from tensorflow.keras.datasets import mnist
import cvxpy as cvx
import argparse

parser = argparse.ArgumentParser(description='Example.')
parser.add_argument('--eps', type=float, default=0.008, help="0.008 0.012 0.016")
parser.add_argument('--index', type=int, default=0, help="< 100")
parser.add_argument('--lb', type=str, default="ER", help="lin or ER or LSE")
parser.add_argument('--ub', type=str, default="LSE", help="lin or LSE")
parser.add_argument('--scoring', type=str, default="NLL", help="NLL or Brier")
parser.add_argument('--network', type=str, default="mnist", help="mnist or mnist-large or robust_mnist-large")
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

    # ENSEMBLE DIMENSIONS
    # Number of models in ensemble
    M = 5
    # Number of hidden layers
    if network == 'mnist':
        L = 2
    elif network in ['mnist-large', 'robust_mnist-large']:
        L = 3
    # Number of classes
    d = 10
    
    # LOAD NN WEIGHTS
    w = []
    b = []
    # Iterate over models
    for m in range(M):
        W = np.load(f'networks/{network}-{m}.npz')
    
        # Put weights into list of arrays
        w.append([None] * (L + 1))
        b.append([None] * (L + 1))
        for ll in range(L + 1):
            w[m][ll] = W['w' + str(ll + 1)]
            b[m][ll] = W['b' + str(ll + 1)]

    # LOAD MNIST TEST IMAGES AND BOUNDS
    (X, y), (Xt, yt) = mnist.load_data()
    # Number of test images to use
    N = [args.index]
    
    # Results arrays
    obj = np.zeros(N)
    logits = np.zeros((M, d))
    probs = np.zeros((M, d))
    
    # ITERATE OVER IMAGES
    for i in N:
        
        # LOAD NEURON BOUNDS
        
        lbs = []
        ubs = []
        # Iterate over models
        for m in range(M):
            with open(f"./bounds/bounds_net{suffix}{m}_ind{i}_eps{eps}.pickle", 'rb') as fp:
                bounds = pickle.load(fp)
            lbs.append(bounds["lbs"])
            ubs.append(bounds["ubs"])
        
        # INPUT LAYER
        
        mid0 = Xt[i].flatten() / 255
        # Input variables
        D = len(mid0)
        x0 = cvx.Variable(D)
        # Input bounds
        l0 = np.maximum(mid0 - eps, 0)
        u0 = np.minimum(mid0 + eps, 1)
        # Initialize list of constraints
        cons = [x0 >= l0, x0 <= u0]

        # HIDDEN LAYERS
        
        # Initialize lists of quantities over models
        z = [None] * M
        xa = [None] * M
        xu = [None] * M
        l = [None] * M
        u = [None] * M
        inactive = [None] * M
        active = [None] * M
        unstable = [None] * M        

        # Iterate over models
        for m in range(M):
            # Initialize lists over layers
            z[m] = [None] * (L + 1)
            xa[m] = [None] * L
            xu[m] = [None] * L
            l[m] = [None] * (L + 1)
            u[m] = [None] * (L + 1)
            inactive[m] = [None] * L
            active[m] = [None] * L
            unstable[m] = [None] * L
            
            # Iterate over hidden layers
            for ll in range(L):
                # Bounds on pre-activation neurons
                # NEEDS TO BE MADE MODEL-SPECIFIC FOR DEEP ENSEMBLE
                l[m][ll] = np.array(lbs[m][2*ll+1])
                u[m][ll] = np.array(ubs[m][2*ll+1])
        
                # Drop inactive neurons
                inactive[m][ll] = u[m][ll] <= 0
                w[m][ll] = w[m][ll][:, ~inactive[m][ll]]
                b[m][ll] = b[m][ll][~inactive[m][ll]]
                l[m][ll] = l[m][ll][~inactive[m][ll]]
                u[m][ll] = u[m][ll][~inactive[m][ll]]
                w[m][ll+1] = w[m][ll+1][~inactive[m][ll], :]
        
                # Find unstable and active neurons
                unstable[m][ll] = l[m][ll] < 0
                active[m][ll] = ~unstable[m][ll]
        
                # Affine transformation
                if ll == 0:
                    # Input layer
                    z[m][ll] = x0 @ w[m][ll] + b[m][ll]
                else:
                    z[m][ll] = xa[m][ll-1] @ w[m][ll][active[m][ll-1], :] + xu[m][ll-1] @ w[m][ll][unstable[m][ll-1], :] + b[m][ll]
                
                # Define active neurons
                xa[m][ll] = z[m][ll][active[m][ll]]
        
                # Define and constrain unstable neurons
                nUnstable = unstable[m][ll].sum()
                if nUnstable:
                    xu[m][ll] = cvx.Variable(nUnstable, nonneg=True)
                    cons.append(xu[m][ll] >= z[m][ll][unstable[m][ll]])
                    g = u[m][ll][unstable[m][ll]] / (u[m][ll][unstable[m][ll]] - l[m][ll][unstable[m][ll]])
                    cons.append(xu[m][ll] <= cvx.multiply(g, z[m][ll][unstable[m][ll]] - l[m][ll][unstable[m][ll]]))
                else:
                    # No unstable neurons
                    xu[m][ll] = np.zeros(nUnstable)
        
        # LOGITS
        
        # Iterate over models
        jmax = np.empty(M, dtype=int)
        for m in range(M):
            z[m][L] = xa[m][L-1] @ w[m][L][active[m][L-1], :] + xu[m][L-1] @ w[m][L][unstable[m][L-1], :] + b[m][L]
            # Bounds on logits
            # NEEDS TO BE MODEL-SPECIFIC FOR DEEP ENSEMBLE
            l[m][L] = np.array(lbs[m][2*L+1])
            u[m][L] = np.array(ubs[m][2*L+1])
            # Index of largest logit in terms of midpoint
            jmax[m] = (l[m][L] + u[m][L]).argmax()

        # SOFTMAX
        
        # Output probabilities
        p = cvx.Variable((M, d), nonneg=True)
        
        # Load and compute bounds
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

        # Constrain softmax outputs
        # Iterate over models
        diffs = [None] * M
        for m in range(M):
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
                        bnd = p[m, j] >= a_lin_l @ z[m][L] + b_lin_l
                    elif LBtype == 'ER' or j == jmax[m]:
                        # LSE2 bound same as ER bound when j = jmax
                        # Differences with z_j
                        others = np.arange(d) != j
                        diffs[m] = z[m][L][others] - z[m][L][j]
                        # Bounding log probability avoids having to use a second-order cone
                        bnd = cvx.log(p[m, j]) >= -cvx.log(1 + cvx.sum((cvx.multiply(np.exp(diffs_l[m, j]), diffs_u[m, j] - diffs[m]) + cvx.multiply(np.exp(diffs_u[m, j]), diffs[m] - diffs_l[m, j])) / (diffs_u[m, j] - diffs_l[m, j])))
                    elif LBtype == 'LSE':
                        # LSE2 bound for j != jmax
                        # Differences with z_{jmax}
                        others = np.arange(d) != jmax[m]
                        diffs[m] = z[m][L] - z[m][L][jmax[m]]
                        bnd = cvx.log(p[m, j]) >= diffs[m][j] - cvx.log(1 + cvx.sum((cvx.multiply(np.exp(diffs_l[m, jmax[m]]), diffs_u[m, jmax[m]] - diffs[m][others]) + cvx.multiply(np.exp(diffs_u[m, jmax[m]]), diffs[m][others] - diffs_l[m, jmax[m]])) / (diffs_u[m, jmax[m]] - diffs_l[m, jmax[m]])))
                        # STILL NEED TO IMPLEMENT LSE1 BOUND
                    # Add constant bound
                    bnd2 = p[m, j] >= sm_l[m, j]
                else:
                    # Incorrect class, need upper bound on probability
                    if UBtype == 'lin':
                        # Linear ER bound
                        a_lin_u = np.zeros(d)
                        others = np.arange(d) != j
                        a_lin_u[others] = -np.exp(diffs_t) / (den_l * den_u)
                        a_lin_u[j] = -a_lin_u[others].sum()
                        b_lin_u = 1 / den_l + 1 / den_u - (1 + np.dot(np.exp(diffs_t), 1 - diffs_t)) / (den_l * den_u)
                        bnd = p[m, j] <= a_lin_u @ z[m][L] + b_lin_u
                    elif UBtype == 'ER':
                        # CAN'T USE: HAVING MORE THAN 2 OF THESE BOUNDS PREVENTS CONVERGENCE 
                        #bnd = p[j] <= sm_l[j] + sm_u[j] - sm_l[j] * sm_u[j] * (1 + cvx.sum(cvx.exp(diffs[j])))
                        bnd = p[m, j] <= sm_l[m, j] + sm_u[m, j] - sm_l[m, j] * sm_u[m, j] * cvx.exp(cvx.log_sum_exp(z[m][L]) - z[m][L][j])
                    elif UBtype == 'LSE':
                        bnd = p[m, j] <= (lsm_u[m, j] * sm_l[m, j] - lsm_l[m, j] * sm_u[m, j] - (sm_u[m, j] - sm_l[m, j]) * (cvx.log_sum_exp(z[m][L]) - z[m][L][j])) / (lsm_u[m, j] - lsm_l[m, j])
                    # Add constant bound
                    bnd2 = p[m, j] <= sm_u[m, j]
                cons.append(bnd2)
                cons.append(bnd)
        
            # Sum to 1 constraint
            cons.append(cvx.sum(p[m, :]) == 1)
        
        # SCORING FUNCTION
        
        if scoring == 'NLL':
            # Negative log-likelihood: equivalently minimize probability of correct class
            score_ub = -cvx.sum(p[:, yt[i]]) / M
        elif scoring == 'Brier':
            # Brier score: linear upper bound
            others = np.arange(d) != yt[i]
            score_ub = -(2 - sm_l[:, yt[i]].mean() - sm_u[:, yt[i]].mean()) * cvx.sum(p[:, yt[i]]) / M + (sm_l[:, others].mean(axis=0) + sm_u[:, others].mean(axis=0)) @ cvx.sum(p[:, others], axis=0) / M - np.dot(sm_l.mean(axis=0), sm_u.mean(axis=0)) + 1 
            
        # SOLVE PROBLEM
        
        prob = cvx.Problem(cvx.Maximize(score_ub), cons)
        if network == 'mnist-large':
            prob.solve(solver='SCS', verbose=True, acceleration_lookback=0, max_iters=int(1e5))
        else:
            prob.solve(solver='SCS', verbose=True, acceleration_lookback=0, max_iters=int(1e4))

        print(f'Status: {prob.status}')
        print(f'Objective value = {prob.value}')
        
        # SAVE RESULTS
        obj = prob.value
        for m in range(M):
            logits[m] = z[m][L].value
        probs = p.value
        
    data = [obj, logits, probs]
    with open(f'results/{network}_ind{i}_eps{eps}_lb{LBtype}_ub{UBtype}_score{scoring}_results.pickle', 'wb') as f:
        pickle.dump(data, f)
