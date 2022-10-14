#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 09:06:48 2022

@author: denniswei

Compare log-sum-exp, exponential-reciprocal, and linear bounds on softmax
Generate midpoints by sampling from Dirichlet distribution
"""

import argparse
from itertools import product
import numpy as np
from numpy.random import default_rng
from scipy.special import expit, logsumexp, softmax
import cvxpy as cvx
from tqdm import tqdm
from pathlib import Path
import pickle

def eval_ER_u(X, sm_l, sm_u):
    """
    Evaluate exponential-reciprocal upper bound at points in X (shape (n, d))
    """
    ER_u = sm_l + sm_u - sm_l * sm_u * (1 + np.exp(X[:, 1:] - X[:, [0]]).sum(axis=1))
    return ER_u

def eval_LSE_u(X, sm_l, sm_u, lsm_l, lsm_u):
    """
    Evaluate log-sum-exp upper bound at points in X (shape (n, d))
    """
    LSE_u = (lsm_u * sm_l - lsm_l * sm_u - (sm_u - sm_l) * (logsumexp(X, axis=1) - X[:, 0])) / (lsm_u - lsm_l)
    return LSE_u

def eval_ER_l(X, diffs_l, diffs_u):
    """
    Evaluate exponential-reciprocal lower bound at points in X (shape (n, d))
    """
    diffs = X[:, 1:] - X[:, [0]]
    # Coefficients of affine function of diffs
    a_ER_l = (np.exp(diffs_u) - np.exp(diffs_l)) / (diffs_u - diffs_l)
    b_ER_l = ((diffs_u * np.exp(diffs_l) - diffs_l * np.exp(diffs_u)) / (diffs_u - diffs_l)).sum()
    
    ER_l = 1 / (1 + np.dot(diffs, a_ER_l) + b_ER_l)
    return ER_l

def eval_LSE_l(X, lse_l, lse_u, lsm_l, lsm_u, l, u):
    """
    Evaluate log-sum-exp lower bound at points in X (shape (n, d))
    """
    d = X.shape[1]
    if d == 2:
        diff = X[:, 1].copy()
    else:
        # Log-linear upper bound on LSE(X[:, 1:])
        a = (np.exp(u[1:]) - np.exp(l[1:])) / (u[1:] - l[1:])
        b = ((u[1:] * np.exp(l[1:]) - l[1:] * np.exp(u[1:])) / (u[1:] - l[1:])).sum()
        diff = np.log(np.dot(X[:, 1:], a) + b)
    diff -= X[:, 0]
    
    LSE_l = np.exp((lse_u * lsm_u - lse_l * lsm_l - (lsm_u - lsm_l) * diff) / (lse_u - lse_l))
    return LSE_l

def eval_hybrid_l(X, l, u):
    """
    Evaluate hybrid LSE-ER lower bound at points in X (shape (n, d))
    """
    # Linear upper bound on sum of exponentials
    a = (np.exp(u) - np.exp(l)) / (u - l)
    b = ((u * np.exp(l) - l * np.exp(u)) / (u - l)).sum()
    se_u = np.dot(X, a) + b
    hybrid_l = np.exp(X[:, 0]) / se_u
    return hybrid_l

def eval_hybrid2_l(X, l, u):
    """
    Evaluate second hybrid LSE-ER lower bound at points in X (shape (n, d))
    """
    # Component with largest midpoint
    jmax = (l + u).argmax()
    # Differences w.r.t. largest component and bounds on differences
    d = X.shape[1]
    others = np.arange(d) != jmax
    diffs = X - X[:, [jmax]]
    diffs_l = l[others] - u[jmax]
    diffs_u = u[others] - l[jmax]
    # Linear upper bound on sum of exponentials
    a = (np.exp(diffs_u) - np.exp(diffs_l)) / (diffs_u - diffs_l)
    b = ((diffs_u * np.exp(diffs_l) - diffs_l * np.exp(diffs_u)) / (diffs_u - diffs_l)).sum()
    se_u = 1 + np.dot(diffs[:, others], a) + b
    hybrid_l = np.exp(diffs[:, 0]) / se_u
    return hybrid_l

def eval_hybrid3_l(X, l, u):
    """
    Evaluate third hybrid LSE-ER lower bound at points in X (shape (n, d))
    """
    # Component with largest midpoint
    jmax = (l + u).argmax()
    # Differences w.r.t. largest component and bounds on differences
    d = X.shape[1]
    others = np.arange(d) != jmax
    diffs = X - X[:, [jmax]]
    diffs_l = l[others] - u[jmax]
    diffs_u = u[others] - l[jmax]
    # Linear upper bound on sum of log(1 + e^x)'s
    a = (np.log(1 + np.exp(diffs_u)) - np.log(1 + np.exp(diffs_l))) / (diffs_u - diffs_l)
    b = ( (diffs_u * np.log(1 + np.exp(diffs_l)) - diffs_l * np.log(1 + np.exp(diffs_u))) / (diffs_u - diffs_l) ).sum()
    hybrid_l = diffs[:, 0] - np.dot(diffs[:, others], a) - b
    return np.exp(hybrid_l)

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=2, help='input dimension')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    args = parser.parse_args()
    d = args.d
    seed = args.seed

    # Random number generator
    rng = default_rng(seed=seed)

    # Number of repetitions
    R = 100
    # Dirichlet parameters: component of mean larger than the others
    idxMax = [0, -1]
    # Dirichlet parameters: mean/concentration parameter of largest component
    muMax = np.array([0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999])
    alphaMax = (d - 1) * muMax / (1 - muMax)
    # Add alphaMax = 1 (uniform distribution)
    alphaMax = np.insert(alphaMax, 0, 1)
    num_alphaMax = len(alphaMax)
    # ell-infinity norm bounds
    eps = [0.01, 0.1, 0.2, 0.5, 1, 2, 5]
    num_eps = len(eps)
    # Number of points sampled from input region
    N = 1000
    
    # Initialize results arrays
    lbs = ['sm_l', 'ER_l', 'LSE_l', 'hybrid_l', 'hybrid2_l', 'hybrid3_l', 'lin_l']
    ubs = ['sm_u', 'ER_u', 'LSE_u', 'lin_u']
    pairs = list(product(['ER_l', 'LSE_l', 'hybrid_l', 'hybrid2_l', 'hybrid3_l'], ['ER_u', 'LSE_u'])) + [('lin_l', 'lin_u')]

#    sm_u_sm_l = np.zeros((R, num_eps))
#    
#    max_ER_u_ER_l = np.zeros((R, num_eps))
#    max_ER_u_LSE_l = np.zeros((R, num_eps))
#    max_ER_u_hybrid_l = np.zeros((R, num_eps))
#    max_ER_u_hybrid2_l = np.zeros((R, num_eps))
#    max_LSE_u_ER_l = np.zeros((R, num_eps))
#    max_LSE_u_LSE_l = np.zeros((R, num_eps))
#    max_LSE_u_hybrid_l = np.zeros((R, num_eps))
#    max_LSE_u_hybrid2_l = np.zeros((R, num_eps))
#    max_lin_u_lin_l = np.zeros((R, num_eps))
    
    mean_sm_l = np.zeros((R, 2, num_alphaMax, num_eps))
    mean_ER_l = np.zeros((R, 2, num_alphaMax, num_eps))
    mean_LSE_l = np.zeros((R, 2, num_alphaMax, num_eps))
    mean_hybrid_l = np.zeros((R, 2, num_alphaMax, num_eps))
    mean_hybrid2_l = np.zeros((R, 2, num_alphaMax, num_eps))
    mean_hybrid3_l = np.zeros((R, 2, num_alphaMax, num_eps))
    mean_lin_l = np.zeros((R, 2, num_alphaMax, num_eps))
    mean_sm_u = np.zeros((R, 2, num_alphaMax, num_eps))
    mean_ER_u = np.zeros((R, 2, num_alphaMax, num_eps))
    mean_LSE_u = np.zeros((R, 2, num_alphaMax, num_eps))
    mean_lin_u = np.zeros((R, 2, num_alphaMax, num_eps))

#    mean_ER_u_ER_l = np.zeros((R, num_eps))
#    mean_ER_u_LSE_l = np.zeros((R, num_eps))
#    mean_ER_u_hybrid_l = np.zeros((R, num_eps))
#    mean_ER_u_hybrid2_l = np.zeros((R, num_eps))
#    mean_LSE_u_ER_l = np.zeros((R, num_eps))
#    mean_LSE_u_LSE_l = np.zeros((R, num_eps))
#    mean_LSE_u_hybrid_l = np.zeros((R, num_eps))
#    mean_LSE_u_hybrid2_l = np.zeros((R, num_eps))
#    mean_lin_u_lin_l = np.zeros((R, num_eps))
    
    # Iterate over index of largest Dirichlet component
    for mm, m in enumerate(idxMax):
        # Iterate over concentration parameter of largest component
        for aa, a in enumerate(alphaMax):
            # Dirichlet concentration parameters
            alpha = np.ones(d)
            alpha[m] = a
            print(f'alpha[{m}] = {a}')
    
            # Iterate over repetitions
            for r in tqdm(range(R)):
                # Sample from Dirichlet distribution
                mid = rng.dirichlet(alpha)
                # Convert to logits and center
                mid = np.log(mid / mid[0])
                mid -= mid.mean()
                
                # Iterate over ell-infinity bounds
                for ee, e in enumerate(eps):
                    # Input region lower and upper bounds
                    l = mid - e
                    u = mid + e
                    
                    # FOR NOW CONSIDER ONLY FIRST OUTPUT OF SOFTMAX (CORRESPONDING TO INPUT 1)
                    
                    # CONSTANT BOUNDS
                    # Bounds on diffs := x[1:] - x[0]
                    diffs_l = l[1:] - u[0]
                    diffs_u = u[1:] - l[0]
                    # Bounds on log-softmax
                    # for scipy>=1.8.0, can use log_expit instead
                    lsm_l = np.log(expit(-logsumexp(diffs_u)))
                    lsm_u = np.log(expit(-logsumexp(diffs_l)))
                    # Bounds on softmax
                    sm_l = np.exp(lsm_l)
                    sm_u = np.exp(lsm_u)
#                    # sm_u - sm_l gap
#                    sm_u_sm_l[r, ee] = sm_u - sm_l
#                    
#                    # MAXIMUM GAPS
#                    
#                    # ER AND LSE UPPER BOUNDS
#                    # Optimization variable and constraints
#                    x = cvx.Variable(d)
#                    cons = [l <= x, x <= u]
#                    
#                    # ER upper bound
#                    ER_u = sm_l + sm_u - sm_l * sm_u * (1 + cvx.sum(cvx.exp(x[1:] - x[0])))
#                    # LSE upper bound
#                    LSE_u = (lsm_u * sm_l - lsm_l * sm_u - (sm_u - sm_l) * (cvx.log_sum_exp(x) - x[0])) / (lsm_u - lsm_l)
#                    
#                    # ER AND LSE LOWER BOUNDS
#                    # ER lower bound
#                    diffs = x[1:] - x[0]
#                    ER_l = cvx.power(1 + cvx.sum((cvx.multiply(np.exp(diffs_l), diffs_u - diffs) + cvx.multiply(np.exp(diffs_u), diffs - diffs_l)) / (diffs_u - diffs_l)), -1)
                    
                    # LSE lower bound
                    # Bounds on log-sum-exp
                    lse_l = logsumexp(l[1:]) - u[0]
                    lse_u = logsumexp(u[1:]) - l[0]
#                    if d == 2:
#                        diff = x[1]
#                    else:
#                        # Log-linear upper bound on LSE(x[1:])
#                        diff = cvx.log(cvx.sum((cvx.multiply(np.exp(l[1:]), u[1:] - x[1:]) + cvx.multiply(np.exp(u[1:]), x[1:] - l[1:])) / (u[1:] - l[1:])))
#                    diff = diff - x[0]
#                    LSE_l = cvx.exp((lse_u * lsm_u - lse_l * lsm_l - (lsm_u - lsm_l) * diff) / (lse_u - lse_l))
#                    
#                    # Hybrid lower bound
#                    se_u = cvx.sum((cvx.multiply(np.exp(l), u - x) + cvx.multiply(np.exp(u), x - l)) / (u - l))
#                    hybrid_l = cvx.exp(x[0] - cvx.log(se_u))
#                    
#                    # Second hybrid lower bound
#                    # Component with largest midpoint
#                    jmax = mid.argmax()
#                    others = np.arange(d) != jmax
#                    # Differences w.r.t. largest component and bounds on differences
#                    diffsMax = x - x[jmax]
#                    diffsMax_l = l[others] - u[jmax]
#                    diffsMax_u = u[others] - l[jmax]
#                    # Linear upper bound on sum of exponentials
#                    se_u2 = cvx.sum((cvx.multiply(np.exp(diffsMax_l), diffsMax_u - diffsMax[others]) + cvx.multiply(np.exp(diffsMax_u), diffsMax[others] - diffsMax_l)) / (diffsMax_u - diffsMax_l))
#                    hybrid2_l = cvx.exp(diffsMax[0] - cvx.log(1 + se_u2))
#                    
#                    # LSE-ER MAXIMUM GAPS
#                    for lb, ub in pairs[:8]:
#                        obj = cvx.Maximize(eval(ub) - eval(lb))
#                        prob = cvx.Problem(obj, cons)
#                        try:
#                            eval('max_' + ub + '_' + lb)[r, ee] = prob.solve(solver='SCS')
#                        except cvx.error.SolverError:
#                            print(f"Solver 'ECOS' failed on r = {r}, eps = {e}, lb = {lb}, ub = {ub}. Trying with 'SCS'.")
#                            eval('max_' + ub + '_' + lb)[r, ee] = prob.solve(solver='SCS')
                    
                    # LINEAR BOUNDS
                    # Tangent points for bounding exponentials from below
                    diffs_t = np.minimum(np.log((np.exp(diffs_u) - np.exp(diffs_l)) / (diffs_u - diffs_l)), diffs_l + 1)
                    # Lower and upper bounds on denominator of softmax
                    den_l = 1 + np.dot(np.exp(diffs_t), diffs_l - diffs_t + 1)
                    den_u = 1 + np.exp(diffs_u).sum()
                    # Tangent point for bounding reciprocal from below
                    den_t = max(np.sqrt(den_l * den_u), den_u / 2)
                    
                    # Coefficients of linear upper bound
                    a_lin_u = np.zeros(d)
                    a_lin_u[1:] = -np.exp(diffs_t) / (den_l * den_u)
                    a_lin_u[0] = -a_lin_u[1:].sum()
                    b_lin_u = 1 / den_l + 1 / den_u - (1 + np.dot(np.exp(diffs_t), 1 - diffs_t)) / (den_l * den_u)
                    
                    # Coefficients of linear lower bound
                    a_lin_l = np.zeros(d)
                    a_lin_l[1:] = -(np.exp(diffs_u) - np.exp(diffs_l)) / (diffs_u - diffs_l) / den_t**2
                    a_lin_l[0] = -a_lin_l[1:].sum()
                    b_lin_l = ((diffs_u * np.exp(diffs_l) - diffs_l * np.exp(diffs_u)) / (diffs_u - diffs_l)).sum()
                    b_lin_l = 1 / den_t * (2 - 1 / den_t * (1 + b_lin_l))
                    
#                    # Maximum gap between linear bounds
#                    a_lin_ul = a_lin_u - a_lin_l
#                    max_lin_u_lin_l[r, ee] = np.dot(a_lin_ul[a_lin_ul > 0], u[a_lin_ul > 0]) + np.dot(a_lin_ul[a_lin_ul < 0], l[a_lin_ul < 0])
#                    max_lin_u_lin_l[r, ee] += b_lin_u - b_lin_l
                    
                    # MEAN GAPS
                    
                    # Sample points uniformly from input region
                    X = rng.uniform(l, u, (N, d))
                    
                    # Evaluate softmax and bounds
                    sm = softmax(X, axis=1)[:, 0]
                    ER_u = eval_ER_u(X, sm_l, sm_u)
                    LSE_u = eval_LSE_u(X, sm_l, sm_u, lsm_l, lsm_u)
                    ER_l = eval_ER_l(X, diffs_l, diffs_u)
                    LSE_l = eval_LSE_l(X, lse_l, lse_u, lsm_l, lsm_u, l, u)
                    hybrid_l = eval_hybrid_l(X, l, u)
                    hybrid2_l = eval_hybrid2_l(X, l, u)
                    hybrid3_l = eval_hybrid3_l(X, l, u)
                    lin_u = np.dot(X, a_lin_u) + b_lin_u
                    lin_l = np.dot(X, a_lin_l) + b_lin_l
                    
                    # Compute mean gaps
                    for lb in lbs:
                        gap = sm - eval(lb)
                        eval('mean_' + lb)[r, mm, aa, ee] = gap.mean()
                        if gap.min() < 0:
                            print(f'Lower bound {lb} invalid! Min gap = {gap.min()}')
                    for ub in ubs:
                        gap = eval(ub) - sm
                        eval('mean_' + ub)[r, mm, aa, ee] = gap.mean()
                        if gap.min() < 0:
                            print(f'Upper bound {ub} invalid!')
#                    for lb, ub in pairs:
#                        gap = eval(ub) - eval(lb)
#                        eval('mean_' + ub + '_' + lb)[r, ee] = gap.mean()

    # Dictionary of results arrays
#    dictSave = {'sm_u_sm_l': sm_u_sm_l}
    dictSave = {}
#    for stat in ['max', 'mean']:
#        for lb, ub in pairs:
#            dictSave[stat + '_' + ub + '_' + lb] = eval(stat + '_' + ub + '_' + lb)
    for b in lbs + ubs:
        dictSave['mean_' + b] = eval('mean_' + b)
    # Save results to file
    filename = Path(__file__).stem + f'_d{d}'
    with open(filename + '.pkl', 'wb') as f:
        pickle.dump(dictSave, f)
