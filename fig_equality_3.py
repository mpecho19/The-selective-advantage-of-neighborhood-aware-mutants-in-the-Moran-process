from scipy.sparse import  csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse import lil_matrix
import numpy as np

def total_fitness(i, N, r):
    return i*r + (N - i)*1

def replacer_transition_matrix(N, r, u):
    P_sparse = lil_matrix((N+1, N+1))
    P_sparse[0, 0] = 1 - u
    P_sparse[0, 1] = u
    P_sparse[N, N] = 1 - u
    P_sparse[N, N-1] = u
    for i in range(1, N):
        P_sparse[i, i+1] = (1-u)*i*r/total_fitness(i, N, r) + u*(N-i)*(N-i-1)/(N-1)/total_fitness(i, N, r)
        P_sparse[i, i-1] = (1-u)*(N-i)*i/(N-1)/total_fitness(i, N, r) #+ u*i*r/total_fitness(i, N, r)
        P_sparse[i, i] = 1 - P_sparse[i, i+1] - P_sparse[i, i-1]    
    P_sparse = P_sparse.tocsc()
    return P_sparse

def find_stationary_distribution(P):
    n = P.shape[0]
    PT = P.transpose()
    A = csc_matrix(PT - np.eye(n))
    A = A.tolil()
    A[-1, :] = 1
    A = A.tocsc()
    b = np.zeros(n)
    b[-1] = 1
    pi = spsolve(A, b)
    
    return pi

def replacer_frequncy(distribution):
    average = 0
    for i in range(len(distribution)):
        average += i*distribution[i]
    return average


def find_frequency_equality(n, u):
    oblivious_frequency = n/2
    fit_max = 1 
    fit_min = 0

    while fit_max - fit_min > 1e-10:
        fit = (fit_max + fit_min) / 2
        P = replacer_transition_matrix(n, fit, u)
        stationary_dist = find_stationary_distribution(P)
        avg_freq = replacer_frequncy(stationary_dist)
        if avg_freq < oblivious_frequency:
            fit_min = fit
        else:
            fit_max = fit
    
    return (fit_max + fit_min) / 2



n_min = 3
n_max = 50

n_range = range(n_min, n_max + 1)
u_valuse = [0.1, 0.01]
list_of_freq_eq = []
for n in n_range:
    freq_eq_values = []
    for u in u_valuse:
        freq_eq = find_frequency_equality(n, u)
        freq_eq_values.append(1-freq_eq)
    list_of_freq_eq.append(freq_eq_values)

import matplotlib.pyplot as plt
from better_plots.better_plots import set_defaults, use_science, fig, save

use_science("science", "ieee")

f, ax = fig(font= 21.5, aspect=.8)



ax.axhline(1-1/np.e, color='black', linestyle='--', linewidth=1.2 )
for i, u in enumerate(u_valuse):
    freq_eq_values = [list_of_freq_eq[j][i] for j in range(len(n_range))]
    ax.plot(n_range, freq_eq_values, linestyle='-', label=f'$u = {u}$')

ax.set_xticks([0, 25, 50])
ax.set_xlabel('Population size, $N$')
ax.set_ylabel(r'Cost to match $\lambda$')
ax.grid(True, linestyle='--', linewidth=0.7, color='gray', alpha=0.7)
ax.legend(loc='best', frameon=True, shadow=True)


save(f, "fig_equality_3.pdf")