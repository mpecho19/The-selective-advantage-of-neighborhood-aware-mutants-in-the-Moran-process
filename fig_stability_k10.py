

from matplotlib import colors
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse import lil_matrix


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


r_size = 1
delta = 0.01
r= 1
N = 10
all_average = []
all_average_normal = []
u_values = [0.5, 0.1, 0.01]
range_r = np.arange(1 - r_size, 1 + r_size + delta, delta)
for u in u_values:
    average_list = []
    average_normal_list = []
    for index, r in enumerate(range_r):


        P_sparse = replacer_transition_matrix(N, r, u)

        pi = find_stationary_distribution(P_sparse.toarray())
        if r == 1:
            print(pi)
        average = 0
        for i in range(len(pi)):
            average += i*pi[i]
        average_list.append(average)     

        P_sparse_normal = lil_matrix((N+1, N+1))
        P_sparse_normal[0,0] = 1- u
        P_sparse_normal[0,1] = u
        P_sparse_normal[N,N] = 1-u
        P_sparse_normal[N, N-1] = u
        for i in range(1,N):
            P_sparse_normal[i, i+1] = (1-u)*i*r*(N-i)/(N-1)/total_fitness(i, N, r) + u*(N-i)*(N-i-1)/(N-1)/total_fitness(i, N, r)
            P_sparse_normal[i, i-1] = (1-u)*(N-i)*i/(N-1)/total_fitness(i, N, r) + u*i*r*(i-1)/(N-1)/total_fitness(i, N, r)
            P_sparse_normal[i, i] = 1 - P_sparse_normal[i, i+1] - P_sparse_normal[i, i-1]

        P_sparse_normal = P_sparse_normal.tocsc()
        pi_normal = find_stationary_distribution(P_sparse_normal.toarray())
        average_normal = 0
        for i in range(len(pi_normal)):
            average_normal += i*pi_normal[i]
        average_normal_list.append(average_normal)
    all_average.append(average_list)
    all_average_normal.append(average_normal_list)



from better_plots.better_plots import set_defaults, use_science, fig, save
use_science("science", "ieee")

f, ax = fig(font=13.8, aspect=.7)


ax.text(1/np.e, -0.1, r'$1/e$', ha='center', va='top',
    transform=ax.get_xaxis_transform(), clip_on=False)
    # vertical line at x = 1/e
ax.axvline(1/np.e, color='black', linestyle=':', linewidth=1.5, alpha=1)

ax.axhline(N/2, color='black', linestyle=':', linewidth=1.5, alpha=1)

colors = ['red', 'green', 'blue', 'purple']
for i, u in enumerate(u_values):
    ax.plot(range_r, all_average[i], color=colors[i]) 
    ax.plot(range_r, all_average_normal[i], linestyle='--', color=colors[i]) 
    
from matplotlib.lines import Line2D


legend_elements = [
    Line2D([0], [0], marker='o', linestyle='none',
           markerfacecolor=c, markersize=8, markeredgecolor='None', label=f'$u={u}$')
    for c, u in zip(colors, u_values)
]

ax.legend(handles=legend_elements, loc='best')
ax.set_xlabel('Mutant fitness, $r$')
ax.set_ylabel(r'Mutant frequency, $\lambda$')


ax.grid(True)

save(f, f"fig-stability-k{N}.pdf")
