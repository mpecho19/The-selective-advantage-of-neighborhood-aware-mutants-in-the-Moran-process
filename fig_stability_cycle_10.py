
import numpy as np
from libs.Numeric_Stationary import Numeric_Solver_Stationary
# from scipy.sparse import csc_matrix, lil_matrix
# from scipy.sparse.linalg import spsolve
import networkx as nx

r_size = 1
delta = 0.01
r= 1
N = 10
all_average = []
all_average_normal = []
u_values = [ 0.5, 0.1, 0.01]
range_r = np.arange(1 - r_size + delta, 1 + r_size + delta, delta)

for u in u_values:
    average_list = []
    average_normal_list = []
    for index, r in enumerate(range_r):
        print(r)
        G = nx.cycle_graph(N)
        numeric_solver = Numeric_Solver_Stationary(G, coefficient=r, mutation_rate=u, replacer=True)
        average_list.append(numeric_solver.find_expected_number_of_mutants())
        numeric_solver_normal = Numeric_Solver_Stationary(G, coefficient=r, mutation_rate=u, replacer=False)
        average_normal_list.append(numeric_solver_normal.find_expected_number_of_mutants())

    all_average.append(average_list)
    all_average_normal.append(average_normal_list)


from better_plots.better_plots import set_defaults, use_science, fig, save
use_science("science", "ieee")

f, ax = fig(font=10, aspect=.7)

import matplotlib.pyplot as plt


ax.axvline(1/2, color='black', linestyle=':', linewidth=1.5, alpha=1)
ax.axhline(N/2, color='black', linestyle=':', linewidth=1.5, alpha=1)

colors = ['red', 'green', 'blue', 'purple']
for i, u in enumerate(u_values):
    ax.plot(range_r, all_average[i], label=f'Replacers ($u={u}$)', color=colors[i])
    ax.plot(range_r, all_average_normal[i], linestyle='--', label=f'Oblivious mutants ($u={u}$)', color=colors[i])
ax.set_xlabel('Mutant fitness, $r$')
ax.set_ylabel(r'Mutant frequency, $\lambda$')
from matplotlib.lines import Line2D


legend_elements = [
    Line2D([0], [0], marker='o', linestyle='none',
           markerfacecolor=c, markersize=8, markeredgecolor='None', label=f'$u={u}$')
    for c, u in zip(colors, u_values)
]

ax.legend(handles=legend_elements, loc='lower right')
ax.grid(True)

save(f, f"fig_stability_cycle_{N}.pdf") 