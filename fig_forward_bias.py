
import matplotlib.pyplot as plt

def sn_formula_cycle(n,i, r):
    numerator = 1 - (2 / ((2 * r) ** i)) + (1 / (2 * r))
    denominator = 1 - (2 / ((2 * r) ** n)) + (1 / (2 * r))
    return numerator / denominator

def binary_find(n, i):
    return (n-1)/(n-i)

range_n = range(3,11)
r_values = [[] for _ in range_n]
for n in range_n: 
    for i in range(1, n):
        r = binary_find(n, i)
        r_values[n-3].append(r)

n_min = 4
n_max = 11
font_size = 16
plt.figure(figsize=(8, 5))
cycle = [1]
for i in range(2, n_max-1):
    cycle.append(2)

    
from better_plots.better_plots import set_defaults, use_science, fig, save

use_science("science", "ieee")

f, ax = fig(font=12, aspect=.7)
for i, n in enumerate(range(n_min, n_max)):

    t_values = list(range(1, n))
    ax.plot(t_values, r_values[n-3], marker='o', linestyle='-', label=f'$N = {n}$')


ax.set_xlabel('Number of mutant individuals, $i$')
ax.set_ylabel(r'Forward bias, $\gamma_i$')

ax.grid(True, linestyle='--', linewidth=0.7, color='gray', alpha=0.7)


ax.axhline(y=1, color='black', linestyle='--', label=r'$\gamma_i(K_N) = 1$')
ax.legend( loc='best',  frameon=True, shadow=True)

save(f, "fig_forward_bias.pdf")