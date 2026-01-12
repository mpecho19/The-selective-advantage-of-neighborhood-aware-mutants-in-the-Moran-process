

def replacer_elimination_probability(n,i, r):
    total_den = 0 
    total_num = 0
    for j in range(1, n):
        prod = 1
        for k in range(1, j+1):
            prod *= r* (n-1)/k
        total_den += prod
    
    for j in range(1, i):
        prod = 1
        for k in range(1, j+1):
            prod *= r* (n-1)/k
        total_num += prod
    return (1 + total_num) / (1 + total_den)

def binary_find(n):
    r_min = 0.01
    r_max = 1
    while r_max - r_min > 1e-10:
        r = (r_max + r_min) / 2
        elim_prob = replacer_elimination_probability(n, 1, r)
        print(f"n={n}, r={r}, elim_prob={elim_prob}")
        if elim_prob < 1/n:
            r_max = r
        else:
            r_min = r
    
    return (r_max + r_min) / 2

n_min = 3
n_max = 100
n_range = range(n_min, n_max + 1)

r_values = []
for n in n_range:  
    r = binary_find(n)
    r_values.append(1-r)

import matplotlib.pyplot as plt
from better_plots.better_plots import set_defaults, use_science, fig, save
use_science("science", "ieee")

f, ax = fig(font= 21.5, aspect=.8)

# plt.figure(figsize=(8, 5))
ax.plot(n_range, r_values, linestyle='-', label='Replacer')

ticks = [round(0.4 + i * 0.1, 2) for i in range(7)] 
ax.set_yticks(ticks)
ax.set_xlabel('Population size, $N$')
ax.set_ylabel(r'Cost to match $\Psi$')
ax.set_ylim(0.37, 1.03)
ax.grid(True, linestyle='--', linewidth=0.7, color='gray', alpha=0.7)
# plt.legend(loc='best', frameon=True, shadow=True)
# horizontal reference line at 1 - 1/e
# ax.axhline(1/np.e, color='red', linestyle='--', linewidth=1.2, label=r'$1-\frac{1}{e}$')
# ax.legend(loc='best', frameon=True, shadow=True)
save(f, "fig_equality_2.pdf")
    

