import numpy as np
# from libs.SaveSystem import SaveSystem
import matplotlib.pyplot as plt

def baseline(n,r):
    if r == 1:
        return 1/n
    return (1 - (1/r)) / (1 - (1/(r**n)))

def sn_formula_complete_fin(n,i, r):
    total_den = 0 
    total_num = 0
    for j in range(1, n):
        prod = 1
        for k in range(1, j+1):
            prod *= (1/r) * (1 - (k-1)/(n-1))
        total_den += prod
    
    for j in range(1, i):
        prod = 1
        for k in range(1, j+1):
            prod *= (1/r) * (1 - (k-1)/(n-1))
        total_num += prod
    return (1 + total_num) / (1 + total_den)

def ns_formula_complete_fin(n,i, r):
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

r= 1
n_range = range(4, 101)

baseline_vals = []
asymp_sqrt  = []
asymp_exp = []
for n in n_range:
    baseline_vals.append(baseline(n, r))
    asymp_sqrt.append(1/np.sqrt(n*np.pi /2))
    asymp_exp.append(2/np.e**(n-1))


n_filtered = []
sn_vals_filtered = []
ns_vals_filtered = []
one_over_n_filtered = []
for n in n_range:
    if n % 5 == 0:
        n_filtered.append(n)
        sn_vals_filtered.append(sn_formula_complete_fin(n, 1, r))
        ns_vals_filtered.append(ns_formula_complete_fin(n, 1, r))
        one_over_n_filtered.append(1/n)


from better_plots.better_plots import set_defaults, use_science, fig, save



use_science("science", "ieee")
f, ax = fig(font=14, aspect=.7)


# plt.figure(figsize=(10, 6))
ax.plot(list(n_range), asymp_sqrt, linestyle='--',color ='green', linewidth=1.5)
ax.plot(list(n_range), asymp_exp, linestyle='--', color='red', linewidth=1.5)
ax.plot(list(n_range), baseline_vals, linestyle='--', color='blue', linewidth=1.5)
ax.plot(n_filtered, sn_vals_filtered, label='Replacer fixation probability', linestyle='None', marker='o', markersize=6, color='green')
ax.plot(n_filtered, ns_vals_filtered, label='Replacer elimination probability', linestyle='None', marker='o', markersize=6, color='red')
ax.plot(n_filtered, one_over_n_filtered, label='Oblivious baseline', linestyle='None', marker='o', markersize=6, color='blue')

ax.set_xlabel('Population size, $N$')
ax.set_ylabel('Probability')

ax.legend()
ax.grid()
save(f, "fig_neutral_fix_elim.pdf")


plt.show()