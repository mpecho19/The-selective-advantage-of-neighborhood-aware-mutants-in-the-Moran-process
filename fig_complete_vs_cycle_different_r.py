import numpy as np
from libs.SaveSystem import SaveSystem
import matplotlib.pyplot as plt 
import numpy as np

def sn_formula_cycle(n,i, r):
    numerator = 1 - (2 / ((2 * r) ** i)) + (1 / (2 * r))
    denominator = 1 - (2 / ((2 * r) ** n)) + (1 / (2 * r))
    return numerator / denominator

def sn_formula_complete(n,i, r):
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
def baseline(n,r):
    if r <= 1:
        return 0
    return (1 - (1/r)) 


ss_load = False

r_values = np.arange(0, 5.01, 0.01)
n1 = 10
n2 = 100
ss = SaveSystem(f"complete_cycle_different_r", load=ss_load)
ss.auto_init_lists(True)
if ss_load == False:
    for r in r_values:
        ss["r"].append(r)
        ss[f'complete_{n1}'].append(sn_formula_complete(n1, 1, r))
        ss[f'complete_{n2}'].append(sn_formula_complete(n2, 1, r))
        if r == 0.5:
            ss[f'cycle_{n1}'].append(1/(1 + 2*(n1-1)))
            ss[f'cycle_{n2}'].append(1/(1 + 2*(n2-1)))
        else:
            ss[f'cycle_{n1}'].append(sn_formula_cycle(n1, 1, r))
            ss[f'cycle_{n2}'].append(sn_formula_cycle(n2, 1, r))

        ss['baseline'].append(baseline(n1, r))
from better_plots.better_plots import set_defaults, use_science, fig, save
use_science("science", "ieee")

f, ax = fig(font=12, aspect=.8)

ax.plot(ss["r"], ss["baseline"], label=r"Oblivious baseline, $N \to \infty$", linestyle='--', color='black')
ax.plot(ss["r"], ss[f"complete_{n1}"], label=f"Well-mixed, $N={n1}$", color='orange')
ax.plot(ss["r"], ss[f"complete_{n2}"], label=f"Well-mixed, $N={n2}$", color='red')
ax.plot(ss["r"], ss[f"cycle_{n1}"], label=f"Cycle, $N={n1}$", color='cyan')
ax.plot(ss["r"], ss[f"cycle_{n2}"], label=f"Cycle, $N={n2}$", color='blue')
ax.set_xlabel("Fitness, $r$")
ax.set_ylabel(r"Fixation probability, $\rho$")
# plt.axhline(1/n, color='black', linewidth=0.5, linestyle='--')
# ax.set_title(f"Complete vs Cycle")
ax.legend()
ax.grid()

save(f, "fig_complete_vs_cycle_different_r.pdf")

