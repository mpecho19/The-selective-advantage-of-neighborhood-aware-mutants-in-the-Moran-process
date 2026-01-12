

from better_plots.better_plots import set_defaults, use_science, fig, save


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

def sn_formula_cycle(n, r):
    numerator = 1 - (1 / (2 * r))
    denominator = 1 - (2 / ((2 * r) ** n)) + (1 / (2 * r))
    return numerator / denominator


def baseline(n,r):
    if abs(r - 1) < 1e-10:
        return 1/n
    else:
        return (1 - (1/r)) / (1 - (1/(r**n)))


def binary_find_complete(n, r, error):
    min_s = 0 
    max_s = 10
    s = 0.5

    fix_prob = sn_formula_complete(n, 1, s)

    while abs(fix_prob - baseline(n, r)) > error:
        if fix_prob < baseline(n, r):
            min_s = s
        else:
            max_s = s
        s = (min_s + max_s) / 2

        fix_prob = sn_formula_complete(n, 1, s)
    return s




n1 = 3
n2 = 101
range_n = range(n1,n2)

r_values = [1]

error= 10**(-15)

list_of_s = []

for r in r_values:
    values_s = []
    for n in range_n:
        values_s.append( r - binary_find_complete(n, r, error))
    print("done")
    list_of_s.append(values_s)


use_science("science", "ieee")

f, ax = fig(font= 21.5, aspect=.8)


for r, i in enumerate(r_values):
    ax.plot(range_n, list_of_s[r], linestyle='-', label=f'$r = {r_values[r]}$')
ax.set_xlabel('Population size, $N$')
import matplotlib.ticker as mticker
ax.set_ylabel(r'Cost to match $\rho$')
ax.set_ylim(0.155, 0.285)
ticks = [round(0.16 + i * 0.03, 2) for i in range(5)]  
ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
# ax.set_title('The cost of sensing on complete vs cycle graphs')
# plt.xscale('log')
# plt.legend(loc='best', frameon=True, shadow=True)
ax.grid(True, linestyle='--', linewidth=0.7, color='gray', alpha=0.7)

save(f, "fig_equality_1.pdf")