import matplotlib.pyplot as plt
import numpy as np


ncl_data = np.load("results/lag_search/lagsearch_1.npz", allow_pickle=True)

rmse_ncl_r_min = ncl_data["rmse_list_ncl_r_min"]
rmse_ncl_r_max = ncl_data["rmse_list_ncl_r_max"]
rmse_ncl_nbins = ncl_data["rmse_list_ncl_nbins"]

rmse_nf_r_min = ncl_data["rmse_list_nf_r_min"]
rmse_nf_r_max = ncl_data["rmse_list_nf_r_max"]
rmse_nf_nbins = ncl_data["rmse_list_nf_nbins"]

r_min_values = ncl_data["r_min_values"]
r_max_values = ncl_data["r_max_values"]
nbins_values = ncl_data["nbins_values"]

searches = [
    (r_min_values, rmse_ncl_r_min, rmse_nf_r_min, r"$r_\mathrm{min}$"),
    (r_max_values, rmse_ncl_r_max, rmse_nf_r_max, r"$r_\mathrm{max}$"),
    (nbins_values, rmse_ncl_nbins, rmse_nf_nbins,  r"$N_\mathrm{bins}$"),
]

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
for i, (ax, (x_vals, rmse_ncl, rmse_nf, xlabel)) in enumerate(zip(axes, searches)):
    best_ncl = int(np.argmin(rmse_ncl))
    best_nf  = int(np.argmin(rmse_nf))

    ax.plot(x_vals, rmse_ncl, marker='o', linewidth=1.5, color='steelblue', label='NCL')
    ax.scatter([x_vals[best_ncl]], [rmse_ncl[best_ncl]], color='steelblue', zorder=5)
    ax.axvline(x_vals[best_ncl], color='steelblue', linestyle='--', linewidth=1.0)

    ax.plot(x_vals, rmse_nf, marker='s', linewidth=1.5, color='darkorange', label='NF')
    ax.scatter([x_vals[best_nf]], [rmse_nf[best_nf]], color='darkorange', zorder=5)
    ax.axvline(x_vals[best_nf], color='darkorange', linestyle='--', linewidth=1.0)

    if i ==0:
        ax.set_xscale('log')
    else:
        ax.set_xscale('linear')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

axes[0].set_title(r"Effect of $r_\mathrm{min}$")
axes[1].set_title(r"Effect of $r_\mathrm{max}$")
axes[2].set_title(r"Effect of $N_\mathrm{bins}$")

fig.suptitle("Lag search: RMSE vs. cluster bin parameters", fontsize=13, y=1.02)
fig.tight_layout()
plt.savefig("results/lag_search/lag_search.png", bbox_inches="tight")
plt.close()


#Plot only nbins 

data = np.load("results/lag_search/lagsearch_nbins.npz", allow_pickle=True)
rmse_ncl_nbins = data["rmse_list_ncl_nbins"]
rmse_nf_nbins = data["rmse_list_nf_nbins"]
nbins_values = data["nbins_values"]

data_linear = np.load("results/lag_search/lagsearch_nbins_linear.npz", allow_pickle=True)
rmse_ncl_nbins_linear = data_linear["rmse_list_ncl_nbins"]
rmse_nf_nbins_linear = data_linear["rmse_list_nf_nbins"]
nbins_values_linear = data_linear["nbins_values"]

plt.figure(figsize=(6,4))
best_ncl_linear = int(np.argmin(rmse_ncl_nbins_linear))
best_nf_linear  = int(np.argmin(rmse_nf_nbins_linear))
best_ncl = int(np.argmin(rmse_ncl_nbins))
best_nf  = int(np.argmin(rmse_nf_nbins))

# Linear-spaced: solid lines, filled markers
plt.plot(nbins_values_linear, rmse_ncl_nbins_linear, marker='o', linewidth=1.5, color='steelblue', linestyle='-', label='NCL (linear)')
plt.plot(nbins_values_linear, rmse_nf_nbins_linear, marker='s', linewidth=1.5, color='darkorange', linestyle='-', label='NF (linear)')
plt.scatter([nbins_values_linear[best_ncl_linear]], [rmse_ncl_nbins_linear[best_ncl_linear]], color='steelblue', zorder=5)
plt.scatter([nbins_values_linear[best_nf_linear]], [rmse_nf_nbins_linear[best_nf_linear]], color='darkorange', zorder=5)
plt.axvline(nbins_values_linear[best_ncl_linear], color='steelblue', linestyle='--', linewidth=1.0)
plt.axvline(nbins_values_linear[best_nf_linear], color='darkorange', linestyle='--', linewidth=1.0)

# Log-spaced: dashed lines, open markers, lighter alpha
plt.plot(nbins_values, rmse_ncl_nbins, marker='o', linewidth=1.5, color='steelblue', linestyle=':', alpha=0.6,
         markerfacecolor='none', label='NCL (log)')
plt.plot(nbins_values, rmse_nf_nbins, marker='s', linewidth=1.5, color='darkorange', linestyle=':', alpha=0.6,
         markerfacecolor='none', label='NF (log)')
plt.scatter([nbins_values[best_ncl]], [rmse_ncl_nbins[best_ncl]], color='steelblue', zorder=5, facecolors='none', edgecolors='steelblue', linewidths=1.5)
plt.scatter([nbins_values[best_nf]], [rmse_nf_nbins[best_nf]], color='darkorange', zorder=5, facecolors='none', edgecolors='darkorange', linewidths=1.5)
plt.axvline(nbins_values[best_ncl], color='steelblue', linestyle=':', linewidth=1.0)
plt.axvline(nbins_values[best_nf], color='darkorange', linestyle=':', linewidth=1.0)
plt.xlabel(r"$N_\mathrm{bins}$", fontsize=12)
plt.ylabel("RMSE", fontsize=12)
plt.title(r"Effect of $N_\mathrm{bins}$", fontsize=13)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.savefig("results/lag_search/lag_search_nbins.png", bbox_inches="tight")
plt.close()