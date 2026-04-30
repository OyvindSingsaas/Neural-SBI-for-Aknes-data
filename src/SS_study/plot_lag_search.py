import matplotlib.pyplot as plt
import numpy as np


def load_version(path):
    d = np.load(path, allow_pickle=True)
    return {
        "r_min_values":       d["r_min_values"],
        "r_max_values":       d["r_max_values"],
        "nbins_values":       d["nbins_values"],
        "rmse_ncl_r_min":     d["rmse_list_ncl_r_min"],
        "se_ncl_r_min":       d["se_list_ncl_r_min"],
        "rmse_nf_r_min":      d["rmse_list_nf_r_min"],
        "se_nf_r_min":        d["se_list_nf_r_min"],
        "rmse_ncl_r_max":     d["rmse_list_ncl_r_max"],
        "se_ncl_r_max":       d["se_list_ncl_r_max"],
        "rmse_nf_r_max":      d["rmse_list_nf_r_max"],
        "se_nf_r_max":        d["se_list_nf_r_max"],
        "rmse_ncl_nbins":     d["rmse_list_ncl_nbins"],
        "se_ncl_nbins":       d["se_list_ncl_nbins"],
        "rmse_nf_nbins":      d["rmse_list_nf_nbins"],
        "se_nf_nbins":        d["se_list_nf_nbins"],
    }


def plot_row(axes, data, row_label, xscales):
    searches = [
        (data["r_min_values"], data["rmse_ncl_r_min"], data["se_ncl_r_min"],
         data["rmse_nf_r_min"], data["se_nf_r_min"], r"$r_\mathrm{min}$"),
        (data["r_max_values"], data["rmse_ncl_r_max"], data["se_ncl_r_max"],
         data["rmse_nf_r_max"], data["se_nf_r_max"], r"$r_\mathrm{max}$"),
        (data["nbins_values"], data["rmse_ncl_nbins"], data["se_ncl_nbins"],
         data["rmse_nf_nbins"], data["se_nf_nbins"], r"$N_\mathrm{bins}$"),
    ]
    for ax, (x_vals, rmse_ncl, se_ncl, rmse_nf, se_nf, xlabel), xscale in zip(axes, searches, xscales):

        best_ncl = int(np.argmin(rmse_ncl))
        best_nf  = int(np.argmin(rmse_nf))

        ax.plot(x_vals, rmse_ncl, marker='o', linewidth=1.5, color='steelblue', label='NCL')
        ax.fill_between(x_vals, rmse_ncl - se_ncl, rmse_ncl + se_ncl, color='steelblue', alpha=0.2)
        ax.scatter([x_vals[best_ncl]], [rmse_ncl[best_ncl]], color='steelblue', zorder=5)
        ax.axvline(x_vals[best_ncl], color='steelblue', linestyle='--', linewidth=1.0)

        ax.plot(x_vals, rmse_nf, marker='s', linewidth=1.5, color='darkorange', label='NF')
        ax.fill_between(x_vals, rmse_nf - se_nf, rmse_nf + se_nf, color='darkorange', alpha=0.2)
        ax.scatter([x_vals[best_nf]], [rmse_nf[best_nf]], color='darkorange', zorder=5)
        ax.axvline(x_vals[best_nf], color='darkorange', linestyle='--', linewidth=1.0)

        ax.set_xscale(xscale)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("RMSE", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[0].set_title(r"$r_\mathrm{min}$ search — " + row_label)
    axes[1].set_title(r"$r_\mathrm{max}$ search — " + row_label)
    axes[2].set_title(r"$N_\mathrm{bins}$ search — " + row_label)


#v1 = load_version("results/lag_search/lagsearch_v1.npz")
#v2 = load_version("results/lag_search/lagsearch_v2.npz")
only_nbins = load_version("results/lag_search/lagsearch_only_nbins.npz")
abc_nbins = np.load("results/lag_search/abc_lagsearch_nbins.npz", allow_pickle=True)

# fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=False)
# xscales = ["log", "linear", "linear"]
# plot_row(axes[0], v1, "round 1", xscales)
# plot_row(axes[1], v2, "round 2", xscales)

# fig.suptitle("Lag search: RMSE vs. cluster bin parameters", fontsize=13)
# fig.tight_layout()
# plt.savefig("results/lag_search/lag_search_looped.png", bbox_inches="tight")
# plt.close()

fig  = plt.figure(figsize=(7, 5))
plt.plot(only_nbins["nbins_values"], only_nbins["rmse_ncl_nbins"], marker='o', linewidth=1.5, color='steelblue', label='NCL')
plt.fill_between(only_nbins["nbins_values"], only_nbins["rmse_ncl_nbins"] - only_nbins["se_ncl_nbins"], only_nbins["rmse_ncl_nbins"] + only_nbins["se_ncl_nbins"], color='steelblue', alpha=0.2)
best_ncl = int(np.argmin(only_nbins["rmse_ncl_nbins"]))
plt.scatter([only_nbins["nbins_values"][best_ncl]], [only_nbins["rmse_ncl_nbins"][best_ncl]], color='steelblue', zorder=5)
plt.axvline(x=only_nbins["nbins_values"][best_ncl], color='steelblue', linestyle='--', linewidth=1.0)

plt.plot(only_nbins["nbins_values"], only_nbins["rmse_nf_nbins"], marker='s', linewidth=1.5, color='darkorange', label='NF')
plt.fill_between(only_nbins["nbins_values"], only_nbins["rmse_nf_nbins"] - only_nbins["se_nf_nbins"], only_nbins["rmse_nf_nbins"] + only_nbins["se_nf_nbins"], color='darkorange', alpha=0.2)
best_nf = int(np.argmin(only_nbins["rmse_nf_nbins"]))
plt.scatter([only_nbins["nbins_values"][best_nf]], [only_nbins["rmse_nf_nbins"][best_nf]], color='darkorange', zorder=5)
plt.axvline(x=only_nbins["nbins_values"][best_nf], color='darkorange', linestyle='--', linewidth=1.0)

plt.plot(abc_nbins["nbins_values"], abc_nbins["rmse_list"], marker='X', color='green', label='ABC')
plt.fill_between(abc_nbins["nbins_values"], abc_nbins["rmse_list"] - abc_nbins["se_list"], abc_nbins["rmse_list"] + abc_nbins["se_list"], color='green', alpha=0.2)
best_abc = int(np.argmin(abc_nbins["rmse_list"]))
plt.scatter([abc_nbins["nbins_values"][best_abc]], [abc_nbins["rmse_list"][best_abc]], color='green', zorder=5)
plt.axvline(x=abc_nbins["nbins_values"][best_abc], color='green', linestyle='--', linewidth=1.0)
plt.xlabel(r"$N_\mathrm{bins}$", fontsize=11)
plt.ylabel("RMSE", fontsize=11)
#plt.title(r"$N_\mathrm{bins}$ search — only $N_\mathrm{bins}$ varied", fontsize=13)
plt.legend(fontsize=14)
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.0)
plt.savefig("results/lag_search/lag_search_only_nbins_2.png", bbox_inches="tight")
plt.close()
