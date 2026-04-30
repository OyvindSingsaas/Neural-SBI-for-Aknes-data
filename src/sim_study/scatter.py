import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
from tensorflow.keras.models import load_model
import utils.utils_surface_NS as us
from scipy.stats import norm

save_scatter_path = "results/sim_study/scatterplot_dedicated_PM_2.png"
data_path = 'data/NS_data_temp_wp_10RK.npz'

nf_results_path = "results/sim_study/nf.npz"
ncl_results_path = "results/sim_study/ncl.npz"
abc_results_path = "results/sim_study/abc_samples_bank.npz"

print("Loading data...")
data = np.load(data_path, allow_pickle=True)
params_train_normalized = data['params_train_normalized']
SS_0_train_normalized_neural = data['SS_0_train_normalized_neural']
response_train = data['response_train']
params_test_normalized = data['params_test_normalized']
SS_0_test_normalized_neural = data['SS_0_test_normalized_neural']
response_test = data['response_test']
params_mean = data['params_mean']
params_std = data['params_std']
SS_mean = data['SS_mean']
SS_std = data['SS_std']
l_bounds_NS = data['l_bounds_NS']
u_bounds_NS = data['u_bounds_NS']
l_bounds_NS_test = data['l_bounds_NS_test']
u_bounds_NS_test = data['u_bounds_NS_test']
T = data['T']
cluster_bins = data['cluster_bins']
percentiles = data['percentiles']
col_names_params_NS = data['col_names_params_NS']

df_metro = pd.DataFrame(data['df_metro_values'],
                    columns=data['df_metro_columns'],
                    index=data['df_metro_index'])
print("Data loaded successfully.")
years = [2023]
gs = len(SS_0_train_normalized_neural[0])  # Dimension of summary statistics
p = len(l_bounds_NS)  # Number of parameters
print("Dimension of summary statistics:", gs)
print("Number of training data points:", len(params_train_normalized))


plt.rcParams.update({
    "font.size": 16,        # default text
    "axes.titlesize": 18,   # title
    "axes.labelsize": 18,   # x and y labels
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 14
})
print("\nLoading normalizing flow results...")
nf_results = np.load(nf_results_path, allow_pickle=True)
posterior_samples = nf_results["posterior_samples"]
posterior_samples_normalized = nf_results["posterior_samples_normalized"]
nf_map = nf_results["nf_map"]
nf_map_normalized = nf_results["nf_map_normalized"]
N_train_nf = nf_results["N_train"]
true_nf = nf_results["true"]
true_nf_normalized = nf_results["true_normalized"]
print("Normalizing flow results loaded successfully.")

print("\nLoading NCL results...")
ncl_results = np.load(ncl_results_path, allow_pickle=True)
ncl_mle = ncl_results["ncl_mle"]
ncl_mle_normalized = ncl_results["ncl_mle_normalized"]
N_train_ncl = ncl_results["N_train"]
true_ncl = ncl_results["true"]
true_ncl_normalized = ncl_results["true_normalized"]
G_inv_NS_ray = ncl_results["G_inv_NS_ray"]
H_neg_NS_ray = ncl_results["H_neg_NS_ray"]
not_converged_index = ncl_results["not_converged_index"]
print("NCL results loaded successfully.")
print("Number of non-converged optimizations:", len(not_converged_index), "at indices:", not_converged_index)
N_train_ncl_converged = N_train_ncl - len(not_converged_index)
converged_ncl = [i for i in range(N_train_ncl) if i not in not_converged_index]

#Load ABC results
abc_results = np.load(abc_results_path, allow_pickle=True)
abc_samples = abc_results["abc_samples"]
abc_samples_normalized = abc_results["abc_samples_normalized"]
abc_map = abc_results["abc_map"]
abc_map_normalized = abc_results["abc_map_normalized"]
N_train_abc = abc_results["N_train"]
true_abc = abc_results["true"]
true_abc_normalized = abc_results["true_normalized"]
print("ABC results loaded successfully.")

n_sub = 100
rng = np.random.default_rng(seed=42)
idx_nf  = rng.choice(len(true_nf),     size=min(n_sub, len(true_nf)),          replace=False)
idx_ncl = rng.choice(converged_ncl,    size=min(n_sub, len(converged_ncl)),     replace=False)
idx_abc = rng.choice(len(true_abc),    size=min(n_sub, len(true_abc)),          replace=False)

# Improved scatter plot: 2x3 grid, only 5 parameters, last subplot removed
num_params = 5
fig, axs = plt.subplots(2, 3, figsize=(12, 7))
axs = axs.ravel()
for i in range(num_params):
    ax = axs[i]
    ax.scatter(nf_map[idx_nf, i], true_nf[idx_nf, i], label="NF MAP", alpha=0.3)
    ax.scatter(ncl_mle[idx_ncl, i], true_ncl[idx_ncl, i], label="NCL MLE", alpha=0.3)
    ax.scatter(abc_map[idx_abc, i], true_abc[idx_abc, i], label="ABC PM", alpha=0.3)
    ax.set_title(col_names_params_NS[i])

    lo = min(l_bounds_NS_test[i],
             nf_map[:, i].min(), ncl_mle[converged_ncl, i].min(), abc_map[:, i].min())
    hi = max(u_bounds_NS_test[i],
             nf_map[:, i].max(), ncl_mle[converged_ncl, i].max(), abc_map[:, i].max())
    ax.set_xlim([lo, hi])
    ax.set_ylim([lo, hi])
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.7)
    if i % 3 == 0:
        ax.set_ylabel("True")
    else:
        ax.set_ylabel("")
    if i >= 3:
        ax.set_xlabel("Predicted")
    else:
        ax.set_xlabel("")

#manually add legend in empty subplot
keys, labels = ax.get_legend_handles_labels()
fig.legend(keys, labels, ncol=1, bbox_to_anchor=(0.83, 0.48))
# Remove the final unused subplot
for j in range(num_params, 6):
    fig.delaxes(axs[j])
fig.tight_layout()
fig.savefig(save_scatter_path)
plt.close(fig)