import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
from tensorflow.keras.models import load_model
import utils.utils_surface_NS as us
from scipy.stats import norm

col_names_params_NS = [
    r"$\log(\delta)$",
    r"$\log(\sigma^2)$",
    r"$\beta_0$",
    r"$\beta_1$",
    r"$\beta_2$",
    r"$\beta_3$",
    r"$\beta_4$",
    r"$\beta_5$",
    r"$\beta_6$"
]


def compute_performance_stats(true_normalized, pred_normalized, params_std, params_mean):
    """
    Compute MSE, MAE, bias (normalized and denormalized) and standard error of MSE.
    """
    true_denorm = true_normalized * params_std + params_mean
    pred_denorm = pred_normalized * params_std + params_mean
    rmse_normalized = np.sqrt(mean_squared_error(true_normalized, pred_normalized))
    rmse = np.sqrt(mean_squared_error(true_denorm, pred_denorm))
    rmse_error = np.sqrt(rmse / len(true_normalized))
    mae = mean_absolute_error(true_normalized, pred_normalized)
    mae_denorm = mean_absolute_error(true_denorm, pred_denorm)
    bias_normalized = np.mean(pred_normalized - true_normalized, axis=0)
    bias = np.mean(pred_denorm - true_denorm, axis=0)

    # Parameter-wise RMSE
    rmse_normalized_paramwise = np.sqrt(np.mean((true_normalized - pred_normalized) ** 2, axis=0))
    rmse_paramwise = np.sqrt(np.mean((true_denorm - pred_denorm) ** 2, axis=0))
    # standard error of parameter-wise RMSE
    rmse_paramwise_error = np.sqrt(rmse_paramwise / len(true_normalized))

    return {
        "rmse_normalized": rmse_normalized,
        "rmse": rmse,
        "rmse_error": rmse_error,
        "mae": mae,
        "mae_denormalized": mae_denorm,
        "bias_normalized": bias_normalized.tolist(),
        "bias": bias.tolist(),
        "rmse_normalized_paramwise": rmse_normalized_paramwise.tolist(),
        "rmse_paramwise": rmse_paramwise.tolist(),
        "rmse_paramwise_error": rmse_paramwise_error.tolist()
    }

print("Loading data...")
data = np.load('data/NS_data_temp_wp_10RK.npz', allow_pickle=True)
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

#Set significance level for confidence intervals
alpha = 10

print("\nLoading normalizing flow results...")
nf_results = np.load("results/sim_study/nf_current.npz", allow_pickle=True)
posterior_samples = nf_results["posterior_samples"]
posterior_samples_normalized = nf_results["posterior_samples_normalized"]
nf_map = nf_results["nf_map"]
nf_map_normalized = nf_results["nf_map_normalized"]
N_train_nf = nf_results["N_train"]
true_nf = nf_results["true"]
true_nf_normalized = nf_results["true_normalized"]
print("Normalizing flow results loaded successfully.")

#Compute empirical coverage and
print("\nComputing empirical coverage and confidence intervals for Normalizing Flow...")
nf_coverage = np.zeros((len(l_bounds_NS),))
nf_interval_lengths = np.zeros((N_train_nf, len(l_bounds_NS)))
for n in range(N_train_nf):
    for param in range(len(l_bounds_NS)):
        lower_bound = np.percentile(posterior_samples_normalized[n,:,param], alpha/2)
        upper_bound = np.percentile(posterior_samples_normalized[n,:,param], 100 - alpha/2)
        upper_bound_denorm = upper_bound * params_std[param] + params_mean[param]
        lower_bound_denorm = lower_bound * params_std[param] + params_mean[param]
        coverage = (true_nf_normalized[n, param] >= lower_bound) and (true_nf_normalized[n, param] <= upper_bound)
        nf_coverage[param] += coverage
        nf_interval_lengths[n, param] = upper_bound_denorm - lower_bound_denorm
nf_coverage /= N_train_nf
nf_interval_lengths_mean = np.mean(nf_interval_lengths, axis=0)
nf_interval_lengths_std = np.std(nf_interval_lengths, axis=0)
print("Empirical coverage for Normalizing Flow (alpha={}):".format(alpha), nf_coverage)
print("Average interval lengths for Normalizing Flow (alpha={}):".format(alpha), nf_interval_lengths_mean, "±", nf_interval_lengths_std)

print("\nLoading NCL results...")
ncl_results = np.load("results/sim_study/ncl_current.npz", allow_pickle=True)
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

#Load platt scaling factor for NCL confidence intervals
platt_scaler = np.load("neural_networks/platt_scaler_NS.npy", allow_pickle=True)
print("Platt scaling factor for NCL confidence intervals loaded successfully:", platt_scaler)

print("\nComputing empirical coverage and confidence intervals for NCL...")
N_G = 70
ncl_coverage = np.zeros((len(l_bounds_NS),))
ncl_interval_lengths = np.zeros((N_train_ncl_converged, len(l_bounds_NS)))

ncl_coverage_platt = np.zeros((len(l_bounds_NS),))
ncl_interval_lengths_platt = np.zeros((N_train_ncl_converged, len(l_bounds_NS)))

ncl_coverage_G = np.zeros((len(l_bounds_NS),))
ncl_interval_lengths_G = np.zeros((N_train_ncl_converged, len(l_bounds_NS)))

z = norm.ppf(1 - alpha/200)  # z-score for two-tailed test
j = 0
for n in range(N_train_ncl):
    if n in not_converged_index:
        pass
    else:
        for param in range(len(l_bounds_NS)):
            std = np.sqrt(np.linalg.inv(H_neg_NS_ray[j])[param][param])
            lower = ncl_mle_normalized[n, param] - z * std
            upper = ncl_mle_normalized[n, param] + z * std
            lower_denorm = lower * params_std[param] + params_mean[param]
            upper_denorm = upper * params_std[param] + params_mean[param]
            coverage_G = (true_ncl_normalized[n, param] >= lower) and (true_ncl_normalized[n, param] <= upper)
            ncl_coverage[param] += coverage_G
            ncl_interval_lengths[j, param] = upper_denorm - lower_denorm

            std_platt = np.sqrt(np.linalg.inv(H_neg_NS_ray[j]*platt_scaler)[param][param])
            lower_platt = ncl_mle_normalized[n, param] - z * std_platt
            upper_platt = ncl_mle_normalized[n, param] + z * std_platt
            lower_platt_denorm = lower_platt * params_std[param] + params_mean[param]
            upper_platt_denorm = upper_platt * params_std[param] + params_mean[param]
            coverage_platt = (true_ncl_normalized[n, param] >= lower_platt) and (true_ncl_normalized[n, param] <= upper_platt)
            ncl_coverage_platt[param] += coverage_platt
            ncl_interval_lengths_platt[j, param] = upper_platt_denorm - lower_platt_denorm

            std_G = np.sqrt(G_inv_NS_ray[j][param][param])
            lower_G = ncl_mle_normalized[n, param] - z * std_G
            upper_G = ncl_mle_normalized[n, param] + z * std_G
            lower_G_denorm = lower_G * params_std[param] + params_mean[param]
            upper_G_denorm = upper_G * params_std[param] + params_mean[param]
            coverage_G = (true_ncl_normalized[n, param] >= lower_G) and (true_ncl_normalized[n, param] <= upper_G)
            ncl_coverage_G[param] += coverage_G
            ncl_interval_lengths_G[j, param] = upper_G_denorm - lower_G_denorm
        j+=1
ncl_coverage /= N_train_ncl_converged
ncl_coverage_platt /= N_train_ncl_converged
ncl_coverage_G /= N_train_ncl_converged
ncl_interval_lengths_mean = np.mean(ncl_interval_lengths, axis=0)
ncl_interval_lengths_platt_mean = np.mean(ncl_interval_lengths_platt, axis=0)
ncl_interval_lengths_G_mean = np.mean(ncl_interval_lengths_G, axis=0)
ncl_interval_lengths_std = np.std(ncl_interval_lengths, axis=0)
ncl_interval_lengths_platt_std = np.std(ncl_interval_lengths_platt, axis=0)
ncl_interval_lengths_G_std = np.std(ncl_interval_lengths_G, axis=0)
print("Empirical coverage for NCL (G-based, alpha={}):".format(alpha), ncl_coverage_G)
print("Average interval lengths for NCL (G-based, alpha={}):".format(alpha), ncl_interval_lengths_G_mean, "±", ncl_interval_lengths_G_std)
print("Empirical coverage for NCL (Platt-scaled, alpha={}):".format(alpha), ncl_coverage_platt)
print("Average interval lengths for NCL (Platt-scaled, alpha={}):".format(alpha), ncl_interval_lengths_platt_mean, "±", ncl_interval_lengths_platt_std)
print("Empirical coverage for NCL (Hessian-based, alpha={}):".format(alpha), ncl_coverage)
print("Average interval lengths for NCL (Hessian-based, alpha={}):".format(alpha), ncl_interval_lengths_mean, "±", ncl_interval_lengths_std)

#Load ABC results
abc_results = np.load("results/sim_study/abc_samples.npz", allow_pickle=True)
abc_samples = abc_results["abc_samples"]
abc_samples_normalized = abc_results["abc_samples_normalized"]
abc_map = abc_results["abc_map"]
abc_map_normalized = abc_results["abc_map_normalized"]
N_train_abc = abc_results["N_train"]
true_abc = abc_results["true"]
true_abc_normalized = abc_results["true_normalized"]
print("ABC results loaded successfully.")

abc_coverage = np.zeros((len(l_bounds_NS),))
abc_interval_lengths = np.zeros((N_train_abc, len(l_bounds_NS)))
for n in range(N_train_abc):
    for param in range(len(l_bounds_NS)):
        lower_bound = np.percentile(abc_samples_normalized[n,:,param], alpha/2)
        upper_bound = np.percentile(abc_samples_normalized[n,:,param], 100 - alpha/2)
        coverage = (true_abc_normalized[n, param] >= lower_bound) and (true_abc_normalized[n, param] <= upper_bound)
        abc_coverage[param] += coverage
        abc_interval_lengths[n, param] = upper_bound - lower_bound
abc_coverage /= N_train_abc
abc_interval_lengths_mean = np.mean(abc_interval_lengths, axis=0)
abc_interval_lengths_std = np.std(abc_interval_lengths, axis=0)
print("Empirical coverage for ABC (alpha={}):".format(alpha), abc_coverage)
print("Average interval lengths for ABC (alpha={}):".format(alpha), abc_interval_lengths_mean, "±", abc_interval_lengths_std)   

#Compute performance statistics for ABC
print("\nComputing performance statistics for ABC...")
abc_stats = compute_performance_stats(true_abc_normalized, abc_map_normalized, params_std, params_mean)

#Compute performance statistics for NF
print("\nComputing performance statistics for Normalizing Flow...")
nf_stats = compute_performance_stats(true_nf_normalized, nf_map_normalized, params_std, params_mean)

#compute performance statistics for NCL
converged_ncl = [i for i in range(N_train_ncl) if i not in not_converged_index]

print("\nComputing performance statistics for NCL...")
ncl_stats = compute_performance_stats(true_ncl_normalized[converged_ncl], ncl_mle_normalized[converged_ncl], params_std, params_mean)

plt.rcParams.update({
    "font.size": 16,        # default text
    "axes.titlesize": 18,   # title
    "axes.labelsize": 18,   # x and y labels
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 14
})

# Improved scatter plot: 2x3 grid, only 5 parameters, last subplot removed
num_params = 5
fig, axs = plt.subplots(2, 3, figsize=(12, 7))
axs = axs.ravel()
for i in range(num_params):
    ax = axs[i]
    ax.scatter(true_nf[:, i], nf_map[:, i], label="NF MAP", alpha=0.5)
    #Dont plot non-converged points for NCL
    ax.scatter(true_ncl[converged_ncl, i], ncl_mle[converged_ncl, i], label="NCL MLE", alpha=0.5)
    ax.scatter(true_abc[:, i], abc_map[:, i], label="ABC MAP", alpha=0.5)
    ax.set_title(col_names_params_NS[i])
    axis_max = max(ax.get_ylim()[1], ax.get_xlim()[1])
    axis_min = min(ax.get_ylim()[0], ax.get_xlim()[0])
    ax.set_xlim([axis_min, axis_max])
    ax.set_ylim([axis_min, axis_max])
    ax.plot([axis_min, axis_max], [axis_min, axis_max], 'k--', alpha=0.7)
    if i % 3 == 0:
        ax.set_ylabel("Predicted"
                      )
    else:
        ax.set_ylabel("")
    if i >= 3:
        ax.set_xlabel("True")
    else:
        ax.set_xlabel("")

#manually add legend in empty subplot
keys, labels = ax.get_legend_handles_labels()
fig.legend(keys, labels, ncol=1, bbox_to_anchor=(0.83, 0.48))
# Remove the final unused subplot
for j in range(num_params, 6):
    fig.delaxes(axs[j])
fig.tight_layout()
fig.savefig("results/sim_study/scatterplot.png")
plt.close(fig)

#Save performance statistics to json file for both methods and coverage and interval lengths
print("\nSaving performance statistics to json file...")
with open("results/sim_study/performance_stats.json", "w") as f:
    json.dump({
        "nf_stats": nf_stats,
        "ncl_stats": ncl_stats,
        "abc_stats": abc_stats,
        "nf_coverage": nf_coverage.tolist(),
        "nf_interval_lengths_mean": nf_interval_lengths_mean.tolist(),
        "nf_interval_lengths_std": nf_interval_lengths_std.tolist(),
        "ncl_coverage_G": ncl_coverage_G.tolist(),
        "ncl_interval_lengths_G_mean": ncl_interval_lengths_G_mean.tolist(),
        "ncl_interval_lengths_G_std": ncl_interval_lengths_G_std.tolist(),
        "ncl_coverage_platt": ncl_coverage_platt.tolist(),
        "ncl_interval_lengths_platt_mean": ncl_interval_lengths_platt_mean.tolist(),
        "ncl_interval_lengths_platt_std": ncl_interval_lengths_platt_std.tolist(),
        "ncl_coverage_hessian": ncl_coverage.tolist(),
        "ncl_interval_lengths_hessian_mean": ncl_interval_lengths_mean.tolist(),
        "ncl_interval_lengths_hessian_std": ncl_interval_lengths_std.tolist(),
        "abc_coverage": abc_coverage.tolist(),
        "abc_interval_lengths_mean": abc_interval_lengths_mean.tolist(),
        "abc_interval_lengths_std": abc_interval_lengths_std.tolist(),
    }, f, indent=4)

event_count = SS_0_test_normalized_neural[response_test==1]
event_count = event_count[:,0]*SS_std[0] + SS_mean[0]
event_count = event_count[:N_train_nf]
event_count = np.exp(event_count)  # Inverse of log1p to get back to original scale
#plot mse vs event count for all methods

plt.figure(figsize=(8, 6))
plt.scatter(event_count, np.sqrt(np.mean((true_nf - nf_map)**2, axis=1)), label="NF", alpha=0.5)
plt.scatter(event_count[converged_ncl], np.sqrt(np.mean((true_ncl[converged_ncl] - ncl_mle[converged_ncl])**2, axis=1)), label="NCL", alpha=0.5)
plt.scatter(event_count, np.sqrt(np.mean((true_abc - abc_map)**2, axis=1)), label="ABC", alpha=0.5)
#add regression line for each method
z_nf = np.polyfit(event_count, np.sqrt(np.mean((true_nf - nf_map)**2, axis=1)), 1)
p_nf = np.poly1d(z_nf)
plt.plot(event_count, p_nf(event_count), "r--", label="NF regression", alpha=0.7)
z_ncl = np.polyfit(event_count[converged_ncl], np.sqrt(np.mean((true_ncl[converged_ncl] - ncl_mle[converged_ncl])**2, axis=1)), 1)
p_ncl = np.poly1d(z_ncl)
plt.plot(event_count[converged_ncl], p_ncl(event_count[converged_ncl]), "b--", label="NCL regression", alpha=0.7)
z_abc = np.polyfit(event_count, np.sqrt(np.mean((true_abc - abc_map)**2, axis=1)), 1)
p_abc = np.poly1d(z_abc)
plt.plot(event_count, p_abc(event_count), "g--", label="ABC regression", alpha=0.7)
plt.xlabel("Event count")
plt.ylabel("RSE")
plt.legend()
plt.tight_layout()
plt.savefig("results/sim_study/rse_vs_event_count.png")
plt.close()
