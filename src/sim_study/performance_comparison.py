    

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
from tensorflow.keras.models import load_model
import utils.utils_surface_NS as us

def compute_performance_stats(true_normalized, pred_normalized, params_std, params_mean):
    """
    Compute MSE, MAE, bias (normalized and denormalized) and standard error of MSE.
    """
    true_denorm = true_normalized * params_std + params_mean
    pred_denorm = pred_normalized * params_std + params_mean
    mse_normalized = mean_squared_error(true_normalized, pred_normalized)
    mse = mean_squared_error(true_denorm, pred_denorm)
    mse_error = np.sqrt(mse / len(true_normalized))
    mae = mean_absolute_error(true_normalized, pred_normalized)
    mae_denorm = mean_absolute_error(true_denorm, pred_denorm)
    bias_normalized = np.mean(pred_normalized - true_normalized, axis=0)
    bias = np.mean(pred_denorm - true_denorm, axis=0)
    return {
        "mse_normalized": mse_normalized,
        "mse": mse,
        "mse_error": mse_error,
        "mae": mae,
        "mae_denormalized": mae_denorm,
        "bias_normalized": bias_normalized.tolist(),
        "bias": bias.tolist(),
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

print("Number of training data points:", len(params_train_normalized))

#Summary statistics dimension
gs = len(SS_0_train_normalized_neural[0])  # Dimension of summary statistics
print("Dimension of summary statistics:", gs)
N_train = 100
true_normalized = params_test_normalized[response_test == 1][:N_train,:]
true = true_normalized * params_std + params_mean
year = [2023]
load_results = False
#######################
#NF
if not load_results:
    print("\nLoading trained normalizing flow posterior...")
    posterior = torch.load("neural_networks/trained_posterior_NS.pt")
    print("Posterior loaded successfully.")
    print("\nSampling from the normalizing flow posterior...")
    SS_nf = torch.tensor(SS_0_test_normalized_neural[response_test == 1], dtype=torch.float32)
    print("SS_nf shape:", SS_nf.shape)
    N_posteriosr_samples = 100
    posterior_samples_normalized = np.zeros((N_train, N_posteriosr_samples, len(l_bounds_NS)))
    for i, ss in enumerate(SS_nf[:N_train]):
        posterior_samples_normalized[i] = posterior.sample((N_posteriosr_samples,), x=ss).numpy()
        if i % (N_train//10) == 0:
                print(round(i/N_train*100), "% done")

    postetrior_samples = posterior_samples_normalized * params_std + params_mean
    print("\nComputing MAP estimate from the normalizing flow posterior...")
    nf_map_normalized = np.mean(posterior_samples_normalized, axis=1)#posterior.map(x=SS_nf)
    nf_map = nf_map_normalized * params_std + params_mean
    #save the samples and MAP estimate
    np.savez("results/sim_study/nf.npz", posterior_samples=postetrior_samples, posterior_samples_normalized =  posterior_samples_normalized, nf_map=nf_map, nf_map_normalized=nf_map_normalized)
else:
    print("\nLoading normalizing flow results...")
    nf_results = np.load("results/sim_study/nf.npz", allow_pickle=True)
    posterior_samples = nf_results["posterior_samples"]
    posterior_samples_normalized = nf_results["posterior_samples_normalized"]
    nf_map = nf_results["nf_map"]
    nf_map_normalized = nf_results["nf_map_normalized"]
    print("Normalizing flow results loaded successfully.")

# Compute and save performance stats for NF
nf_stats = compute_performance_stats(
    true_normalized,
    nf_map_normalized,
    params_std,
    params_mean
)
#######################
#NCL
if not load_results:
    print("\nLoading trained neural network for NCL...")
    classification_NN_NS = load_model("neural_networks/classification_NN_NS.h5")
    print("Model loaded successfully.")
    print("\nPredicting parameters using the neural network model...")
    N_grid = 1000
    ncl_mle = np.zeros((N_train, len(l_bounds_NS)))
    for i in range(N_train):
        print("\nPredicting parameters using the neural network model...")
        if i % (N_train//10) == 0:
            print(round(i/N_train*100), "% done")
        grid = us.LHS(N_grid, l_bounds_NS, u_bounds_NS, year)[:,:-1]
        grid_norm = (grid - params_mean) / params_std
        ll_grid = classification_NN_NS.predict([SS_0_test_normalized_neural[response_test==1][i].reshape((1,-1)).repeat(N_grid, axis = 0), grid_norm]).reshape(-1)  
        start = grid_norm[np.argmax(ll_grid)]
        MLE_NCL, MLE_NCL_normalized, final_logit, _, end_state = us.numerical_optim(start, SS_0_test_normalized_neural[response_test==1][i], l_bounds_NS, u_bounds_NS, classification_NN_NS, gs, params_mean=params_mean, params_std=params_std)
        ncl_mle[i] = MLE_NCL
    ncl_mle_normalized = (ncl_mle - params_mean) / params_std
    print("\nSaving the NCL MLE estimates...")
    np.savez("results/sim_study/ncl.npz", ncl_mle=ncl_mle, ncl_mle_normalized=ncl_mle_normalized)
else:
    print("\nLoading NCL results...")
    ncl_results = np.load("results/sim_study/ncl.npz", allow_pickle=True)
    ncl_mle = ncl_results["ncl_mle"]
    ncl_mle_normalized = ncl_results["ncl_mle_normalized"]
    print("NCL results loaded successfully.")
# Compute and save performance stats for NF
ncl_stats = compute_performance_stats(
    true_normalized,
    ncl_mle_normalized,
    params_std,
    params_mean
)
#######################
plt.figure(figsize=(10, 10))
num_params = len(l_bounds_NS)
rows, cols = 2, 2
for i in range(num_params):
    plt.subplot(rows, cols, i+1)
    plt.scatter(nf_map[:, i], true[:, i], alpha=0.5, label=f"NF MAP({np.mean((nf_map[:, i] - true[:, i])**2):.3f})")
    plt.scatter(ncl_mle[:, i], true[:, i], alpha=0.5, label=f"NCL MLE({np.mean((ncl_mle[:, i] - true[:, i])**2):.3f})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.plot([l_bounds_NS[i], u_bounds_NS[i]], [l_bounds_NS[i], u_bounds_NS[i]], '--', color = "grey")
    plt.xlim(l_bounds_NS[i], u_bounds_NS[i])
    plt.ylim(l_bounds_NS[i], u_bounds_NS[i])
    plt.title(f"{col_names_params_NS[i]}")
    plt.legend()
plt.tight_layout()
plt.savefig("results/sim_study/scatterplot.png")
plt.close()


with open("results/sim_study/performance_stats.json", "w") as f:
    json.dump(nf_stats, f, indent=4)
    json.dump(ncl_stats, f, indent=4)