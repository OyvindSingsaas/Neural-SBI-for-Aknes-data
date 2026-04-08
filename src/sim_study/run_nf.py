import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
from tensorflow.keras.models import load_model
import utils.utils_surface_NS as us


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
N_train = 100
year = [2023]
gs = len(SS_0_train_normalized_neural[0])  # Dimension of summary statistics
print("Dimension of summary statistics:", gs)
true_normalized = params_test_normalized[response_test == 1][:N_train,:]
true = true_normalized * params_std + params_mean

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
np.savez("results/sim_study/nf.npz", posterior_samples=postetrior_samples, posterior_samples_normalized =  posterior_samples_normalized, nf_map=nf_map, nf_map_normalized=nf_map_normalized, N_train=N_train, true=true, true_normalized=true_normalized)