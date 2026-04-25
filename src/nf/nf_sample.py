
from sbi.inference import NLE
from sbi import inference as sbi_inference
import numpy as np
import torch
import pandas as pd
from sbi.utils import BoxUniform
#from numba import jit
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error


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
p = len(l_bounds_NS)

df_metro = pd.DataFrame(data['df_metro_values'],
                    columns=data['df_metro_columns'],
                    index=data['df_metro_index'])

print("Data loaded successfully.")

print("\nLoading true parameter and observed data from example_config...")
example_config = np.load("results/example4_2/theta_true.npz", allow_pickle=True)
X_obs = example_config["X_obs"]
Y_obs = example_config["Y_obs"]
metro_year_list_obs = example_config["metro_year_list_obs"]
theta_true = example_config["theta_true"]
theta_normalized = (theta_true - params_mean) / params_std
SS_obs = example_config["SS_obs"]
SS_obs_normalized = (SS_obs - SS_mean) / SS_std
print("True parameter and observed data loaded successfully.")
year = [2023]
gs = len(SS_mean)
print("SS_obs_normalized = ", SS_obs_normalized, "theta_true = ", theta_true)

#load posterior
print("\nLoading trained normalizing flow posterior...")
posterior = torch.load("neural_networks/trained_posterior_NS.pt")
ss_torch = torch.tensor(SS_obs_normalized, dtype=torch.float32)
print("Posterior loaded successfully.")
print("\nSampling from the normalizing flow posterior...")
samples_normalized = posterior.sample((10000,), x=ss_torch)
samples = samples_normalized*params_std + params_mean
print("Sampling completed successfully.")
print("\nComputing MAP estimate from the normalizing flow posterior...")
posterior.set_default_x(ss_torch)
NF_map_normalized = posterior.map()
NF_map_normalized  = np.array(NF_map_normalized).reshape((-1))
NF_map = NF_map_normalized*params_std + params_mean

#save the samples and MAP estimate
print("\nSaving the normalizing flow samples and MAP estimate...")
np.savez("results/NF_samples_NS.npz", samples=samples, NF_map=NF_map)
print("Normalizing flow samples and MAP estimate saved successfully.")

