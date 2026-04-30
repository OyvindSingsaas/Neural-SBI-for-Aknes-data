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
data = np.load('data/NS_data_temp_wp_10RK_small.npz', allow_pickle=True)
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
l_bounds_NS_norm = (l_bounds_NS - params_mean) /params_std
u_bounds_NS_norm = (u_bounds_NS - params_mean) /params_std
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


#set up the prior distribution for the parameters based on the bounds from the data generation process, casted as tensors for use in sbi
prior = BoxUniform(low=torch.tensor(l_bounds_NS_norm, dtype=torch.float32), high=torch.tensor(u_bounds_NS_norm, dtype=torch.float32))

inference = sbi_inference.SNLE(prior=prior)

#Reformat the training data for use in sbi
params_train = params_train_normalized[response_train == 1]
SS_train = SS_0_train_normalized_neural[response_train == 1]
params_test = params_test_normalized[response_test == 1]
SS_test = SS_0_test_normalized_neural[response_test == 1]
#print shape of training and test data
print("Shape of training parameters:", params_train.shape)
print("Shape of training summary statistics:", SS_train.shape)


print("Training the neural network posterior...")
density_estimator = inference.append_simulations(
    theta=torch.tensor(params_train, dtype=torch.float32),  # (N, 4)
    x = torch.tensor(SS_train, dtype=torch.float32)       # (N, 9)
).train()
posterior = inference.build_posterior(density_estimator)
print("Neural network posterior trained successfully.")

#save the trained posterior for later use
torch.save(posterior, 'neural_networks/trained_posterior_NS_small.pt')
print("Neural network posterior trained and saved successfully.")