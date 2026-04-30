import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
from tensorflow.keras.models import load_model
import utils.utils_surface_NS as us
import utils.utils_abc as abc
from scipy.stats import t, invgamma, norm
import time

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
N_train = 500
N_start = 100
year = [2023]
gs = len(SS_0_train_normalized_neural[0])  # Dimension of summary statistics
print("Dimension of summary statistics:", gs)
true_normalized = params_test_normalized[response_test == 1][N_start:N_train,:]
true = true_normalized * params_std + params_mean

SS_abc = SS_0_test_normalized_neural[response_test == 1][N_start:N_train,:]
SS_abc = SS_abc * SS_std + SS_mean  # Unnormalize summary statistics for ABC

k_pilot = 1000  # Number of pilot simulations
k_abc = 200  # Number of posterior samples to obtain from ABC
m = 5  # Number of data points for each simulation
epsilon_percentile = 1  # Percentile for determining the tolerance level (epsilon) in ABC rejection sampling
lasso_penalty = 0.001  # Lasso penalty for linear regression in the pilot run
dim = len(l_bounds_NS)  # Dimensionality of the parameter space
max_iter = 10000  # Maximum number of iterations to prevent infinite loops in rejection sampling

print("\nRunning ABC ")
average_time_pilot = 0
average_time_rejection = 0
abc_samples = np.zeros((N_train - N_start, k_abc, len(l_bounds_NS)))
completed = 0
for i in range(N_train - N_start):
    print("\nRunning ABC for data point", i+1, "out of", N_train - N_start)
    try:
        t_0 = time.time()
        #Pilot run to fit linear model
        var_theta, epsilon, a_array, b_array = abc.abc_pilot_run(k_pilot, m, SS_obs=SS_abc[i], dim=dim, epsilon_percentile=epsilon_percentile,
                                                                  l_bounds=l_bounds_NS, u_bounds=u_bounds_NS, df_metro=df_metro, T=T, cluster_bins=cluster_bins,
                                                                    percentiles=percentiles, lasso_penalty=lasso_penalty)
        t_1 = time.time()
        #Run ABC rejection sampling to obtain posterior samples
        abc_samples[i], _ = abc.abc_rejection_sampling(k_abc, SS_abc[i], epsilon, m, dim, a_array, b_array, var_theta, l_bounds=l_bounds_NS, u_bounds=u_bounds_NS,
                                                                                    df_metro=df_metro, T=T, cluster_bins=cluster_bins, percentiles=percentiles, max_iter = max_iter)
        t_3 = time.time()
        average_time_pilot += t_1 - t_0
        average_time_rejection += t_3 - t_1
        completed += 1

        print_every = max((N_train - N_start) // 10, 1)
        if (i+1) % print_every == 0:
            print(round((i+1)/N_train*100), "% done")
            print(f"Average time taken per data point for ABC run so far: {(average_time_pilot + average_time_rejection)/completed:.2f} seconds")
            print(f"Expected time remaining: {((average_time_pilot + average_time_rejection)/((i+1)/(N_train - N_start)) - (average_time_pilot + average_time_rejection))/60:.2f} minutes")

    except Exception as e:
        print(f"\nERROR at data point {i+1}: {e}")
        print(f"Saving {completed} completed samples to emergency checkpoint...")
        np.savez("results/sim_study/abc_samples_v1_emergency.npz", abc_samples=abc_samples, completed=completed,
                 true=true, true_normalized=true_normalized, N_train=N_train)
        raise

n_done = max(completed, 1)
average_time_pilot /= n_done
average_time_rejection /= n_done
print(f"\nAverage time taken per data point for ABC pilot run: {average_time_pilot:.2f} seconds")
print(f"\nAverage time taken per data point for ABC rejection sampling: {average_time_rejection:.2f} seconds")
print("\nDONE\n")
abc_samples_normalized = (abc_samples - params_mean) / params_std
abc_map_normalized = np.mean(abc_samples_normalized, axis=1)
abc_map = np.mean(abc_samples, axis=1)

#save the samples and MAP estimate
np.savez("results/sim_study/abc_samples_large_v2_second.npz", abc_samples=abc_samples, abc_samples_normalized=abc_samples_normalized, abc_map=abc_map, abc_map_normalized=abc_map_normalized, N_train=N_train, true=true, true_normalized=true_normalized)