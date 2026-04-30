import numpy as np
import pandas as pd
import utils.utils_abc_bank as abc_bank
import time

print("Loading data...")
data = np.load('data/NS_data_temp_wp_10RK.npz', allow_pickle=True)
params_test_normalized = data['params_test_normalized']
SS_0_test_normalized_neural = data['SS_0_test_normalized_neural']
response_test = data['response_test']
params_mean = data['params_mean']
params_std = data['params_std']
SS_mean = data['SS_mean']
SS_std = data['SS_std']
l_bounds_NS = data['l_bounds_NS']
u_bounds_NS = data['u_bounds_NS']
T = data['T']
cluster_bins = data['cluster_bins']
percentiles = data['percentiles']

df_metro = pd.DataFrame(data['df_metro_values'],
                    columns=data['df_metro_columns'],
                    index=data['df_metro_index'])
print("Data loaded successfully.")

N_train = 500
dim = len(l_bounds_NS)
k_pilot = None       # use the full bank for regression fitting
k_abc = 1000          # number of posterior samples per test point
N_bank = 100000       # size of the pre-generated simulation bank
m = 5                # minimum events per simulation
epsilon_percentile = 1
lasso_penalty = 0.001

true_normalized = params_test_normalized[response_test == 1][:N_train, :]
true = true_normalized * params_std + params_mean

SS_abc = SS_0_test_normalized_neural[response_test == 1][:N_train, :]
SS_abc = SS_abc * SS_std + SS_mean  # Unnormalize to raw scale for ABC

# --- Generate simulation bank (done once) ---
print(f"\nGenerating simulation bank with N_bank={N_bank}...")
t_bank_0 = time.time()
theta_bank, SS_bank = abc_bank.generate_bank(
    N_bank=N_bank, dim=dim, l_bounds=l_bounds_NS, u_bounds=u_bounds_NS,
    df_metro=df_metro, T=T, cluster_bins=cluster_bins, percentiles=percentiles, m=m)
t_bank_1 = time.time()
print(f"Bank generated in {(t_bank_1 - t_bank_0)/60:.1f} minutes.")

# --- ABC loop (no simulation inside) ---
print("\nRunning ABC rejection sampling from bank...")
abc_samples = np.zeros((N_train, k_abc, dim))
completed = 0
average_time_pilot = 0
average_time_rejection = 0

for i in range(N_train):
    print(f"\nData point {i+1}/{N_train}")
    try:
        t_0 = time.time()
        var_theta, epsilon, a_array, b_array = abc_bank.abc_pilot_run_bank(
            theta_bank, SS_bank, SS_obs=SS_abc[i], dim=dim,
            epsilon_percentile=epsilon_percentile, lasso_penalty=lasso_penalty,
            k_pilot=k_pilot)
        t_1 = time.time()

        abc_samples[i], _ = abc_bank.abc_rejection_sampling_bank(
            theta_bank, SS_bank, SS_obs=SS_abc[i], epsilon=epsilon,
            k_abc=k_abc, a_array=a_array, b_array=b_array, var_theta=var_theta)
        t_2 = time.time()

        average_time_pilot += t_1 - t_0
        average_time_rejection += t_2 - t_1
        completed += 1

        print_every = max(N_train // 10, 1)
        if (i + 1) % print_every == 0:
            print(f"{round((i+1)/N_train*100)}% done")
            print(f"Avg time per point: {(average_time_pilot + average_time_rejection)/completed:.2f}s")
            print(f"Est. remaining: {((average_time_pilot + average_time_rejection)/((i+1)/N_train) - (average_time_pilot + average_time_rejection))/60:.2f} minutes")

    except Exception as e:
        print(f"\nERROR at data point {i+1}: {e}")
        print(f"Saving {completed} completed samples to emergency checkpoint...")
        np.savez("results/sim_study/abc_samples_bank_emergency.npz", abc_samples=abc_samples,
                 completed=completed, true=true, true_normalized=true_normalized, N_train=N_train)
        raise

n_done = max(completed, 1)
print(f"\nAvg time per point — pilot: {average_time_pilot/n_done:.2f}s, rejection: {average_time_rejection/n_done:.2f}s")
print("\nDONE\n")

abc_samples_normalized = (abc_samples - params_mean) / params_std
abc_map_normalized = np.mean(abc_samples_normalized, axis=1)
abc_map = np.mean(abc_samples, axis=1)

np.savez("results/sim_study/abc_samples_bank.npz",
         abc_samples=abc_samples, abc_samples_normalized=abc_samples_normalized,
         abc_map=abc_map, abc_map_normalized=abc_map_normalized,
         N_train=N_train, true=true, true_normalized=true_normalized)
print("Results saved to results/sim_study/abc_samples_bank.npz")
