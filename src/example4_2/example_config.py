import numpy as np
import utils.utils_surface_NS as us
import pandas as pd

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

print("Bound on parameters: ", l_bounds_NS, u_bounds_NS)

year = [2023]
theta_true = np.array([np.log(1), np.log(0.1), 1.5, -0.6, 0.5])
#theta_true = l_bounds_NS + 0.5 * (u_bounds_NS - l_bounds_NS)  # Set theta_true to the midpoint of the bounds for each parameter
theta_true_with_year = np.hstack([theta_true, year]).reshape((1, -1))
X_obs, Y_obs, metro_year_list_obs, _ = us.simulate_given_params(J=1, K=1, T=T, p=p, df_metro = df_metro, params=theta_true_with_year, verbose=False)
print("True parameters (theta_true): ", theta_true)
print("Number of data points in observed data: ", X_obs[0].shape[0])

#compute summary statistics for the observed data
SS_obs = us.summary_statistics(X_obs[0], T, cluster_bins, percentiles, df_metro, metro_year_list_obs[0], verbose=False, p = p)
cluster_bins_abc = np.logspace(np.log10(0.001), np.log10(12), 20)
SS_obs_abc = us.summary_statistics(X_obs[0], T, cluster_bins_abc, percentiles, df_metro, metro_year_list_obs[0], verbose=False, p = p)

#save the context variables to a .npz file for use in the ABC algorithm
np.savez('results/example4_2/theta_true.npz', X_obs=X_obs, Y_obs=Y_obs, metro_year_list_obs=metro_year_list_obs, theta_true=theta_true, SS_obs=SS_obs, SS_obs_abc=SS_obs_abc)


