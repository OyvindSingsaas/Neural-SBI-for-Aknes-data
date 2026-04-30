import numpy as np
import pandas as pd
import utils.utils_abc_bank as abc_bank
import utils.utils_surface_NS as us
import time

# --- Config ---
N_test = 50
N_bank = 100000
nbins_configs = 10
nbins_plan = np.linspace(1, 30, nbins_configs, dtype=int)
r_min = 0.01   # fixed based on prior search
r_max = 10      # fixed based on prior search
percentiles = [10, 50, 90]

k_abc = 200
m = 5
epsilon_percentile = 1
lasso_penalty = 0.01

# --- Load data ---
print("Loading data...")
data = np.load('data/NS_data_temp_wp_10RK.npz', allow_pickle=True)


df_metro = pd.DataFrame(data['df_metro_values'],
                        columns=data['df_metro_columns'],
                        index=data['df_metro_index'])
print("Data loaded successfully.")

# Raw test events — load from NS_events.npz which has the original event arrays
print("Loading raw test events...")
events_data = np.load('data/NS_events.npz', allow_pickle=True)
X_train = events_data['X_train']
Y_train = events_data['Y_train']
metro_year_list_train = events_data['metro_year_list_train']
invalid_index_train = events_data['invalid_index_train']
X_test = events_data['X_test']
Y_test = events_data['Y_test']
metro_year_list_test = events_data['metro_year_list_test']
#invalid_index_test = events_data['invalid_index_test']
l_bounds_NS = events_data['l_bounds_NS']
u_bounds_NS = events_data['u_bounds_NS']
params_sample_test_NS = events_data['params_sample_test_NS']

true = params_sample_test_NS[:N_test, :-1]
dim = len(l_bounds_NS)
T = 365
# --- Generate raw simulation bank once ---
print(f"\nGenerating raw simulation bank (N_bank={N_bank})...")
t0 = time.time()
theta_bank, X_bank, metro_year_bank = abc_bank.generate_bank_raw(
    N_bank=N_bank, dim=dim, l_bounds=l_bounds_NS, u_bounds=u_bounds_NS,
    df_metro=df_metro, T=T, m=m)
print(f"Bank generated in {(time.time() - t0)/60:.1f} minutes.")

# --- Search over N_bins ---
rmse_list = []
se_list = []

print(f"\n=== Searching N_bins (r_min={r_min}, r_max={r_max}) ===")
print(f"N_bins plan: {nbins_plan}")

for config_i, N_bins in enumerate(nbins_plan):
    print(f"\n[{config_i+1}/{len(nbins_plan)}] N_bins = {N_bins}")
    cluster_bins = np.logspace(np.log10(r_min), np.log10(r_max), N_bins)

    # Compute SS for bank with this cluster_bins config
    print(f"  Computing SS for bank ({N_bank} entries)...")
    t_ss = time.time()
    SS_bank = abc_bank.compute_SS_for_bank(X_bank, metro_year_bank, T, cluster_bins, percentiles, df_metro, dim)
    print(f"  Bank SS computed in {time.time() - t_ss:.1f}s, shape: {SS_bank.shape}")

    # Compute SS for test points with this cluster_bins config
    print(f"  Computing SS for {N_test} test points...")
    SS_test = np.zeros((N_test, SS_bank.shape[1]))
    for i in range(N_test):
        SS_test[i] = us.summary_statistics(X_test[i], T, cluster_bins, percentiles, df_metro, metro_year_list_test[i], verbose=False, p=dim)

    # Run ABC for each test point
    abc_map = np.zeros((N_test, dim))
    t_abc = time.time()
    for i in range(N_test):
        var_theta, epsilon, a_array, b_array = abc_bank.abc_pilot_run_bank(
            theta_bank, SS_bank, SS_obs=SS_test[i], dim=dim,
            epsilon_percentile=epsilon_percentile, lasso_penalty=lasso_penalty)

        samples_theta, _ = abc_bank.abc_rejection_sampling_bank(
            theta_bank, SS_bank, SS_obs=SS_test[i], epsilon=epsilon,
            k_abc=k_abc, a_array=a_array, b_array=b_array, var_theta=var_theta)

        abc_map[i] = np.mean(samples_theta, axis=0)

    print(f"  ABC done in {(time.time() - t_abc):.1f}s")

    # RMSE on original scale
    per_sample_se = np.mean((true - abc_map) ** 2, axis=1)
    mse = np.mean(per_sample_se)
    rmse = np.sqrt(mse)
    se = np.std(per_sample_se, ddof=1) / (2 * rmse * np.sqrt(N_test))
    rmse_list.append(rmse)
    se_list.append(se)
    print(f"  => RMSE: {rmse:.4f} ± {se:.4f}")

# --- Results ---
best_idx = int(np.argmin(rmse_list))
print(f"\n=== N_bins search complete ===")
print(f"{'N_bins':>8}  {'RMSE':>10}  {'SE':>10}")
for nb, rmse, se in zip(nbins_plan, rmse_list, se_list):
    marker = "  <-- best" if nb == nbins_plan[best_idx] else ""
    print(f"{nb:>8d}  {rmse:>10.4f}  {se:>10.4f}{marker}")
print(f"\nBest N_bins: {nbins_plan[best_idx]} (RMSE={rmse_list[best_idx]:.4f})")

np.savez("results/lag_search/abc_lagsearch_nbins.npz",
         nbins_values=nbins_plan,
         rmse_list=rmse_list,
         se_list=se_list,
         r_min=r_min, r_max=r_max,
         N_test=N_test, N_bank=N_bank)
print("Results saved to results/lag_search/abc_lagsearch_nbins.npz")
