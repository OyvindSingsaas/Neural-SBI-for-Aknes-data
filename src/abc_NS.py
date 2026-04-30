import numpy as np
import matplotlib.pyplot as plt 
import utils.utils_abc as abc
from scipy.stats import t, invgamma, norm
import time
import pandas as pd

def main():
    # Load the data
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

    t_0 = time.time()  # Start timer for the entire ABC process
    # Step 1: Pilot run to fit linear model
    k_pilot = 10000  # Number of pilot simulations
    k_abc = 100  # Number of posterior samples to obtain from ABC
    m = 50  # Number of data points for each simulation
    epsilon_percentile = 1  # Percentile for determining the tolerance level (epsilon) in ABC rejection sampling
    lasso_penalty = 0.01  # Lasso penalty for linear regression in the pilot run
    dim = len(l_bounds_NS)  # Dimensionality of the parameter space
    max_iter = 10000  # Maximum number of iterations to prevent infinite loops in rejection sampling
    cluster_bins = np.logspace(np.log10(0.001), np.log10(12), 20)

    # Load the observed summary statistics and true parameter values from the example configuration
    example_data = np.load('results/example4_2/theta_true.npz', allow_pickle=True)
    SS_obs = example_data['SS_obs']
    SS_obs_abc = example_data['SS_obs_abc']
    theta_true = example_data['theta_true']

    theta_true_normalized = (theta_true - params_mean) / params_std
    SS_obs_normalized = (SS_obs - SS_mean) / SS_std

    print("SS_obs = ", SS_obs)
    print("SS_obs_abc = ", SS_obs_abc)
    print("theta_true = ", theta_true)
    print("\nRunning ABC pilot run to fit linear model...")
    var_theta, epsilon, a_array, b_array = abc.abc_pilot_run(k_pilot, m, SS_obs=SS_obs_abc, dim=dim, epsilon_percentile=epsilon_percentile,
                                                              l_bounds=l_bounds_NS, u_bounds=u_bounds_NS, df_metro=df_metro, T=T, cluster_bins=cluster_bins,
                                                                percentiles=percentiles, lasso_penalty=lasso_penalty)
    print("Empirical variance of the fitted parameter vectors from pilot run:", var_theta)
    print("Tolerance level (epsilon) based on the specified percentile:", epsilon)
    print("\n Linear model coefficients (a_array):", a_array)
    print("\n Linear model coefficients (b_array):", b_array)

    print("\nRunning ABC rejection sampling...")
    posterior_samples_theta, posterior_samples_SS = abc.abc_rejection_sampling(k_abc, SS_obs_abc, epsilon, m, dim, a_array, b_array, var_theta, l_bounds=l_bounds_NS, u_bounds=u_bounds_NS,
                                                                                df_metro=df_metro, T=T, cluster_bins=cluster_bins, percentiles=percentiles, max_iter = max_iter)
    t_1 = time.time()  # End timer for the entire ABC process
    print(f"ABC sampling completed in {t_1 - t_0:.2f} seconds.")

    #Save posterior samples and summary statistics
    np.savez('results/NS_ABC_example.npz', posterior_samples_theta=posterior_samples_theta, posterior_samples_SS=posterior_samples_SS, SS_obs=SS_obs, SS_obs_abc=SS_obs_abc, theta_true=theta_true, theta_true_normalized=theta_true_normalized, SS_obs_normalized=SS_obs_normalized)

    posterior_samples_theta_normalized = (posterior_samples_theta - params_mean) / params_std

    print("\n Plotting ABC posterior samples for the parameters...")
    # Extract ABC samples for the parameters
    for i in range(dim):
        plt.figure()
        plt.hist(posterior_samples_theta[:, i], bins=30, density=True, alpha=0.7, color='blue')
        plt.axvline(theta_true[i], color='red', linestyle='dashed', linewidth=2, label='True value')
        plt.axvline(np.mean(posterior_samples_theta[:, i]), color='green', linestyle='dashed', linewidth=2, label='Posterior mean')
        plt.axvline(np.percentile(posterior_samples_theta[:, i], 2.5), color='green', linestyle='dashed', linewidth=2, label='95% CI')
        plt.axvline(np.percentile(posterior_samples_theta[:, i], 97.5), color='green', linestyle='dashed', linewidth=2)
        plt.title(f"Posterior distribution for {col_names_params_NS[i]}")
        plt.xlabel(f"{col_names_params_NS[i]}")
        plt.ylabel("Density")
        plt.legend()
        #save the plot
        plt.savefig(f"plots/test/NS_ABC_posterior_param{i}.png")
        plt.close()
    
    # Plot all summary statistics in a single figure with subplots
    print("\n Plotting ABC posterior samples for the summary statistics (all in one figure)...")
    n_stats = len(SS_obs)
    n_cols = min(4, n_stats)
    n_rows = int(np.ceil(n_stats / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 2.5*n_rows), squeeze=False)
    for i in range(n_stats):
        ax = axes[i // n_cols, i % n_cols]
        ax.hist(posterior_samples_SS[:, i], bins=30, density=True, alpha=0.7, color='blue')
        ax.axvline(SS_obs_normalized[i], color='red', linestyle='dashed', linewidth=1.2)
        ax.axvline(np.mean(posterior_samples_SS[:, i]), color='green', linestyle='dashed', linewidth=1.2)
        ax.axvline(np.percentile(posterior_samples_SS[:, i], 2.5), color='green', linestyle='dashed', linewidth=1.2)
        ax.axvline(np.percentile(posterior_samples_SS[:, i], 97.5), color='green', linestyle='dashed', linewidth=1.2)
        ax.set_title(f"SS {i}", fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    # Remove empty subplots
    for j in range(n_stats, n_rows * n_cols):
        fig.delaxes(axes[j // n_cols, j % n_cols])
    plt.tight_layout()
    plt.savefig("plots/test/NS_ABC_posterior_SS_all.png", dpi=150)
    plt.close()

main()