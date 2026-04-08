import numpy as np
import matplotlib.pyplot as plt 
import utils.utils_abc as abc
from scipy.stats import t, invgamma, norm
import time



def abc_main():

    r"""
    TEST
    Main function to run the ABC algorithm.

    x \sim N(theta, 1)
    theta \sim N(0, 10^2)
    S = (mean(x), var(x))

    """
    t_0 = time.time()  # Start timer for the entire ABC process
    # Step 1: Pilot run to fit linear model
    k_pilot = 1000  # Number of pilot simulations
    k_abc = 10000 #Number of posterior samples to obtain from ABC
    m = 1  # Number of data points for each simulation
    epsilon_percentile = 1  # Percentile for determining the tolerance level (epsilon) in ABC rejection sampling
    dim = 2  # Dimensionality of the parameter space
    n_observed = 5  # Number of observed data points

    logsigma_2_true = np.log(1**2)  # True log(sigma^2) value for testing
    mu_true  =-0.5  # True mu value for testing
    theta_true = [mu_true, logsigma_2_true]  # True parameter value for testing

    x_obs = abc.sample_data(theta_true, n_observed)  # Simulate observed data using the true parameter value
    SS_obs = abc.compute_summary_statistics(x_obs)  # Compute summary statistics for the observed data
    print("SS_obs = ", SS_obs)

    print("\nRunning ABC pilot run to fit linear model...")
    var_theta, epsilon, a_array, b_array = abc.abc_pilot_run(k_pilot, m, SS_obs=SS_obs, n_observed=n_observed, dim=dim, epsilon_percentile=epsilon_percentile)
    print("Empirical variance of the fitted parameter vectors from pilot run:", var_theta)
    print("Tolerance level (epsilon) based on the specified percentile:", epsilon)

    print("\nRunning ABC rejection sampling...")
    posterior_samples_theta, posterior_samples_SS = abc.abc_rejection_sampling(k_abc, SS_obs, epsilon, m, dim, a_array, b_array, var_theta, n_observed)
    t_1 = time.time()  # End timer for the entire ABC process

    print("\n Plotting ABC and analytic posteriors for mu and sigma^2...")
    # Extract ABC samples for mu and sigma^2
    mu_samples = posterior_samples_theta[:, 0]
    sigma2_samples = np.exp(posterior_samples_theta[:, 1])  # Convert log(sigma^2) to sigma^2 for comparison with the analytic posterior

    # Normal-Inverse-Gamma prior hyperparameters (must match utils_abc)
    alpha0 = 1
    beta0 = 1
    kappa0 = 1
    mu0 = 0

    x_bar = np.mean(x_obs)
    S = np.sum((x_obs - x_bar)**2)

    kappa_n = kappa0 + n_observed
    mu_n = (kappa0 * mu0 + n_observed * x_bar) / kappa_n
    alpha_n = alpha0 + n_observed/2
    beta_n = beta0 + 0.5 * S + (kappa0 * n_observed * (x_bar - mu0)**2) / (2 * kappa_n)

    # Plot mu posterior (marginal: Student-t)
    # Student-t parameters
    df = 2 * alpha_n
    loc = mu_n
    scale = np.sqrt(beta_n / (kappa_n * alpha_n))
    # Set x boundaries for mu
    mu_x_min = np.percentile(mu_samples, 1)
    mu_x_max = np.percentile(mu_samples, 99)
    mu_x = np.linspace(mu_x_min, mu_x_max, 1000)
    mu_analytic = t.pdf(mu_x, df=df, loc=loc, scale=scale)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(mu_samples, bins=100, density=True, alpha=0.6, color='tab:green', label='ABC Posterior (mu)')
    ax.plot(mu_x, mu_analytic, color='tab:red', linewidth=2, label='Analytic Posterior (mu)')
    ax.axvline(mu_n, color='tab:orange', linestyle='--', linewidth=2, label=f'Posterior Mean ({mu_n:.2f})')
    ax.axvline(np.mean(mu_samples), color='tab:purple', linestyle='--', linewidth=2, label=f'ABC Posterior Mean ({np.mean(mu_samples):.2f})')
    ax.axvline(theta_true[0], color='tab:blue', linestyle='--', linewidth=2, label=f'True Value ({theta_true[0]:.2f})')
    ax.set_xlabel('mu', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_xlim(mu_x_min, mu_x_max)
    ax.legend(loc='upper left')
    plt.title('Posterior for mu: ABC vs Analytic', fontsize=14)
    plt.tight_layout()
    plt.savefig('plots/abc_test_mu_posterior_density.png')
    plt.close(fig)

    # Plot sigma^2 posterior (Inverse-Gamma)
    # Use central 99.8% interval for x boundaries
    sigma2_x_min = np.percentile(sigma2_samples, 1)
    sigma2_x_max = np.percentile(sigma2_samples, 99)    
    sigma2_x = np.linspace(sigma2_x_min, sigma2_x_max, 1000)
    sigma2_analytic = invgamma.pdf(sigma2_x, a=alpha_n, scale=beta_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(sigma2_samples, bins=100, density=True, alpha=0.6, color='tab:green', label='ABC Posterior')
    ax.plot(sigma2_x, sigma2_analytic, color='tab:red', linewidth=2, label='Analytic Posterior')
    ax.axvline(np.mean(sigma2_samples), color='tab:purple', linestyle='--', linewidth=2, label=f'ABC Posterior Mean ({np.mean(sigma2_samples):.2f})')
    ax.axvline(invgamma.mean(a=alpha_n, scale=beta_n), color='tab:orange', linestyle='--', linewidth=2, label=f'Posterior Mean ({invgamma.mean(a=alpha_n, scale=beta_n):.2f})')
    ax.axvline(np.exp(theta_true[1]), color='tab:blue', linestyle='--', linewidth=2, label=f'True Value ({np.exp(theta_true[1]):.2f})')
    ax.set_xlabel('sigma^2', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_xlim(sigma2_x_min, sigma2_x_max)
    ax.legend(loc='upper right')
    plt.title('Posterior for sigma^2: ABC vs Analytic', fontsize=14)
    plt.tight_layout()
    plt.savefig('plots/abc_test_sigma2_posterior_density.png')
    plt.close(fig)

    #Plot priors for mu and sigma^2 together with the true values and posterior means
    fig, ax = plt.subplots(figsize=(8, 6))
    # Prior for mu
    mu_prior_x = np.linspace(-10, 10, 1000)
    mu_prior = norm.pdf(mu_prior_x, loc=mu0, scale=np.sqrt(beta0 / (kappa0 * alpha0)))
    ax.plot(mu_prior_x, mu_prior, color='tab:blue', linestyle='-', linewidth=2, label='Prior (mu)')
    ax.axvline(theta_true[0], color='tab:blue', linestyle='--', linewidth=2, label='True Value (mu)')
    ax.axvline(mu_n, color='tab:orange', linestyle='--', linewidth=2, label='Posterior Mean (mu)')
    ax.axvline(np.mean(mu_samples), color='tab:purple', linestyle='--', linewidth=2, label='ABC Posterior Mean (mu)')
    ax.set_xlabel('Parameter Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend(loc='upper right')
    plt.title('Prior for mu', fontsize=14)
    plt.tight_layout()
    plt.savefig('plots/abc_test_prior_mu.png')
    plt.close(fig)

    # Prior for sigma^2
    fig, ax = plt.subplots(figsize=(8, 6))
    sigma2_prior_x = np.linspace(0.01, 10, 1000)
    sigma2_prior = invgamma.pdf(sigma2_prior_x, a=alpha0, scale=beta0)
    ax.plot(sigma2_prior_x, sigma2_prior, color='tab:blue', linestyle='-', linewidth=2, label='Prior (sigma^2)')
    ax.axvline(np.exp(theta_true[1]), color='tab:blue', linestyle='--', linewidth=2, label=f'True Value ({np.exp(theta_true[1]):.2f})')
    ax.axvline(invgamma.mean(a=alpha_n, scale=beta_n), color='tab:orange', linestyle='--', linewidth=2, label=f'Posterior Mean ({invgamma.mean(a=alpha_n, scale=beta_n):.2f})')
    ax.axvline(np.mean(sigma2_samples), color='tab:purple', linestyle='--', linewidth=2, label=f'ABC Posterior Mean ({np.mean(sigma2_samples):.2f})')
    ax.set_xlabel('Parameter Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend(loc='upper right')
    plt.title('Prior for sigma^2', fontsize=14)
    plt.tight_layout()
    plt.savefig('plots/abc_test_prior_sigma2.png')
    plt.close(fig)

    print("\nABC test completed. Posterior density plots saved as 'plots/abc_test_mu_posterior_density.png' and 'plots/abc_test_sigma2_posterior_density.png'.")
    print(f"Total ABC runtime: {t_1 - t_0:.2f} seconds\nDone.\n")  

    print("a_arrray", a_array)
    print("b_array", b_array)

abc_main()