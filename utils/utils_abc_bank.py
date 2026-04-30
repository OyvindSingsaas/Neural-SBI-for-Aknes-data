import numpy as np
import utils.utils_surface_NS as us
from sklearn.linear_model import Lasso, LinearRegression


def abc_prior(n, l_bounds=None, u_bounds=None):
    return np.random.uniform(low=l_bounds, high=u_bounds, size=(n, len(l_bounds)))


def sample_data(theta, df_metro, T):
    fixed_year = 2023
    param = np.hstack([theta, np.array([[fixed_year]])])
    X_train, Y_train, metro_year_list_train, _ = us.simulate_given_params(
        J=1, K=1, T=T, p=len(theta[0]), df_metro=df_metro, params=param, error=0, verbose=False)
    return X_train, Y_train, metro_year_list_train


def compute_summary_statistics(x, T, cluster_bins, percentiles, df_metro, metro_year, dim):
    return us.summary_statistics(x, T, cluster_bins, percentiles, df_metro, metro_year, verbose=False, p=dim)


def generate_bank_raw(N_bank, dim, l_bounds, u_bounds, df_metro, T, m=1):
    """
    Generate a bank of raw events (no SS computed yet).
    Use this when you want to recompute SS for different cluster_bins configs
    without re-running the simulator.

    Returns:
    theta_bank (N_bank, dim): Parameter vectors.
    X_bank (list of N_bank arrays): Raw event times for each simulation.
    metro_year_bank (list of N_bank): Metro year lists.
    """
    theta_bank = np.zeros((N_bank, dim))
    X_bank = []
    metro_year_bank = []
    k = 0
    print_every = max(N_bank // 10, 1)
    while k < N_bank:
        theta = abc_prior(1, l_bounds=l_bounds, u_bounds=u_bounds)
        x, _, metro_year = sample_data(theta, df_metro, T)
        if len(x[0]) < m:
            continue
        theta_bank[k] = theta
        X_bank.append(x[0])
        metro_year_bank.append(metro_year[0])
        k += 1
        if k % print_every == 0:
            print(f"  Bank: {k}/{N_bank} entries generated")
    return theta_bank, X_bank, metro_year_bank


def compute_SS_for_bank(X_bank, metro_year_bank, T, cluster_bins, percentiles, df_metro, dim):
    """Compute summary statistics for all bank entries given a cluster_bins config."""
    SS_bank = np.zeros((len(X_bank), dim + len(cluster_bins) + len(percentiles) - 1))
    for i, (x, metro_year) in enumerate(zip(X_bank, metro_year_bank)):
        SS_bank[i] = compute_summary_statistics(x, T, cluster_bins, percentiles, df_metro, metro_year, dim)
    return SS_bank


def generate_bank(N_bank, dim, l_bounds, u_bounds, df_metro, T, cluster_bins, percentiles, m=1):
    """
    Pre-generate a bank of (theta, SS) pairs by sampling from the prior.
    This is done once and reused for all test points.

    Parameters:
    N_bank (int): Number of (theta, SS) pairs to generate.
    m (int): Minimum number of events required per simulation.

    Returns:
    theta_bank (N_bank, dim): Parameter vectors.
    SS_bank (N_bank, gs): Summary statistics.
    """
    theta_bank = np.zeros((N_bank, dim))
    SS_bank = []
    k = 0
    print_every = max(N_bank // 10, 1)
    while k < N_bank:
        theta = abc_prior(1, l_bounds=l_bounds, u_bounds=u_bounds)
        x, _, metro_year = sample_data(theta, df_metro, T)
        if len(x[0]) < m:
            continue
        SS = compute_summary_statistics(x[0], T, cluster_bins, percentiles, df_metro, metro_year[0], dim)
        theta_bank[k] = theta
        SS_bank.append(SS)
        k += 1
        if k % print_every == 0:
            print(f"  Bank: {k}/{N_bank} entries generated")
    return theta_bank, np.array(SS_bank)


def _distances_batch(SS_bank, SS_obs, a, b, var_theta):
    """Vectorised distance computation for all bank entries at once."""
    X = SS_bank - SS_obs[np.newaxis, :]          # (N_bank, gs)
    theta_hat = a[np.newaxis, :] + X @ b.T       # (N_bank, dim)
    distances = np.sum((theta_hat - a[np.newaxis, :]) ** 2 / var_theta[np.newaxis, :], axis=1)
    return distances


def _fit_regression(theta_pilot, SS_pilot, SS_obs, dim, lasso_penalty):
    """Lasso variable selection + OLS refit."""
    X = SS_pilot - SS_obs
    a_array = np.zeros(dim)
    b_array = np.zeros((dim, len(SS_obs)))
    for d in range(dim):
        lasso = Lasso(alpha=lasso_penalty)
        lasso.fit(X, theta_pilot[:, d])
        selected = np.where(lasso.coef_ != 0)[0]
        if len(selected) == 0:
            a_array[d] = np.mean(theta_pilot[:, d])
        else:
            ols = LinearRegression()
            ols.fit(X[:, selected], theta_pilot[:, d])
            a_array[d] = ols.intercept_
            b_array[d, selected] = ols.coef_
    return a_array, b_array


def abc_pilot_run_bank(theta_bank, SS_bank, SS_obs, dim,
                       epsilon_percentile=1, lasso_penalty=0.01, k_pilot=None):
    """
    Fit the regression model using the pre-generated bank.

    Parameters:
    theta_bank, SS_bank: full pre-generated bank.
    SS_obs: observed summary statistics (raw scale, matching the bank).
    k_pilot: if set, randomly subsample this many rows from the bank for regression fitting.
             If None, use the full bank.

    Returns: var_theta, epsilon, a_array, b_array
    """
    if k_pilot is not None and k_pilot < len(theta_bank):
        idx = np.random.choice(len(theta_bank), k_pilot, replace=False)
        theta_pilot = theta_bank[idx]
        SS_pilot = SS_bank[idx]
    else:
        theta_pilot = theta_bank
        SS_pilot = SS_bank

    a_array, b_array = _fit_regression(theta_pilot, SS_pilot, SS_obs, dim, lasso_penalty)

    X = SS_pilot - SS_obs
    theta_pilot_hat = a_array[np.newaxis, :] + X @ b_array.T
    var_theta = np.var(theta_pilot_hat, axis=0)
    var_theta = np.maximum(var_theta, 1e-10)

    distances = _distances_batch(SS_pilot, SS_obs, a_array, b_array, var_theta)
    epsilon = np.percentile(distances, epsilon_percentile)

    return var_theta, epsilon, a_array, b_array


def abc_rejection_sampling_bank(theta_bank, SS_bank, SS_obs, epsilon,
                                k_abc, a_array, b_array, var_theta):
    """
    Rejection sampling from the pre-generated bank — no simulation required.

    Computes distances for all bank entries, accepts those within epsilon,
    and subsamples k_abc from the accepted set. If fewer than k_abc entries
    are accepted, falls back to the k_abc closest entries.

    Returns: posterior_samples_theta (k_abc, dim), posterior_samples_SS (k_abc, gs)
    """
    distances = _distances_batch(SS_bank, SS_obs, a_array, b_array, var_theta)
    accepted_idx = np.where(distances <= epsilon)[0]

    if len(accepted_idx) < k_abc:
        print(f"  Warning: only {len(accepted_idx)} accepted (need {k_abc}), using {k_abc} closest.")
        accepted_idx = np.argsort(distances)[:k_abc]
    else:
        accepted_idx = np.random.choice(accepted_idx, k_abc, replace=False)

    return theta_bank[accepted_idx], SS_bank[accepted_idx]
