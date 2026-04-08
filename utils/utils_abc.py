import numpy as np
from scipy.stats import invgamma
import utils.utils_surface_NS as us
from sklearn.linear_model import Lasso



# theta = (mu, log(sigma^2))

def abc_prior(n, l_bounds=None, u_bounds=None):
    """
    TEST
    Sample a parameter vector from the prior distribution.

    Returns:
    numpy.ndarray: A parameter vector sampled from the prior distribution.
    """
    theta = np.random.uniform(low=l_bounds, high=u_bounds, size=(n, len(l_bounds)))  # Example: Uniform prior over the specified bounds
    
    return theta

def sample_data(theta, df_metro, T, n=1):
    """
    TEST
    Sample data from the model using the given parameter vector.

    Parameters:
    theta (numpy.ndarray): (Array of)Parameter vector(s) to use for sampling data.
    n (int): Number of data points to sample for each theta.

    Returns:
    numpy.ndarray: Simulated data based on the input parameter vector.
    """
    fixed_year = 2023
    #add fixed year to theta to create a parameter vector of the form (theta, year)
    param = np.hstack([theta, np.array([[fixed_year]])])
    #simulate data using the simulate_given_params function from utils_surface_NS

    X_train, Y_train, metro_year_list_train, _ = us.simulate_given_params(J=1, K=1, T=T, p=len(theta[0]), df_metro = df_metro, params=param, error = 0, verbose=False)

    return X_train, Y_train, metro_year_list_train

def compute_summary_statistics(x, T, cluster_bins, percentiles, df_metro, metro_year_list_train, dim):
    """
    TEST
    Compute summary statistics for the given data.

    Parameters:
    x (numpy.ndarray): Input data for which to compute summary statistics.

    Returns:
    numpy.ndarray: Summary statistics computed from the input data.
    """
    SS = us.summary_statistics(x, T, cluster_bins, percentiles, df_metro, metro_year_list_train, verbose=False, p = dim)

    return SS  

def abc_distance(SS, SS_obs, a, b, var_theta):
    """
    Compute the distance for ABC.

    sum_j (theta_j - alpha_j)^2 / var_theta_j

    Parameters:
    SS (numpy.ndarray): Summary statistics for the simulated data.
    SS_obs (numpy.ndarray): Observed summary statistics.
    a (numpy.ndarray): Intercept vector from the linear model.
    b (numpy.ndarray): Slope matrix from the linear model.
    var_theta (numpy.ndarray): Empirical variance of the parameter vectors from linear model.
    Returns:
    float: The computed distance between the two parameter vectors.
    """
    #Compute the fitted parameter vector using the linear model
    X = SS - SS_obs
    theta_hat = a + b @ X

    # Compute the distance using a weighted Euclidean distance
    distance = np.sum((theta_hat - a) ** 2 / var_theta)
    return distance

def abc_pilot_run(k_pilot, m, SS_obs, dim, l_bounds, u_bounds, df_metro, T, cluster_bins, percentiles, SS_mean, SS_std, epsilon_percentile = 1, lasso_penalty = 0.01):
    """
    Perform a pilot run for ABC to estimate the empirical variance of the parameter vectors.

    Parameters:
    k_pilot (int): Number of pilot simulations to achieve.
    m (int): Minimum number of points required.
    SS_obs (numpy.ndarray): Observed summary statistics.
    epsilon_percentile (float): Percentile for the tolerance level to accept parameter vectors.
    dim (int): Dimension of the parameter space.
    SS_mean (numpy.ndarray): Mean of the summary statistics for normalization.
    SS_std (numpy.ndarray): Standard deviation of the summary statistics for normalization.
    Returns:
    numpy.ndarray var_theta: Empirical variance of the accepted parameter vectors.
    numpy.ndarray epsilon: Tolerance level given epsilon_percentile
    """

    theta_pilot = np.zeros((k_pilot, dim))  # Placeholder for parameter vectors from pilot run
    SS_pilot = np.zeros((k_pilot, len(SS_obs)))  # Placeholder for summary statistics from pilot run
    for k in range(k_pilot):
        n = 0
        while n<m:
            theta = abc_prior(1, l_bounds=l_bounds, u_bounds=u_bounds) #sample theta from the prior distribution
            x, y, metro_year = sample_data(theta, df_metro, T, 1) #sample data from the model using theta
            SS = compute_summary_statistics(x[0], T, cluster_bins, percentiles, df_metro, metro_year[0], dim) #compute summary statistics for the simulated data
            SS = (SS - SS_mean)/SS_std

            n = len(x[0]) #number of points in simulated data
        theta_pilot[k] = theta  # Store the accepted parameter vector
        SS_pilot[k] = SS  # Store the summary statistics for the accepted parameter vector

    #Lasso regression to find a and b for the linear model
    X = SS_pilot - SS_obs
    Y = theta_pilot
    a_array = np.zeros(dim)
    b_array = np.zeros((dim, len(SS_obs)))
    for d in range(dim):
        lasso = Lasso(alpha=lasso_penalty)  # Adjust alpha as needed
        lasso.fit(X, Y[:, d])
        a_array[d] = lasso.intercept_
        b_array[d] = lasso.coef_

    """
    # Center summary statistics
    X = SS_pilot - SS_obs
    Y = theta_pilot
    # Add intercept
    X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
    # Solve for regression coefficients for each parameter
    B, _, _, _ = np.linalg.lstsq(X_aug, Y, rcond=None)
    # B shape: (len(SS_obs)+1, dim)
    a_array = B[0, :]  # Intercepts
    b_array = B[1:, :].T  # Slopes, shape (dim, len(SS_obs))
    """

    theta_pilot_hat = a_array[np.newaxis, :] + X @ b_array.T
    # Compute empirical variance of the fitted parameter vectors from the pilot run
    var_theta = np.var(theta_pilot_hat, axis=0)
    # Compute the tolerance level based on the specified percentile
    epsilon = np.percentile([abc_distance(SS_pilot[i], SS_obs, a_array, b_array, var_theta) for i in range(k_pilot)], epsilon_percentile)  
    return var_theta, epsilon, a_array, b_array

def abc_rejection_sampling(k_abc, SS_obs, epsilon, m, dim, a_array, b_array, var_theta, l_bounds, u_bounds, df_metro, T, cluster_bins, percentiles, SS_mean, SS_std, max_iter = 1000):
    """
    Perform ABC rejection sampling to obtain parameter vectors that are close to the observed summary statistics.
    Parameters:
    k_abc (int): Number of ABC simulations to perform.
    SS_obs (numpy.ndarray): Observed summary statistics.
    epsilon (float): Tolerance level for accepting parameter vectors.
    m (int): Minimum number of points required.
    dim (int): Dimension of the parameter space.
    a_array (numpy.ndarray): Array of intercepts for the linear models.
    b_array (numpy.ndarray): Array of slopes for the linear models.
    var_theta (numpy.ndarray): Empirical variance of the fitted parameter vectors.
    l_bounds (numpy.ndarray): Lower bounds for the prior distribution.
    u_bounds (numpy.ndarray): Upper bounds for the prior distribution.
    df_metro (pandas.DataFrame): DataFrame containing metro data.
    T (int): Time parameter for the model.
    cluster_bins (list): List of cluster bins for summary statistics.
    percentiles (list): List of percentiles for summary statistics.
    SS_mean (numpy.ndarray): Mean of the summary statistics for normalization.
    SS_std (numpy.ndarray): Standard deviation of the summary statistics for normalization.
    max_iter (int): Maximum number of iterations to prevent infinite loops.
    """
    
    posterior_samples_theta = np.zeros((k_abc, dim)) # Placeholder for accepted parameter vectors from ABC rejection sampling
    posterior_samples_SS = np.zeros((k_abc, len(SS_obs))) # Placeholder for summary statistics corresponding to the accepted parameter vectors
    for k in range(k_abc):
        if k % (k_abc // 10) == 0:
            print(f"Simulation {k}/{k_abc}")
        distance = float('inf')  # Initialize distance to infinity
        iter_count = 0
        
        while distance > epsilon:  # Continue sampling until the distance is within the tolerance level
            iter_count += 1
            if iter_count > max_iter:
                raise RuntimeError("Maximum number of iterations reached in ABC rejection sampling.")
            n  = 0
            while n < m:
                theta = abc_prior(1, l_bounds=l_bounds, u_bounds=u_bounds) #sample theta from the prior distribution
                x, y, metro_year = sample_data(theta, df_metro, T, n = 1) #sample data from the model using theta
                SS = compute_summary_statistics(x[0], T, cluster_bins, percentiles, df_metro, metro_year[0], dim) #compute summary statistics for the simulated data
                SS = (SS - SS_mean)/SS_std
        
                n = len(x[0]) #number of points in simulated data
                # Compute the distance between the simulated summary statistics and the observed summary statistics
                distance = abc_distance(SS, SS_obs, a_array, b_array, var_theta) 
        posterior_samples_theta[k] = theta  # Store the accepted parameter vector
        posterior_samples_SS[k] = SS  # Store the summary statistics for the accepted parameter vector 
    return posterior_samples_theta, posterior_samples_SS