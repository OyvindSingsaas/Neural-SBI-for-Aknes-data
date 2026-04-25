
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from scipy.stats import uniform, beta
import numba
from numba import jit
import time
from scipy.stats import qmc
import tensorflow as tf
from scipy.stats import norm
import keras
import os
from scipy.stats import poisson



def read_cataloge(path):
    
    df = pd.DataFrame()
    entries = os.listdir(path)

    for entry in entries: 
        full_path = path + "/" + entry
        print(path + "/" + entry)
        try:
          df_temp = pd.read_csv(full_path, sep='\t')
          df_temp.Date = pd.DatetimeIndex(df_temp.Date)
          
        except FileNotFoundError as e:
          print(f"Error: {e}")
        
        df = pd.concat([df, df_temp], ignore_index=True)
    df = df.set_index("Date")
    
    return df
#---------------------------------------------------
# SIMULATE
def LHS(J,l_bounds, u_bounds, years):

    p = len(l_bounds)
    
    full_sample = np.zeros((J,p+1))

    metro_year = np.random.choice(years, size=J)

    params_sampler = qmc.LatinHypercube(d=p)
    params_sample = params_sampler.random(n=J)

    sample_scaled = qmc.scale(params_sample, l_bounds, u_bounds)

    full_sample[:, :-1] = sample_scaled
    full_sample[:, -1] =  metro_year

    return full_sample[:J]


def lambda_intensity(t, X_cov, beta, error = 0):

    t = int(t)
    log_lambda = beta[0]

    for i, x in enumerate(X_cov):
        log_lambda += x[t]*beta[i+1]

    #log_lambda = beta_0 + beta_1*x_1[t] + beta_2*x_2[t] + beta_3*x_3[t] + beta_4*x_4[t] + beta_5*x_5[t]

    log_lambda = log_lambda + np.random.normal(0, error)

    return np.exp(log_lambda)

def sample_parents(T, X_cov, beta, error = 0):
    
    parents_uniform = []
    #parents_rejection = []
    
    lambda_surface = np.array([[lambda_intensity(j, X_cov, beta, error = error) for j in range(T)]]).reshape((-1))
    lambda_max = np.max(lambda_surface)
    """
    while t < T:
    inter_event_time = np.random.exponential(1 / lambda_max)
    t += inter_event_time
    if t < T:
        parents_uniform.append(t)    
    """
    inter_event_times = np.random.exponential(1 / lambda_max, int(T * lambda_max * 4))  # Oversample to ensure we cover T
    event_times = np.cumsum(inter_event_times)
    parents_uniform = event_times[event_times < T]

    if event_times[-1] < T: print("Under-simulated")

    parents_uniform_ints = np.floor(parents_uniform).astype(int)
    u = np.random.uniform(size = len(parents_uniform))

    parents_rejection = parents_uniform[u < lambda_surface[parents_uniform_ints]/lambda_max]
    
    return parents_rejection 

#   MAKE IT A MIXTURE
def sample_N_k(n, delta):
    N_k = np.zeros(n, dtype=int)
    for i in range(n):
        N_k[i] = np.random.poisson(np.exp(delta))
    
    if np.sum(N_k == 0) == n and n > 0:
        N_k[np.random.randint(n)] = 1

    return N_k

def sample_offspring(mu_ray, N_k, sima_2):
    X = []
    for i,mu in enumerate(mu_ray):
        X.append(np.random.normal(mu, sima_2, size = N_k[i]))
        
    return np.concatenate(X)
    
def simulate_NS(T, X_cov, delta, sigma_2, beta, error = 0):
    mu_ray = sample_parents(T, X_cov, beta, error = error) #Parent locations
    if len(mu_ray) == 0:
        mu_ray = np.array([np.random.random(1)*T])

    n = mu_ray.shape[0] #Number of parent locations
    N_k = sample_N_k(n, delta) #Offspring per parent location
    X = sample_offspring(mu_ray, N_k, sigma_2) #Final offspring samples
    return np.sort(X), N_k, n, mu_ray

def covariate_formater(df_metro, metro_year, x_p, T):
        
        X_cov = np.zeros((x_p, T))
        df_temp = df_metro[df_metro.index.year.isin(metro_year)]

        X_cov[0] = df_temp.wp.values[:T]
        X_cov[1] = df_temp.temperature.values[:T]
        #X_cov[2] = df_temp.N_geophones.values[:T] #-np.log(0.01) + np.log(df_temp.N_geophones.values[:T] + 0.01) #

        #X_cov[0] = df_temp.doy_sin.values[:T]
        #X_cov[1] = df_temp.doy_cos.values[:T]

        return X_cov


def simulate_given_params(J, K, T, p, df_metro, params, error = 0, verbose = True):

    X_list = []
    metro_year_list = []
    Y = np.zeros((len(params)*K, p))
    #params = sample_theta(J, T)
    invalid_index =[]
    for j, param in enumerate(params):
        if verbose:
            print(j ,"/", len(params))
        metro_year = [int(param[-1])]
        #metro_year = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
        metro_year =[2023]
        #metro_year = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]


        X_cov = covariate_formater(df_metro, metro_year, x_p = p - 3, T = T)

        for k in range(K):
            beta = np.array([param[i] for i in range(2, len(param))]) 
            X, _, _, _ = simulate_NS(T, X_cov, delta = param[0], sigma_2 = np.exp(param[1]), beta = beta, error = error)
            X_list.append(X[X>0])
            metro_year_list.append(metro_year)
            Y[int(j*K + k)] = np.array([param[i] for i in range(len(param)-1)])

    return X_list, Y, metro_year_list, invalid_index

#---------------------------------------------------
#SUMMARY STATISTICS

@jit(nopython=True)
def inter_event_time_histogram_percentiles(X, percentiles):
    inter_event_times = np.diff(X)
    event_time_percentiles = np.percentile(inter_event_times, percentiles)
    event_time_mean_median_diff = np.mean(inter_event_times)/np.median(inter_event_times)
    return event_time_percentiles, event_time_mean_median_diff

def g_function(timestamps, cluster_bins):
    timestamps = np.sort(timestamps)
    if len(timestamps) < 2:
        return np.zeros_like(cluster_bins, dtype=float)

    # Compute nearest-neighbor distances
    diffs = np.diff(timestamps)
    left_dists = np.append(np.inf, diffs)
    right_dists = np.append(diffs, np.inf)
    nn_dist = np.minimum(left_dists, right_dists)

    # Compute empirical CDF over the bins
    g_vals = np.array([(nn_dist <= b).mean() for b in cluster_bins])
    return g_vals

def summary_statistics(X, T, cluster_bins, percentiles, df_metro, metro_year, verbose = 0, p = 2):

    if X.shape[0] < 2:
        return np.zeros([p +len(cluster_bins) + len(percentiles) - 1]) #!!!
    
    N = X.shape[0] #Number of events
    log_N_events = np.log(N)
    quantiles, mean_median = inter_event_time_histogram_percentiles(X, percentiles)

    #D = get_D(X)
    #if verbose: print("Got D")
    #Ripleys_k = ripleys_k_beta(X, D, ripley_bins, T)

    #G = g_function(X, cluster_bins)
    # t_max, max_dev
    RK = ripley_K(X, cluster_bins, T)

    #Normalize RK with respesct to the expected RK for a homogeneous Poisson process with the same number of events
    cluster_bins = np.asarray(cluster_bins)
    RK_homogeneous = RK/(2*cluster_bins)-1

    #temp = df_metro.loc[metro_year].temperature.values[:365]
    #N_g = df_metro.loc[metro_year].N_geophones.values[:365]
    #wp = np.log(df_metro.loc[metro_year].wp.values[:365] + 0.0001)

    X_cov = covariate_formater(df_metro, metro_year, p-3, T = T)

    N_day = np.histogram(X, bins=np.arange(T+1))[0]

    cov_sum = np.zeros(X_cov.shape[0])

    for i,x in enumerate(X_cov):
        cov_sum[i] = np.sum(x*N_day)/X.shape[0]

    #interaction = np.sum(X_cov[0]*X_cov[1]**N_day)
    #[t_max], [max_dev],
    theta = np.concatenate([[log_N_events], quantiles, np.array([mean_median]), RK_homogeneous,  [x for x in cov_sum]])
    
    return theta

#---------------------------------------------------

def boundary_penalty_sharp(z, l_bounds, u_bounds, penalty_strength=1.0, sharpness=10.0):
    distance_from_lower = tf.norm(tf.maximum(l_bounds - z, 0.0))  
    distance_from_upper = tf.norm(tf.maximum(z - u_bounds, 0.0))  

    penalty_lower = penalty_strength * tf.pow(distance_from_lower, sharpness)
    penalty_upper = penalty_strength * tf.pow(distance_from_upper, sharpness)

    total_penalty = penalty_lower + penalty_upper
    return total_penalty

def numerical_optim(start, x_ss, l_bounds, u_bounds, classification_NN, gs, params_mean, params_std, ridge_penalty = 0, verbose = False):

    path = []
    x_fixed = x_ss.reshape((1, gs))
    ridge_penalty = tf.constant(ridge_penalty, dtype=tf.float32)
    x_fixed = tf.convert_to_tensor(x_fixed, dtype=tf.float32)
    z_variable = tf.Variable(tf.ones(shape=(1,len(params_mean))) * start, trainable=True, dtype=tf.float32)

    l_bounds_normalized = (np.array(l_bounds) - params_mean) / params_std 
    u_bounds_normalized = (np.array(u_bounds) - params_mean) / params_std 

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01, use_ema = True)

    step = 0
    grad_norm = 1
    inside = True
    while step < 10000:  # Number of steps
        step +=1

        with tf.GradientTape() as tape:
            # Compute the output from the network
            output = classification_NN([x_fixed, z_variable])

        # Compute the gradient of the output w.r.t. z_variable
        gradients = tape.gradient(output, z_variable)

        #Update z_variable using the optimizer
        optimizer.apply_gradients([(-gradients, z_variable)])
        grad_norm = np.sum(gradients.numpy()*gradients.numpy())

        if not np.all((z_variable >= l_bounds_normalized) & (z_variable <= u_bounds_normalized)):
            if inside:
                inside = False
                end_state = False
                print("OUTSIDE")
        if np.all((z_variable >= l_bounds_normalized) & (z_variable <= u_bounds_normalized)):
            if not inside:
                inside = True
                print("Inside")

        # Optionally print progress
        if step % 100 == 0 and verbose:
            print(f"Step {step}, Output: {output.numpy()}, Gradient norm: {grad_norm}")
            #print("Penalty = ", penalty)
            #print(optimizer.learning_rate.numpy())
            path.append([np.array(z_variable[0])])

        if grad_norm < 1e-9:
            print("Grad good")
            H, _ = get_neg_hessian(z_variable.numpy(), x_fixed.numpy(), gs, len(params_mean), classification_NN)
            H = - H.numpy()
            H_sym = 0.5 * (H + H.T)
            eigvals = np.linalg.eigvalsh(H_sym)
            if np.all(eigvals < -1e-8):
                break
            else:
                z_variable.assign_add(tf.random.normal(shape = z_variable.shape, loc = 0, stdev = 0.1, dtype = tf.float32))
                print("good_grad")


    # Optimized z
    final_logit = classification_NN([x_fixed,  z_variable])
    z_final_numpy = np.array(z_variable)[0]
    z_final_un_normalized = z_final_numpy*params_std + params_mean
    print("Optimized z:", z_final_un_normalized)
    print("Gradient norm:", grad_norm)
    print("Inside:", inside)
    print(f"Steps: {step}")
    
    if grad_norm < 10**-8 and inside:
        end_state = True
    else:
        end_state = False
        
    return z_final_un_normalized, z_final_numpy, final_logit.numpy(), np.array(path).reshape((-1, len(l_bounds)))*params_std+params_mean, end_state

def numerical_optim_sequential(events, x, x_hist, l_bounds, u_bounds, classification_NN, gs, params_mean, params_std, initial, T, verbose = False):

    x_fixed = x.reshape((1, gs))
    x_fixed = tf.convert_to_tensor(x_fixed, dtype=tf.float32)
    
    x_histogram = tf.convert_to_tensor(x_hist, dtype=tf.float32)

    z_variable = initial

    l_bounds_normalized = (np.array(l_bounds) - params_mean) / params_std 
    u_bounds_normalized = (np.array(u_bounds) - params_mean) / params_std 
    l_bounds_tf = tf.constant([l_bounds_normalized], dtype=tf.float32)
    u_bounds_tf = tf.constant([u_bounds_normalized], dtype=tf.float32)

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01, use_ema = True)

    step = 0
    grad_norm = 1

    while step < 2000 and grad_norm > 10**-9:  # Number of steps
        step +=1
        with tf.GradientTape() as tape:
            output = classification_NN([x_fixed, x_histogram, z_variable])

        gradients = tape.gradient(output, z_variable)

        optimizer.apply_gradients([(-gradients, z_variable)])
        grad_norm = np.sqrt(np.sum(gradients.numpy()*gradients.numpy()))
        if step % 100 == 0 and verbose:
            print(f"Output: {output.numpy()}, Gradient norm: {(grad_norm):.7f}")        

        if tf.reduce_any((z_variable < l_bounds_tf) | (z_variable > u_bounds_tf)):
            print("Outside!")
            print(f"Last iteration: Output: {output.numpy()}, Gradient norm: {grad_norm:.7f}", ", Optim steps: ", step)
            return z_variable, False
            
    return z_variable, True

#---------------------------------------------------

def kernel_intensity_estimate(times, bandwidth):
    """ 
    Estimate intensity at t_eval points using Gaussian kernel. 
    times: event times (1D array)
    t_eval: time points to evaluate λ̂(t)
    bandwidth: kernel bandwidth h
    """
    n = len(times)
    t_eval = np.arange(0, T, dtype="float")
    intensity = np.zeros_like(t_eval)
    for t_i in times:
        intensity += norm.pdf(t_eval, loc=t_i, scale=bandwidth)
    return intensity / bandwidth

def kernel_intensity_estimate_reflect(times, bandwidth, domain=(0, 365)):
    t_eval = np.arange(domain[0], domain[1], dtype="float")

    # Reflect points only for bias correction — don't count them in the intensity total
    reflected = np.concatenate([
        2*domain[0] - times[times < domain[0] + 3*bandwidth],
        2*domain[1] - times[times > domain[1] - 3*bandwidth]
    ])
    all_times = np.concatenate([times, reflected])

    intensity = np.zeros_like(t_eval)
    for t_i in all_times:
        intensity += norm.pdf(t_eval, loc=t_i, scale=bandwidth)

    # Scale by number of original (not reflected) events
    return (len(times) / np.sum(intensity)) * intensity

#---------------------------------------------------
def shuffle_data(*arrays):
    assert all(len(arr) == len(arrays[0]) for arr in arrays)
    indices = np.random.permutation(len(arrays[0]))
    return (arr[indices] for arr in arrays)

def sequential_training_and_fit(J_new, K, max_iter, min_iter, delta_0, year, NN, year_dict, l_bounds, u_bounds,
                                 params_mean, params_std, years, X_mean, X_std, df_metro, gs, start_z, percentiles, cluster_bins, T):

    p = len(l_bounds)
    #ES =keras.callbacks.EarlyStopping("val_loss", patience=10, verbose = 1, restore_best_weights=True)
    
    converged = False
    converged_streak = 0
    l_bounds_temp = l_bounds
    u_bounds_temp = u_bounds
    
    bounds_ray = [[[l_bounds_temp[i], u_bounds_temp[i]] for i in range(p)]]
    delta = delta_0 + 1
    delta_ray = [delta_0]
    old_z = start_z
    delta_theta_ray = []

    x_histogram = np.zeros((1, T, 3)) 
    x_histogram[0, :, 0] = np.histogram(year_dict[year][0], bins = T)[0]
    x_histogram[0, :, 1] = np.abs(df_metro.loc[str(year)].temperature.values[:T])
    x_histogram[0, :, 2] = np.log(df_metro.loc[str(year)].wp.values[:T])

    iter_count = 0
    start_z_norm = (start_z - params_mean) / params_std
    z_variable = tf.Variable(tf.ones(shape=(1,len(params_mean)))*start_z_norm, trainable=True, dtype=tf.float32)
    path = [start_z]
    while (converged_streak < 10 and iter_count < max_iter) or iter_count < min_iter:
        print("\n ###############")
        print("Iteration", iter_count+1)
        iter_count +=1

        z_variable, converged = numerical_optim_sequential(year_dict[year][0], year_dict[year][2], x_histogram, l_bounds_temp, u_bounds_temp, NN, gs, params_mean, params_std, initial = z_variable, T=T, verbose = True)
        if converged:
            converged_streak +=1
        else:
            converged_streak = 0
        z_variable_0 = z_variable.numpy()[0]*params_std + params_mean
        delta_theta_ray.append(np.sum((old_z-z_variable_0)**2))
        old_z = z_variable_0
        closest_point = tf.clip_by_value(z_variable_0, l_bounds_temp, u_bounds_temp)
        path.append(closest_point)
        l_bounds_temp = closest_point - delta
        u_bounds_temp = closest_point + delta
        bounds_ray.append([[l_bounds_temp[i], u_bounds_temp[i]] for i in range(p)])
        delta = delta_0 * (0.9**iter_count) + 1
        delta_ray.append(delta)

        print("Converged?: ", converged, ", Streak: ", converged_streak, ", Change:", round(delta_theta_ray[-1], 4))
        print("Current point: ", z_variable_0)

        params_temp = LHS(J_new, l_bounds_temp, u_bounds_temp, [year])
        X_new, Y_new, metro_year_list_train_new, _ = simulate_given_params(J=J_new, K=K, T=365, p=p, df_metro = df_metro, params=params_temp, verbose = False)

        print("Mean number of events in new training data: ", round(np.mean([len(x) for x in X_new])), f"(Observed {len(year_dict[year][0])})")

        X_SS_new = np.zeros((len(X_new), gs))
        X_hist_new = np.zeros((len(X_new), T, 3))
        for i,x in enumerate(X_new):
            X_SS_new[i,:] = summary_statistics(x, T, cluster_bins, percentiles, df_metro, metro_year_list_train_new[i], verbose=False, p = p)
            X_hist_new[i, :, 0] = np.log1p(np.histogram(x, bins = T)[0])
            X_hist_new[i, :, 1] = np.abs(df_metro.loc[str(year)].temperature.values[:T])
            X_hist_new[i, :, 2] = np.log(df_metro.loc[str(year)].wp.values[:T])


        Y_shuffled = params_temp[:, :-1]
        np.random.shuffle(Y_shuffled)
        Y_shuffled = Y_shuffled.repeat(K, axis = 0)
        params_full = np.concatenate([Y_new, Y_shuffled])
        response_full = np.concatenate([np.repeat(1, len(Y_new)), np.repeat(0, len(Y_new))])

        X_SS_full = np.concatenate([X_SS_new, X_SS_new])
        X_hist_new = np.concatenate([X_hist_new, X_hist_new])

        response_full, params_full, X_SS_full, X_hist_new = shuffle_data(response_full, params_full, X_SS_full, X_hist_new)

        # Normalize the training sets
        params_full_normalized = (params_full - params_mean) / params_std
        X_SS_full_normalized = (X_SS_full - X_mean) / X_std

        NN.fit([X_SS_full_normalized, X_hist_new, params_full_normalized], response_full, epochs=5, batch_size=32, verbose = 0)
        
    print("\n ############### \n OUT OF LOOP")

    if converged_streak > 2:
        print("CONVERGED!!!!")
        return z_variable.numpy(), z_variable_0, np.array(bounds_ray), NN, np.array(path), delta_ray, delta_theta_ray
    else:
        print("Misson Failed. Last atempt")


    z_variable, converged = numerical_optim_sequential(year_dict[year][0], year_dict[year][2], x_histogram, l_bounds_temp, u_bounds_temp, NN, gs, params_mean, params_std, initial = z_variable, T = T, verbose = True)
    z_variable_0 = z_variable.numpy()[0]*params_std + params_mean
    path.append(z_variable_0)
    closest_point = tf.clip_by_value(z_variable_0, l_bounds_temp, u_bounds_temp)
    l_bounds_temp = closest_point - delta
    u_bounds_temp = closest_point + delta
    delta = delta_0 * (0.9**iter_count) + 0.5
    delta_ray.append(delta)
    bounds_ray.append([[l_bounds_temp[i], u_bounds_temp[i]] for i in range(p)])
    return z_variable.numpy(), z_variable_0, np.array(bounds_ray), NN, np.array(path), delta_ray, delta_theta_ray
    
#---------------------------------------------------
#NHPP

def compute_likelihood(beta, X, Y):

    # Compute the integral of the intensity function
    intensity_integral = np.sum(np.exp(beta[0] + np.sum([beta[i+1]*X[i] for i in range(len(X))], axis = 0)))
    
    # Compute the product of intensities at event times
    event_intensities = np.sum(beta[0] + np.sum([beta[i+1]*X[i][Y.astype(int)] for i in range(len(X))], axis = 0))  # Log-sum instead of product
    
    # Compute log-likelihood
    log_likelihood = -intensity_integral + event_intensities
    
    return log_likelihood 


def negative_log_likelihood(params, X, Y):
    return -compute_likelihood(params, X, Y)


#---------------------------------------------------


def ripley_K(events, evals, T):
    events = np.sort(events)
    n = len(events)
    K_vals = np.zeros(len(evals))
    
    for idx, r in enumerate(evals):
        left = np.searchsorted(events, events - r, side='left')
        right = np.searchsorted(events, events + r, side='right')
        counts = right - left - 1
        K_vals[idx] = T * np.sum(counts) / (n * (n - 1))
        
    return K_vals

def ripley_K_1D_SS_test(events, evals, T):

    events = np.sort(events)
    n = len(events)
    K_vals = np.zeros(len(evals))
    
    for idx, t in enumerate(evals):
        # for each event, count how many events are within distance t ahead
        j = np.searchsorted(events, events + t, side='right')
        counts = j - np.arange(n) - 1  # subtract self
        K_vals[idx] = np.sum(counts) / (n * (n - 1))
    """
    K_poisson = 2 * np.array(evals)
    deviations = np.abs(K_vals - K_poisson)
    max_idx = np.argmax(deviations)
    t_max = evals[max_idx]
    max_dev = deviations[max_idx]
    return K_vals, t_max, max_dev
    """
    return K_vals



def poisson_envelope(events, max_dist, n_sim=100, n_eval=100):
    T = events[-1] - events[0]
    t0, t1 = events[0], events[-1]
    lambda_hat = len(events) / T
    K_sim = []

    for _ in range(n_sim):
        n_sim_pts = np.random.poisson(lambda_hat * T)
        sim = np.sort(np.random.uniform(t0, t1, n_sim_pts))
        _, K = ripley_K_1D(sim, max_dist, n_eval)
        K_sim.append(K)

    K_sim = np.array(K_sim)
    lower = np.percentile(K_sim, 2.5, axis=0)
    upper = np.percentile(K_sim, 97.5, axis=0)
    return lower, upper

def simulate_nhpp(lambda_intensity):
    T = len(lambda_intensity)
    events = []
    for t in range(T):
        rate = lambda_intensity[t]
        n_events = np.random.poisson(rate)
        times = np.random.uniform(t, t + 1, size=n_events)
        events.extend(times)
    return np.sort(np.array(events))

def NHPP_envelope(max_dist, lambda_ray,n_sim=100, n_eval=100):
    K_sim = []

    for _ in range(n_sim):
        sim = simulate_nhpp(lambda_ray)
        _, K = ripley_K_1D(sim, max_dist, n_eval)
        K_sim.append(K)

    K_sim = np.array(K_sim)
    lower = np.percentile(K_sim, 2.5, axis=0)
    upper = np.percentile(K_sim, 97.5, axis=0)
    return lower, upper

def NS_envelope(max_dist, params_fitted, T, p, df_metro, n_sim=100, n_eval=100):
    K_sim = []
    sim, _, _, _ = simulate_given_params(J=n_sim, K=1, T=T, p=p, df_metro = df_metro, params=params_fitted, verbose = False)
    for s in sim:
        _, K = ripley_K_1D(s, max_dist, n_eval)
        K_sim.append(K)

    K_sim = np.array(K_sim)
    lower = np.percentile(K_sim, 2.5, axis=0)
    upper = np.percentile(K_sim, 97.5, axis=0)
    return lower, upper

def vmr_test(events, bins=20):
    """Variance-to-mean ratio"""
    counts, _ = np.histogram(events, bins=bins)
    mean = np.mean(counts)
    var = np.var(counts, ddof=1)
    vmr = var / mean
    return vmr, var, mean

def get_neg_hessian(theta_MLE, x_fixed, gs, p, classification_NN):
    theta_MLE = tf.Variable(tf.reshape(theta_MLE, (1, -1)))
    x_fixed = x_fixed.reshape((1, gs))
    x_fixed = tf.convert_to_tensor(x_fixed, dtype=tf.float32)

    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch(theta_MLE)
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(theta_MLE)  
            output = classification_NN([x_fixed, theta_MLE])
            #neg_output = -output  

        grad = tape2.gradient(output, theta_MLE)  # Gradient w.r.t theta_MLE
        #print("Gradient:", grad)

    hessian = -tape1.jacobian(grad, theta_MLE)  # Hessian w.r.t theta_MLE
    hessian = tf.reshape(hessian, (p, p))
    del tape1

    #print("Hessian: \n", hessian.numpy())
    return hessian, grad.numpy()[0]






import numpy as np
from matplotlib.patches import Ellipse
from scipy.stats import chi2

def plot_gaussian_ellipse(mean, cov=None, precision=None, p=0.95, ax=None, **kwargs):
    """
    Plot the contour ellipse of a 2D Gaussian on a given Matplotlib axis.

    Parameters
    ----------
    mean : array-like of shape (2,)
        Mean of the Gaussian (x, y).
    cov : array-like of shape (2, 2), optional
        Covariance matrix.
    precision : array-like of shape (2, 2), optional
        Precision matrix (inverse covariance). One of `cov` or `precision` must be given.
    p : float, default=0.95
        Probability level for the ellipse (e.g. 0.95 for a 95% contour).
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, uses current axis.
    **kwargs :
        Passed to `matplotlib.patches.Ellipse` (e.g., edgecolor, lw, linestyle).
    """
    if (cov is None) == (precision is None):
        raise ValueError("Provide exactly one of `cov` or `precision`.")

    cov = np.asarray(cov if cov is not None else np.linalg.inv(precision))
    mean = np.asarray(mean)

    # Eigen-decomposition
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    # Confidence scaling factor
    k = np.sqrt(chi2.ppf(p, 2))
    width, height = 2 * k * np.sqrt(vals)
    angle = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    ell = Ellipse(xy=mean, width=width, height=height, angle=angle,
                  facecolor='none', **kwargs)

    ax = ax or plt.gca()
    ax.add_patch(ell)
    return ell