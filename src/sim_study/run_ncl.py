import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
from tensorflow.keras.models import load_model
import utils.utils_surface_NS as us
import tensorflow as tf
tf.keras.backend.set_floatx("float32")
tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})  # workaround

def get_neg_hessian(theta_MLE, x_fixed, gs, classification_NN, p):
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

def get_neg_hessian(theta_MLE, x_fixed, gs, classification_NN, p):
    theta_MLE = tf.Variable(tf.convert_to_tensor(np.reshape(theta_MLE, (1, -1)), dtype=tf.float32))
    x_fixed = tf.convert_to_tensor(np.reshape(x_fixed, (1, gs)), dtype=tf.float32)

    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch(theta_MLE)
        with tf.GradientTape() as tape2:
            tape2.watch(theta_MLE)
            output = classification_NN(
                {"input_layer_1": x_fixed, "input_layer_2": theta_MLE},
                training=False
            )
        grad = tape2.gradient(output, theta_MLE)

    hessian = -tape1.jacobian(grad, theta_MLE, experimental_use_pfor=False)
    hessian = tf.reshape(hessian, (p, p))
    del tape1
    return hessian.numpy(), grad.numpy()[0]

def godambe_G_NS(N_G, theta_0, params_mean, params_std, gs, year, p, T, cluster_bins, percentiles, df_metro, classification_NN_NS, SS_mean, SS_std, plot = False):

    theta_0_repeated = np.repeat(np.reshape(theta_0, (1,-1)), N_G, axis = 0)
    theta_0_repeated_shape = np.zeros((theta_0_repeated.shape[0], theta_0_repeated.shape[1]+1))
    theta_0_repeated_shape[:, :-1] = theta_0_repeated
    theta_0_repeated_shape[:, -1] = np.repeat(year, theta_0_repeated.shape[0])

    SS_observed_bootstrap =  np.zeros((N_G, gs))
    X_boot, Y_boot, metro_year_list_train_boot, invalid_index_train = us.simulate_given_params(J=N_G, K=1, T=T, p=p, df_metro = df_metro, params=theta_0_repeated_shape, error = 0, verbose=False)

    for i in range(len(theta_0_repeated)):
        SS_observed_bootstrap[i,:] = us.summary_statistics(X_boot[i], T, cluster_bins, percentiles, df_metro, metro_year_list_train_boot[i], verbose=False, p = p)


    SS_observed_bootstrap_norm = (SS_observed_bootstrap - SS_mean) / SS_std

    H = np.zeros((p,p))
    J = np.zeros((p,p))
    U_ray = []

    theta_0_normalized = (theta_0 - params_mean)/params_std
    norm_change = []
    for n in range(N_G):
        h, U = get_neg_hessian(theta_0_normalized, SS_observed_bootstrap_norm[n], gs, classification_NN_NS, p)
        H += h
        J += np.outer(U, U)
        U_ray.append(U)
        if plot and n!=0:
            H_temp = H/n
            J_temp = J/n
            norm_change.append([np.trace(H_temp), np.trace(J_temp)])

    H = H/N_G
    J = J/N_G
    H_inv = np.linalg.inv(H)
    G = np.dot(np.dot(H_inv, J), H_inv)
    norm_change = np.array(norm_change)

    if plot:
        plt.figure()
        plt.plot(norm_change[:, 0], label = "trace(H)")
        plt.plot(norm_change[:, 1], label = "trace(J)")
        plt.legend()
        plt.show()
    return G, H


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
year = [2023]
gs = len(SS_0_train_normalized_neural[0])  # Dimension of summary statistics
p = len(l_bounds_NS)  # Number of parameters
print("Dimension of summary statistics:", gs)
true_normalized = params_test_normalized[response_test == 1][:N_train,:]
true = true_normalized * params_std + params_mean

print("\nLoading trained neural network for NCL...")
classification_NN_NS = load_model("neural_networks/classification_NN_NS.h5", compile=False)
print("Model loaded successfully.")
print("\nPredicting parameters using the neural network model...")
N_grid = 1000
N_G = 70
ncl_mle = np.zeros((N_train, len(l_bounds_NS)))
G_inv_NS_ray = []
H_neg_NS_ray = []
not_converged_index = []

grid = us.LHS(N_grid, l_bounds_NS, u_bounds_NS, year)[:,:-1]
grid_norm = (grid - params_mean) / params_std
for i in range(N_train):
    if i % (N_train//10) == 0:
        print("\n",round(i/N_train*100), "% done\n")

    #ll_grid = classification_NN_NS.predict([SS_0_test_normalized_neural[response_test==1][i].reshape((1,-1)).repeat(N_grid, axis = 0), grid_norm]).reshape(-1)  
    
    # replace .predict(...) inside loop
    x_obs = SS_0_test_normalized_neural[response_test == 1][i].reshape(1, -1).astype(np.float32)
    x_obs = np.repeat(x_obs, N_grid, axis=0)
    grid_norm = ((grid - params_mean) / params_std).astype(np.float32)

    ll_grid = classification_NN_NS(
        {"input_layer_1": x_obs, "input_layer_2": grid_norm},
        training=False
    ).numpy().reshape(-1)
    
    start = grid_norm[np.argmax(ll_grid)]
    MLE_NCL, MLE_NCL_normalized, final_logit, _, end_state = us.numerical_optim(start, SS_0_test_normalized_neural[response_test==1][i], l_bounds_NS, u_bounds_NS, classification_NN_NS, gs, params_mean=params_mean, params_std=params_std)
    
    if not end_state:
        not_converged_index.append(i)
    else:
        ncl_mle[i] = MLE_NCL
        G_inv_NS, H_neg_NS_mean = godambe_G_NS(N_G, MLE_NCL.reshape((1,-1)), params_mean, params_std, year = year[0], gs = gs, p = p, T = T, cluster_bins = cluster_bins, percentiles = percentiles, df_metro = df_metro, classification_NN_NS = classification_NN_NS, SS_mean = SS_mean, SS_std = SS_std, plot = False)
        G_inv_NS_ray.append(G_inv_NS)
        #Hessian
        H_neg_NS, _ = get_neg_hessian(MLE_NCL_normalized, SS_0_test_normalized_neural[response_test==1][i], gs, classification_NN_NS, p)
        H_neg_NS_ray.append(H_neg_NS)
print("\nNumber of non-converged optimizations:", len(not_converged_index))
ncl_mle_normalized = (ncl_mle - params_mean) / params_std
G_inv_NS_ray = np.array(G_inv_NS_ray)
H_neg_NS_ray = np.array(H_neg_NS_ray)
print("\nSaving the NCL MLE estimates...")
np.savez("results/sim_study/ncl.npz", ncl_mle=ncl_mle, ncl_mle_normalized=ncl_mle_normalized, N_train=N_train, true=true, true_normalized=true_normalized, G_inv_NS_ray=G_inv_NS_ray, H_neg_NS_ray=H_neg_NS_ray, not_converged_index=not_converged_index)