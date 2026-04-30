import numpy as np
import pandas as pd
import keras
import tensorflow as tf
import os

import src.ncl.ncl_idea_utils as ncl_utils

print("Loading data...")
data = np.load('data/NS_data_temp_wp_10RK_small.npz', allow_pickle=True)
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
col_names_params_NS = data['col_names_params_NS']

df_metro = pd.DataFrame(data['df_metro_values'],
                    columns=data['df_metro_columns'],
                    index=data['df_metro_index'])
print("Data loaded successfully.")

N_train = 500
gs = len(SS_0_train_normalized_neural[0])
p = len(l_bounds_NS)
print("Dimension of summary statistics:", gs)
print("Number of parameters:", p)

true_normalized = params_test_normalized[response_test == 1][:N_train, :]
true = true_normalized * params_std + params_mean
S_test = SS_0_test_normalized_neural[response_test == 1][:N_train, :]

print("\nLoading trained idea network models...")
model = keras.models.load_model("neural_networks/idea_network/model_small.keras")
F_point = keras.models.load_model("neural_networks/idea_network/F_point_small.keras")
delta_net_path = "neural_networks/idea_network/delta_net_small.keras"
delta_net = keras.models.load_model(delta_net_path) if os.path.exists(delta_net_path) else None
curvature_head = model.get_layer("curvature_head")
print("Models loaded.")

print("\nComputing MLE and covariance for each test observation...")
S_test_tf = tf.constant(S_test, dtype=tf.float32)
idea_mle_normalized, cov_ray = ncl_utils.mle_with_covariance(
    F_point, delta_net, curvature_head, S_test_tf
)
idea_mle_normalized = idea_mle_normalized.numpy()   # (N_train, p)
cov_ray = cov_ray.numpy()                           # (N_train, p, p)

idea_mle = idea_mle_normalized * params_std + params_mean

print("\nSaving the idea NCL MLE estimates and covariance matrices...")
os.makedirs("results/sim_study", exist_ok=True)
np.savez(
    "results/sim_study/idea_ncl_small.npz",
    idea_mle=idea_mle,
    idea_mle_normalized=idea_mle_normalized,
    cov_ray=cov_ray,
    N_train=N_train,
    true=true,
    true_normalized=true_normalized,
)
print("Saved to results/sim_study/idea_ncl_small.npz")
