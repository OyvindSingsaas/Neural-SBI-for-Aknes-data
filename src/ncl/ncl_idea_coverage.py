import keras
from keras import layers, ops
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import os

import src.ncl.ncl_idea_utils as ncl_utils
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

print("Number of training data points:", len(params_train_normalized))

gs = len(SS_0_train_normalized_neural[0])
p = len(params_train_normalized[0])

print("\nLoading trained models...")
model = keras.models.load_model("neural_networks/idea_network/model.keras")
F_point = keras.models.load_model("neural_networks/idea_network/F_point.keras")
delta_net_path = "neural_networks/idea_network/delta_net.keras"
delta_net = keras.models.load_model(delta_net_path) if os.path.exists(delta_net_path) else None
curvature_head = model.get_layer("curvature_head")
print("Models loaded.")

# Empirical coverage using the trained model
S_test_pos = SS_0_test_normalized_neural[response_test == 1]
theta_test_pos = params_test_normalized[response_test == 1]
mle_F_point = ncl_utils.mle_point(S_obs=S_test_pos, F_point=F_point)
mle_F_class = ncl_utils.mle(S_obs=S_test_pos, F_point=F_point, delta_net=delta_net)
mle_F_point_denorm = mle_F_point.numpy() * params_std + params_mean
mle_F_class_denorm = mle_F_class.numpy() * params_std + params_mean
param_names = [str(n) for n in col_names_params_NS[:len(mle_F_point_denorm[0])]]

# Scatterplot of denormalized predictions vs. true parameters
true_denorm = theta_test_pos * params_std + params_mean
fig_sc, axes_sc = plt.subplots(1, p, figsize=(4 * p, 4))
if p == 1:
    axes_sc = [axes_sc]
for i, ax in enumerate(axes_sc):
    name = param_names[i] if i < len(param_names) else f"θ_{i}"
    vmin = min(true_denorm[:, i].min(), mle_F_point_denorm[:, i].min(), mle_F_class_denorm[:, i].min())
    vmax = max(true_denorm[:, i].max(), mle_F_point_denorm[:, i].max(), mle_F_class_denorm[:, i].max())
    ax.scatter(true_denorm[:, i], mle_F_point_denorm[:, i],
               alpha=0.4, s=15, color='steelblue', label='F_point')
    ax.scatter(true_denorm[:, i], mle_F_class_denorm[:, i],
               alpha=0.4, s=15, color='darkorange', label='F_class')
    ax.plot([vmin, vmax], [vmin, vmax], 'k--', linewidth=1.0, label='Ideal')
    ax.set_xlabel(f"True {name}", fontsize=11)
    ax.set_ylabel(f"Predicted {name}", fontsize=11)
    ax.set_title(name, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
fig_sc.suptitle("Predicted vs. true parameters (de-normalized)", fontsize=13)
fig_sc.tight_layout()
plt.savefig("results/idea_network/scatter_predictions.png", bbox_inches="tight")
plt.close()

rmse_point = np.sqrt(np.mean((mle_F_point_denorm - theta_test_pos * params_std - params_mean)**2, axis=0))
rmse_class = np.sqrt(np.mean((mle_F_class_denorm - theta_test_pos * params_std - params_mean)**2, axis=0))
print("\nRMSE")
for i, name in enumerate(col_names_params_NS[:len(rmse_point)]):
    print(f"  {name}: F_point RMSE = {rmse_point[i]:.4f}, F_class RMSE = {rmse_class[i]:.4f}")
param_names = [str(n) for n in col_names_params_NS]

results = ncl_utils.empirical_coverage(F_point, delta_net, curvature_head, S_test_pos, theta_test_pos,
                                       params_mean=params_mean, params_std=params_std)

ncl_utils.coverage_report(results, param_names=param_names)

diag = ncl_utils.coverage_diagnostics(results)
nominal = diag['nominal']
marginal_curve = diag['marginal_curve']   # (99, d)
joint_curve = diag['joint_curve']         # (99,)
d = marginal_curve.shape[1]

fig, axes = plt.subplots(1, d + 1, figsize=(4 * (d + 1), 4))

for i in range(d):
    ax = axes[i]
    ax.plot(nominal, marginal_curve[:, i], color='steelblue', linewidth=1.5, label='Empirical')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.0, label='Ideal')
    ax.set_xlabel("Nominal coverage", fontsize=11)
    ax.set_ylabel("Empirical coverage", fontsize=11)
    ax.set_title(param_names[i] if i < len(param_names) else f"θ_{i}", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

ax = axes[d]
ax.plot(nominal, joint_curve, color='darkorange', linewidth=1.5, label='Empirical (joint)')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.0, label='Ideal')
ax.set_xlabel("Nominal coverage", fontsize=11)
ax.set_ylabel("Empirical coverage", fontsize=11)
ax.set_title("Joint (Mahalanobis)", fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

fig.suptitle("Empirical coverage", fontsize=13)
fig.tight_layout()
plt.savefig("results/idea_network/coverage.png", bbox_inches="tight")
plt.close()


fisher = ncl_utils.fisher_information(curvature_head, S_test_pos, p=5).numpy()
print("Fisher diag mean:", np.diagonal(fisher, axis1=1, axis2=2).mean(axis=0))
