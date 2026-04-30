import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull
from jaxopt import ScipyMinimize
import optax
import jax
from jax import numpy as jnp
import keras
from keras.layers import Dense, Flatten, Input, Concatenate,Conv1D, Conv2D, Dropout, MaxPooling2D, MaxPooling1D
from keras.models import Model, Sequential
from keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
from sklearn.metrics import mean_squared_error

import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve
from scipy.stats import chi2
import scipy.stats
from sklearn.linear_model import LogisticRegression
import utils.utils_surface_NS as us
from sklearn.cross_decomposition import PLSRegression
from tensorflow.keras.models import load_model


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

print("\nLoading trained neural network for NCL...")
classification_NN_NS = load_model("neural_networks/classification_NN_NS_small.h5", compile=False)
print("Model loaded successfully.")
print("\nPredicting parameters using the neural network model...")

J_platt = 5000
K = 3
T = 1*365
error = 0.0
#log(delta), log(sigma^2), beta_0, beta_1, beta_2, beta_3, beta_4, beta_5, beta_6
#l_bounds_NS = np.array([np.log(0.1), np.log(0.001), -1.5, -1.5, -1.5])
#u_bounds_NS = np.array([np.log(10),  np.log(5),  1.5, 1.5, 1.5])

p = len(l_bounds_NS)
years = df_metro.index.year.unique().values[1:-1]
years = years[years!=2019]
years = [2023]

params_sample_platt_NS = us.LHS(J_platt, l_bounds_NS_test, u_bounds_NS_test, years)

print("Simulating data for training and test sets...")
X_platt, Y_platt, metro_year_list_platt, invalid_index_train = us.simulate_given_params(J=J_platt, K=K, T=T, p=p, df_metro = df_metro, params=params_sample_platt_NS, error = error, verbose=False)
print("Max number of events in a year = ", np.max([x.shape[0] for x in X_platt]))
print("Min number of events in a year = ", np.min([x.shape[0] for x in X_platt]))
print("Data shape = ", len(X_platt))


gs = len(l_bounds_NS) + len(cluster_bins) + len(percentiles) - 1 
SS_0_platt = np.zeros((len(X_platt), gs))

print("Computing summary statistics for Platt set...")
for i,x in enumerate(X_platt):
    SS_0_platt[i,:] = us.summary_statistics(x, T, cluster_bins, percentiles, df_metro, metro_year_list_platt[i], verbose=False, p = p)
    #print every 10 percent completed
    if i % (len(X_platt) // 10) == 0:
        print(round(i/len(X_platt)*100), "% done")



SS_0_platt_normalized = (SS_0_platt - SS_mean) / SS_std

Y_shuffled_platt = params_sample_platt_NS[:, :-1].copy()
np.random.shuffle(Y_shuffled_platt)
Y_shuffled_platt = Y_shuffled_platt.repeat(K, axis = 0)
params_platt = np.concatenate([Y_platt, Y_shuffled_platt])
params_platt_unshuffled_copy = params_platt.copy()
response_platt = np.concatenate([np.repeat(1, len(Y_platt)), np.repeat(0, len(Y_platt))])

SS_0_platt_normalized_neural = np.concatenate([SS_0_platt_normalized, SS_0_platt_normalized])

print(SS_0_platt_normalized_neural.shape)

#--------------------
params_platt_not_shuffled = params_sample_platt_NS[:, :-1].copy()
SS_0_platt_normalized_not_shuffled = SS_0_platt_normalized.copy()
    
response_platt, params_platt, SS_0_platt_normalized_neural = us.shuffle_data(response_platt, params_platt, SS_0_platt_normalized_neural)

params_platt_normalized = (params_platt - params_mean) / params_std

response_platt_pred_logits = classification_NN_NS.predict([SS_0_platt_normalized_neural, params_platt_normalized]).reshape((-1))

response_platt_pred_probs = 1 / (1 + np.exp(-response_platt_pred_logits))

# Confusion matrix
response_platt_pred = (response_platt_pred_probs >= 0.5).astype(int)


n_samples = response_platt_pred_probs.shape[0]
logit_h = response_platt_pred_logits

X_platt_ = logit_h.reshape(-1, 1)
model_calibration = LogisticRegression()
model_calibration.fit(X_platt_, response_platt) 
platt_scaler = model_calibration.coef_[0][0]
#save the Platt scaling factor
np.save("neural_networks/platt_scaler_NS_small_new.npy", platt_scaler)
print(f"Platt scaling factor for NCL confidence intervals: {platt_scaler}")
platt_scaler_old = np.load("neural_networks/platt_scaler_NS_small.npy", allow_pickle=True)
print("\n Old Platt scaling factor for NCL confidence intervals: ", platt_scaler_old)
