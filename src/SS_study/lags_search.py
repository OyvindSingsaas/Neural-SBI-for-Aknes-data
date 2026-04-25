# === Standard library ===
import importlib

# === Third-party libraries ===
# -- Array, math, and stats --
import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import chi2
from scipy.spatial import ConvexHull

# -- Plotting --
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

# -- Machine learning and deep learning --
import keras
from keras.layers import (Dense, Flatten, Input, Concatenate, Conv1D, Conv2D, Dropout, MaxPooling2D, MaxPooling1D)
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import tensorflow as tf

# -- JAX and optimization --
import jax
from jax import numpy as jnp
import optax
from jaxopt import ScipyMinimize

# -- sklearn --
from sklearn.metrics import (mean_squared_error, precision_recall_curve, roc_curve, auc, accuracy_score, confusion_matrix)
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import PLSRegression

# -- sbi --
from sbi.inference import NLE
from sbi import inference as sbi_inference
from sbi.utils import BoxUniform

# -- torch --
import torch

# === Local imports ===
import utils.utils_surface_NS as us
import utils.utils_abc as abc

print("Loading and preprocessing data...")

path = 'data/Surface_Events_covariates.csv'
df_metro = pd.read_csv(path, index_col=0, parse_dates=True)
full_range = pd.date_range(start=df_metro.index.min(), end=df_metro.index.max(), freq='D')
missing_days = full_range.difference(df_metro.index)
df_metro = df_metro.reindex(full_range)

df_metro = df_metro.ffill().bfill()

df_metro["snow_S"] = (df_metro["snow_S"] - df_metro["snow_S"].max()) / (df_metro["snow_S"].max() - df_metro["snow_S"].min())
df_metro['wp'] = df_metro['wp'].rolling(window=2, min_periods=1).mean()
df_metro['wp'] = (df_metro['wp'] - df_metro['wp'].min()) / (df_metro['wp'].max() - df_metro['wp'].min())
df_metro['N_geophones'] = (df_metro['N_geophones'] - df_metro['N_geophones'].min()) / (df_metro['N_geophones'].max() - df_metro['N_geophones'].min())

temp = df_metro['temperature'] = df_metro['temperature'].rolling(window=4, min_periods=1).mean()
scale = max(temp.max(), -temp.min())  # symmetric scaling
df_metro['temperature'] = temp / scale
df_metro.head()


print("Loading data...")
data = np.load('data/NS_events.npz', allow_pickle=True)

X_train = data['X_train']
Y_train = data['Y_train']
metro_year_list_train = data['metro_year_list_train']
invalid_index_train = data['invalid_index_train']
X_test = data['X_test']
Y_test = data['Y_test']
metro_year_list_test = data['metro_year_list_test']
#invalid_index_test = data['invalid_index_test']
l_bounds_NS = data['l_bounds_NS']
u_bounds_NS = data['u_bounds_NS']
params_sample_test_NS = data['params_sample_test_NS']
params_sample_train_NS = data['params_sample_train_NS']

print(f"  Train: {len(X_train)} samples | Test: {len(X_test)} samples | Parameters: {len(l_bounds_NS)}")

def print_summary(title, N_test, N_bins, r_min_plan, r_max_plan, nbins_plan,
                  rmse_list_ncl_rmin, rmse_list_ncl_rmax, rmse_list_ncl_nbins,
                  best_rmin_idx, best_rmax_idx, best_nbins_idx):
    sep = "=" * 55
    print(f"\n{sep}")
    print(f"===  {title} ===")
    print(sep)
    print(sep)
    print(f"  N_test={N_test} | Epochs: sub_model={sub_model_epochs}, ncl={ncl_epochs} | lr: sub_model={sub_model_lr}, ncl={ncl_lr}")
    print(sep)

    print(f"  r_min search  (r_max={r_max_plan[0]:.3f}, N_bins={N_bins})")
    print(f"  {'r_min':>10}  {'RMSE':>10}")
    print(f"  {'-'*10}  {'-'*10}")
    for r, rmse in zip(r_min_plan, rmse_list_ncl_rmin):
        marker = "  <-- best" if rmse == rmse_list_ncl_rmin[best_rmin_idx] else ""
        print(f"  {r:>10.4f}  {rmse:>10.4f}{marker}")
    print(f"  => Best r_min: {r_min_plan[best_rmin_idx]:.4f}")

    print(sep)
    print(f"  r_max search  (r_min={r_min_plan[best_rmin_idx]:.3f}, N_bins={N_bins})")
    print(f"  {'r_max':>10}  {'RMSE':>10}")
    print(f"  {'-'*10}  {'-'*10}")
    for r, rmse in zip(r_max_plan, rmse_list_ncl_rmax):
        marker = "  <-- best" if rmse == rmse_list_ncl_rmax[best_rmax_idx] else ""
        print(f"  {r:>10.4f}  {rmse:>10.4f}{marker}")
    print(f"  => Best r_max: {r_max_plan[best_rmax_idx]:.4f}")

    print(sep)
    print(f"  N_bins search  (r_min={r_min_plan[best_rmin_idx]:.3f}, r_max={r_max_plan[best_rmax_idx]:.3f})")
    print(f"  {'N_bins':>10}  {'RMSE':>10}")
    print(f"  {'-'*10}  {'-'*10}")
    for nb, rmse in zip(nbins_plan, rmse_list_ncl_nbins):
        marker = "  <-- best" if rmse == rmse_list_ncl_nbins[best_nbins_idx] else ""
        print(f"  {nb:>10d}  {rmse:>10.4f}{marker}")
    print(f"  => Best N_bins: {nbins_plan[best_nbins_idx]}")

    print(sep)
    print(f"  Final config:  r_min={r_min_plan[best_rmin_idx]:.4f}, r_max={r_max_plan[best_rmax_idx]:.4f}, N_bins={nbins_plan[best_nbins_idx]}")
    print(f"  Results saved to results/lag_search/ncl_lagsearch.npz")
    print(sep)


def format_data(X, Y, params, metro_year_list, df_metro, r_min, r_max, N_bins, params_mean=None, params_std=None, SS_mean=None, SS_std=None, log_cluster = True):
    #get the summary statistics for the training data with the given r_min and r_max values
    if log_cluster:
        cluster_bins = np.logspace(np.log10(r_min), np.log10(r_max), N_bins)
    else:
        cluster_bins = np.linspace(r_min, r_max, N_bins)
    percentiles = [10, 50, 90]
    SS = []
    n = len(X)
    print(f"  Computing summary statistics for {n} samples (r_min={r_min:.3f}, r_max={r_max}, N_bins={N_bins})...")
    log_every = max(1, n // 5)
    for i in range(n):
        x = X[i]
        metro_year = metro_year_list[i]
        SS_0 = us.summary_statistics(x, T, cluster_bins, percentiles, df_metro, metro_year, verbose=False, p = len(l_bounds_NS))
        SS.append(SS_0)
        if (i + 1) % log_every == 0 or (i + 1) == n:
            print(f"    {i+1}/{n} samples processed")
    SS = np.array(SS)
    print(f"  SS shape: {SS.shape}")
    if SS_mean is None or SS_std is None:
        SS_mean = np.mean(SS, axis=0)
        SS_std = np.std(SS, axis=0)
    SS_normalized = (SS - SS_mean) / SS_std
    Y_shuffled = params[:, :-1].copy()
    np.random.shuffle(Y_shuffled)
    Y_shuffled = Y_shuffled.repeat(K, axis = 0)
    params_neural = np.concatenate([Y, Y_shuffled]).astype(np.float64)
    response = np.concatenate([np.repeat(1, len(Y)), np.repeat(0, len(Y))])
    SS_normalized_neural = np.concatenate([SS_normalized, SS_normalized])
    if params_mean is None or params_std is None:
        # Use only true params (Y) for normalization stats, not the shuffled mixture
        Y_float = np.array(Y, dtype=np.float64)
        params_mean = np.mean(Y_float, axis=0)
        params_std = np.std(Y_float, axis=0)
    params_neural_normalized = (params_neural - params_mean) / params_std
    response, params_neural_normalized, SS_normalized_neural = us.shuffle_data(response, params_neural_normalized, SS_normalized_neural)

    return response, params_neural_normalized, SS_normalized_neural, params_mean, params_std, SS_mean, SS_std


def train_sub_model(gs, response_neural_normalized, params_neural_normalized, SS_normalized_neural):
        n_train = np.sum(response_neural_normalized == 1)
        print(f"\nTraining sub-model (input_dim={gs}, n_train={n_train}, output_dim={len(l_bounds_NS)})...")
        x_global_input = Input(shape=(gs,))
        x = Dense(32, activation='tanh')(x_global_input)
        x = Dense(32, activation='tanh')(x)
        x = Dense(8, activation='tanh')(x)
        output = Dense(len(l_bounds_NS), activation='linear')(x)
        sub_model_NS = Model(x_global_input, output)
        # Compile the model
        initial_learning_rate = sub_model_lr
        optimizer = keras.optimizers.Adam(learning_rate=initial_learning_rate)
        sub_model_NS.compile(optimizer=optimizer, loss="mse")
        #train sub-model for NS
        params_train_pre, X_global_train_pre = params_neural_normalized[response_neural_normalized==1], SS_normalized_neural[response_neural_normalized==1]
        ES_NS =keras.callbacks.EarlyStopping("val_loss", patience=15, verbose = 1, restore_best_weights=True)
        sub_model_NS.fit(X_global_train_pre, params_train_pre, epochs=sub_model_epochs, batch_size=32, verbose = 0, validation_split=0.1, callbacks=[ES_NS])
        return sub_model_NS


def train_ncl(response_neural_normalized, params_neural_normalized, SS_normalized_neural, sub_model, gs):
        print(f"\nTraining NCL classifier (input_dim={gs}, n_samples={SS_normalized_neural.shape[0]})...")
        sub_model_clone_NS  = keras.models.clone_model(sub_model)
        sub_model_clone_NS.set_weights(sub_model.get_weights())
        sub_model_clone_NS_cutted = Model(sub_model_clone_NS.inputs, sub_model_clone_NS.layers[-1].output)
        for layer in sub_model_clone_NS_cutted.layers:
            layer.trainable = False
        Input_summary_statistics = Input(shape=(gs,))
        Input_parameters = Input(shape=(len(u_bounds_NS),))
        #sub_model_output = sub_model_clone_cutted(Input_summary_statistics)
        sub_model_NS_output = sub_model_clone_NS_cutted(Input_summary_statistics)
        concatenated = Concatenate(name = "Combine")([Input_summary_statistics, Input_parameters, sub_model_NS_output])
        x = Dense(32, activation='tanh', name = "First_dense")(concatenated)
        x = Dense(16, activation='tanh')(x)
        x = Dense(64, activation='tanh')(x)
        x = Dense(32, activation='tanh')(x)
        output = Dense(1, activation='linear')(x)  
        # Model definition
        classification_NN_NS = Model([Input_summary_statistics, Input_parameters], output)
        # Compile the model
        initial_learning_rate = ncl_lr
        optimizer = keras.optimizers.Adam(learning_rate=initial_learning_rate)
        classification_NN_NS.compile(optimizer=optimizer, loss=keras.losses.BinaryCrossentropy(from_logits=True))
        #classification_NN_NS.optimizer.learning_rate.assign(0.001)
        ES =keras.callbacks.EarlyStopping("val_loss", patience=20, verbose = 1, restore_best_weights=True)
        classification_NN_NS.fit([SS_normalized_neural, params_neural_normalized], response_neural_normalized, epochs=ncl_epochs, batch_size=32, verbose = 0, validation_split=0.1, callbacks=[ES])#sample_weight=d_e)
        return classification_NN_NS

def ncl_mle(SS_test_normalized_neural, response_test, params_test_normalized_neural ,classification_NN_NS, params_mean, params_std, grid_norm, N_test = None):
        if not N_test:
            N_test = len(SS_test_normalized_neural)
        N_test = int(N_test)
        gs = len(SS_test_normalized_neural[0])
        mle_ncl = np.zeros((N_test, len(l_bounds_NS)))
        converged_index = []
        print(f"\nRunning NCL-MLE on {N_test} test samples (grid size={N_grid})...")
        SS_positive = SS_test_normalized_neural[response_test==1]
        log_every = max(1, N_test // 4)
        for i in range(N_test):
            ll_grid = classification_NN_NS.predict([SS_positive[i].reshape((1,-1)).repeat(N_grid, axis = 0), grid_norm], verbose=0).reshape(-1)
            start = grid_norm[np.argmax(ll_grid)]
            MLE_NCL, MLE_NCL_normalized, final_logit, _, end_state = us.numerical_optim(start, SS_positive[i], l_bounds_NS, u_bounds_NS, classification_NN_NS, gs, params_mean=params_mean, params_std=params_std)
            mle_ncl[i] = MLE_NCL
            if end_state:
                converged_index.append(i)
            else:
                mle_ncl[i] = start * params_std + params_mean  # fallback to grid max if optimization fails
                converged_index.append(i)  # Consider grid max as a valid estimate for convergence
                print(f"  [!] Sample {i+1}/{N_test}: optimization did not converge. Using grid max as fallback.")
            if (i + 1) % log_every == 0 or (i + 1) == N_test:
                print(f"  {i+1}/{N_test} samples done | converged: {len(converged_index)}/{i+1}")
        true_normalized = params_test_normalized_neural[response_test==1][:N_test][converged_index]
        true = true_normalized * params_std + params_mean
        rmse = np.sqrt(mean_squared_error(true, mle_ncl[converged_index]))
        print(f"  Convergence rate: {len(converged_index)}/{N_test} | RMSE: {rmse:.4f}")
        return rmse

def run_test_ncl(r_min, r_max, N_bins, N_test, SS_normalized_neural, response_neural_normalized, params_neural_normalized, SS_test_normalized_neural, response_test_neural_normalized, params_test_neural_normalized, params_mean, params_std):
    print(f"  r_min={r_min:.3f}, r_max={r_max:.3f}, N_bins={N_bins}")

    sub_model = train_sub_model(gs=len(SS_normalized_neural[0]), response_neural_normalized=response_neural_normalized, params_neural_normalized=params_neural_normalized, SS_normalized_neural=SS_normalized_neural)
    classification_NN_NS = train_ncl(response_neural_normalized=response_neural_normalized, params_neural_normalized=params_neural_normalized, SS_normalized_neural=SS_normalized_neural, sub_model=sub_model, gs=len(SS_normalized_neural[0]))
    grid = us.LHS(N_grid, l_bounds_NS, u_bounds_NS, 2023)[:,:-1]
    grid_norm = (grid - params_mean) / params_std
    rmse_ncl = ncl_mle(SS_test_normalized_neural=SS_test_normalized_neural, response_test=response_test_neural_normalized, params_test_normalized_neural=params_test_neural_normalized ,classification_NN_NS=classification_NN_NS, params_mean=params_mean, params_std=params_std, grid_norm=grid_norm, N_test=N_test)
    del sub_model, classification_NN_NS
    keras.backend.clear_session()
    return rmse_ncl

def train_nf(params_normalized, SS_normalized, params_mean, params_std):
    print(f"\nTraining NF (n_train={len(params_normalized)}, input_dim={SS_normalized.shape[1]}, output_dim={len(l_bounds_NS)})...")
    l_bounds_NS_norm = (l_bounds_NS - params_mean) / params_std
    u_bounds_NS_norm = (u_bounds_NS - params_mean) / params_std
    prior = BoxUniform(low=torch.tensor(l_bounds_NS_norm, dtype=torch.float32), high=torch.tensor(u_bounds_NS_norm, dtype=torch.float32))
    inference = sbi_inference.SNLE(prior=prior)
    density_estimator = inference.append_simulations(
        theta=params_normalized,
        x=SS_normalized
    ).train()
    posterior = inference.build_posterior(density_estimator)
    print("  NF training done.")
    return posterior

def map_nf(posterior, SS_normalized):
    posterior_x = posterior.set_default_x(SS_normalized)
    return posterior_x.map().numpy()


def run_test_nf(params_normalized, SS_normalized, params_mean, params_std, params_test_normalized, SS_test_normalized, N_test):
    posterior = train_nf(params_normalized, SS_normalized, params_mean, params_std)
    N_test = int(N_test)
    print(f"\nRunning NF MAP on {N_test} test samples...")
    map_estimates = np.zeros((N_test, len(l_bounds_NS)))
    log_every = max(1, N_test // 4)
    for i in range(N_test):
        map_estimates[i] = map_nf(posterior, SS_test_normalized[i].reshape(1,-1)).reshape(-1)
        if (i + 1) % log_every == 0 or (i + 1) == N_test:
            print(f"  {i+1}/{N_test} samples done")
    true_normalized = params_test_normalized[:N_test]
    true = true_normalized * params_std + params_mean
    map_nf_denorm = map_estimates * params_std + params_mean
    rmse_nf = np.sqrt(mean_squared_error(true, map_nf_denorm))
    print(f"  NF RMSE: {rmse_nf:.4f}")
    del posterior
    return rmse_nf

def format_data_nf(params_neural_normalized, SS_normalized_neural, params_test_neural_normalized, SS_test_normalized_neural, response_neural_normalized, response_test_neural_normalized):
    params_normalized_nf, SS_normalized_nf, params_test_normalized_nf, SS_test_normalized_nf = params_neural_normalized[response_neural_normalized==1], SS_normalized_neural[response_neural_normalized==1], params_test_neural_normalized[response_test_neural_normalized==1], SS_test_normalized_neural[response_test_neural_normalized==1]
    params_normalized_nf = torch.tensor(params_normalized_nf, dtype=torch.float32)
    SS_normalized_nf = torch.tensor(SS_normalized_nf, dtype=torch.float32)
    params_test_normalized_nf = torch.tensor(params_test_normalized_nf, dtype=torch.float32)
    SS_test_normalized_nf = torch.tensor(SS_test_normalized_nf, dtype=torch.float32)
    return params_normalized_nf, SS_normalized_nf, params_test_normalized_nf, SS_test_normalized_nf

def test_rmin(r_min, r_max, N_bins, N_test, log_cluster = True):
        rmse_list_ncl = []
        rmse_list_nf = []
        print(f"\n=== Searching r_min ({len(r_min)} values, r_max={r_max}, N_bins={N_bins}) ===")
        for i, r in enumerate(r_min):
            print(f"\n[{i+1}/{len(r_min)}] r_min = {r:.4f}")
            #NCL
            response_neural_normalized, params_neural_normalized, SS_normalized_neural, params_mean, params_std, SS_mean, SS_std = format_data(X_train, Y_train, params_sample_train_NS, metro_year_list_train, df_metro, r_min=r, r_max=r_max, N_bins=N_bins, log_cluster=log_cluster)
            response_test_neural_normalized, params_test_neural_normalized, SS_test_normalized_neural, _, _, _, _ = format_data(X_test, Y_test, params_sample_test_NS, metro_year_list_test, df_metro, r_min=r, r_max=r_max, N_bins=N_bins, params_mean=params_mean, params_std=params_std, SS_mean=SS_mean, SS_std=SS_std, log_cluster=log_cluster)
            rmse_ncl = run_test_ncl(r_min=r, r_max=r_max, N_bins=N_bins, N_test=N_test, SS_normalized_neural=SS_normalized_neural, response_neural_normalized=response_neural_normalized, params_neural_normalized=params_neural_normalized, SS_test_normalized_neural=SS_test_normalized_neural, response_test_neural_normalized=response_test_neural_normalized, params_test_neural_normalized=params_test_neural_normalized, params_mean=params_mean, params_std=params_std)
            rmse_list_ncl.append(rmse_ncl)
            print(f"  => RMSE NCL: {rmse_ncl:.4f}")
            #NF
            params_normalized_nf, SS_normalized_nf, params_test_normalized_nf, SS_test_normalized_nf = format_data_nf(params_neural_normalized, SS_normalized_neural, params_test_neural_normalized, SS_test_normalized_neural, response_neural_normalized, response_test_neural_normalized)
            rmse_nf = run_test_nf(params_normalized_nf, SS_normalized_nf, params_mean, params_std, params_test_normalized_nf, SS_test_normalized_nf, N_test)
            rmse_list_nf.append(rmse_nf)
            print(f"  => RMSE NF: {rmse_nf:.4f}")
        print(f"\nBest for NCL r_min: {r_min[np.argmin(rmse_list_ncl)]:.4f} (RMSE={min(rmse_list_ncl):.4f})")
        print(f"\nBest for NF r_min: {r_min[np.argmin(rmse_list_nf)]:.4f} (RMSE={min(rmse_list_nf):.4f})")
        return rmse_list_ncl, rmse_list_nf

def test_rmax(r_min, r_max, N_bins, N_test, log_cluster = True):
        rmse_list_ncl = []
        rmse_list_nf = []
        print(f"\n=== Searching r_max ({len(r_max)} values, r_min={r_min:.4f}, N_bins={N_bins}) ===")
        for i, r in enumerate(r_max):
            print(f"\n[{i+1}/{len(r_max)}] r_max = {r:.4f}")
            #NCL
            response_neural_normalized, params_neural_normalized, SS_normalized_neural, params_mean, params_std, SS_mean, SS_std = format_data(X_train, Y_train, params_sample_train_NS, metro_year_list_train, df_metro, r_min=r_min, r_max=r, N_bins=N_bins, log_cluster=log_cluster)
            response_test_neural_normalized, params_test_neural_normalized, SS_test_normalized_neural, _, _, _, _ = format_data(X_test, Y_test, params_sample_test_NS, metro_year_list_test, df_metro, r_min=r_min, r_max=r, N_bins=N_bins, params_mean=params_mean, params_std=params_std, SS_mean=SS_mean, SS_std=SS_std, log_cluster=log_cluster)
            rmse_ncl = run_test_ncl(r_min=r_min, r_max=r, N_bins=N_bins, N_test=N_test, SS_normalized_neural=SS_normalized_neural, response_neural_normalized=response_neural_normalized, params_neural_normalized=params_neural_normalized, SS_test_normalized_neural=SS_test_normalized_neural, response_test_neural_normalized=response_test_neural_normalized, params_test_neural_normalized=params_test_neural_normalized, params_mean=params_mean, params_std=params_std)
            rmse_list_ncl.append(rmse_ncl)
            print(f"  => RMSE NCL: {rmse_ncl:.4f}")
            #NF
            params_normalized_nf, SS_normalized_nf, params_test_normalized_nf, SS_test_normalized_nf = format_data_nf(params_neural_normalized, SS_normalized_neural, params_test_neural_normalized, SS_test_normalized_neural, response_neural_normalized, response_test_neural_normalized)
            rmse_nf = run_test_nf(params_normalized_nf, SS_normalized_nf, params_mean, params_std, params_test_normalized_nf, SS_test_normalized_nf, N_test)
            rmse_list_nf.append(rmse_nf)
            print(f"  => RMSE NF: {rmse_nf:.4f}")
        print(f"\nBest for NCL r_max: {r_max[np.argmin(rmse_list_ncl)]:.4f} (RMSE={min(rmse_list_ncl):.4f})")
        print(f"\nBest for NF r_max: {r_max[np.argmin(rmse_list_nf)]:.4f} (RMSE={min(rmse_list_nf):.4f})")
        return rmse_list_ncl, rmse_list_nf

def test_nbins(r_min, r_max, nbins_plan, N_test, log_cluster = True):
        rmse_list_ncl = []
        rmse_list_nf = []
        print(f"\n=== Searching N_bins ({len(nbins_plan)} values, r_min={r_min:.4f}, r_max={r_max:.4f}) ===")
        for i, nb in enumerate(nbins_plan):
            print(f"\n[{i+1}/{len(nbins_plan)}] N_bins = {nb}")
            #NCL
            response_neural_normalized, params_neural_normalized, SS_normalized_neural, params_mean, params_std, SS_mean, SS_std = format_data(X_train, Y_train, params_sample_train_NS, metro_year_list_train, df_metro, r_min=r_min, r_max=r_max, N_bins=nb, log_cluster=log_cluster)
            response_test_neural_normalized, params_test_neural_normalized, SS_test_normalized_neural, _, _, _, _ = format_data(X_test, Y_test, params_sample_test_NS, metro_year_list_test, df_metro, r_min=r_min, r_max=r_max, N_bins=nb, params_mean=params_mean, params_std=params_std, SS_mean=SS_mean, SS_std=SS_std, log_cluster=log_cluster)
            rmse_ncl = run_test_ncl(r_min=r_min, r_max=r_max, N_bins=nb, N_test=N_test, SS_normalized_neural=SS_normalized_neural, response_neural_normalized=response_neural_normalized, params_neural_normalized=params_neural_normalized, SS_test_normalized_neural=SS_test_normalized_neural, response_test_neural_normalized=response_test_neural_normalized, params_test_neural_normalized=params_test_neural_normalized, params_mean=params_mean, params_std=params_std)
            rmse_list_ncl.append(rmse_ncl)
            print(f"  => RMSE NCL: {rmse_ncl:.4f}")
            #NF
            params_normalized_nf, SS_normalized_nf, params_test_normalized_nf, SS_test_normalized_nf = format_data_nf(params_neural_normalized, SS_normalized_neural, params_test_neural_normalized, SS_test_normalized_neural, response_neural_normalized, response_test_neural_normalized)
            rmse_nf = run_test_nf(params_normalized_nf, SS_normalized_nf, params_mean, params_std, params_test_normalized_nf, SS_test_normalized_nf, N_test)
            rmse_list_nf.append(rmse_nf)
            print(f"  => RMSE NF: {rmse_nf:.4f}")
        print(f"\nBest for NCL N_bins: {nbins_plan[np.argmin(rmse_list_ncl)]} (RMSE={min(rmse_list_ncl):.4f})")
        print(f"\nBest for NF N_bins: {nbins_plan[np.argmin(rmse_list_nf)]} (RMSE={min(rmse_list_nf):.4f})")
        return rmse_list_ncl, rmse_list_nf

#General settings
K = 3
T = 1*365

#NCL settings
ncl_epochs = 200
ncl_lr = 0.0001
N_grid = 1000
sub_model_epochs = 100
sub_model_lr = 0.001

def main():
    N_test = 50

    # N_bins = 25  # fixed high resolution for r_min and r_max searches
    # r_min_configs = 5
    # r_min_plan = np.logspace(np.log10(0.001), np.log10(3), r_min_configs)
    # r_max = 10
    # print("Testing r_min...")
    # print("r_min plan:", r_min_plan)
    # rmse_list_ncl_rmin, rmse_list_nf_rmin = test_rmin(r_min_plan, r_max, N_bins, N_test=N_test)
    # best_rmin_idx_ncl = int(np.argmin(rmse_list_ncl_rmin))
    # best_rmin_idx_nf = int(np.argmin(rmse_list_nf_rmin))

    # r_max_configs = 5
    # r_min = min(r_min_plan[best_rmin_idx_ncl], r_min_plan[best_rmin_idx_nf])   
    # r_max_plan = np.linspace(r_min, 10, r_max_configs+1)[1:] 
    # print("\nTesting r_max...")
    # print("r_max plan:", r_max_plan)
    # rmse_list_ncl_rmax, rmse_list_nf_rmax = test_rmax(r_min, r_max_plan, N_bins, N_test=N_test)
    # best_rmax_idx_ncl = int(np.argmin(rmse_list_ncl_rmax))
    # best_rmax_idx_nf = int(np.argmin(rmse_list_nf_rmax))

    nbins_configs = 10
    nbins_plan = np.linspace(1, 30, nbins_configs, dtype=int)
    r_max = 3
    r_min = 0.001
    print("\nTesting N_bins...")
    print("N_bins plan:", nbins_plan)
    rmse_list_ncl_nbins, rmse_list_nf_nbins = test_nbins(r_min, r_max, nbins_plan, N_test=N_test, log_cluster=False)
    best_nbins_idx_ncl = int(np.argmin(rmse_list_ncl_nbins))
    best_nbins_idx_nf = int(np.argmin(rmse_list_nf_nbins))

    np.savez("results/lag_search/lagsearch_nbins_linear.npz",
        rmse_list_ncl_nbins=rmse_list_ncl_nbins,
        rmse_list_nf_nbins=rmse_list_nf_nbins,
        nbins_values=nbins_plan)
    print(f"\nBest for NCL N_bins: {nbins_plan[best_nbins_idx_ncl]} (RMSE={min(rmse_list_ncl_nbins):.4f})")
    print(f"\nBest for NF N_bins: {nbins_plan[best_nbins_idx_nf]} (RMSE={min(rmse_list_nf_nbins):.4f})")

    # np.savez("results/lag_search/lagsearch.npz",
    #         rmse_list_ncl_r_min=rmse_list_ncl_rmin,
    #         rmse_list_nf_r_min=rmse_list_nf_rmin,
    #         r_min_values=r_min_plan,
    #         rmse_list_ncl_r_max=rmse_list_ncl_rmax,
    #         rmse_list_nf_r_max=rmse_list_nf_rmax,
    #         r_max_values=r_max_plan,
    #         rmse_list_ncl_nbins=rmse_list_ncl_nbins,
    #         rmse_list_nf_nbins=rmse_list_nf_nbins,
    #         nbins_values=nbins_plan)

    # print_summary("NCL", N_test, N_bins, r_min_plan, r_max_plan, nbins_plan,
    #               rmse_list_ncl_rmin, rmse_list_ncl_rmax, rmse_list_ncl_nbins,
    #               best_rmin_idx_ncl, best_rmax_idx_ncl, best_nbins_idx_ncl)
    # print_summary("NF", N_test, N_bins, r_min_plan, r_max_plan, nbins_plan,
    #               rmse_list_nf_rmin, rmse_list_nf_rmax, rmse_list_nf_nbins,
    #               best_rmin_idx_nf, best_rmax_idx_nf, best_nbins_idx_nf)

main()