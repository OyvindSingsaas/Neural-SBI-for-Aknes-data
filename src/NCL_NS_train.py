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

    print("Number of training data points:", len(params_train_normalized))

    #Define network architecture and training parameters
    gs = len(SS_0_train_normalized_neural[0])  # Dimension of summary statistics
    print("Dimension of summary statistics:", gs)


    #Define the sub-model for NS, trained on point prediction task
    x_global_input = Input(shape=(gs,))

    x = Dense(32, activation='tanh')(x_global_input)
    x = Dense(32, activation='tanh')(x)
    x = Dense(8, activation='tanh')(x)
    output = Dense(len(l_bounds_NS), activation='linear')(x)  

    sub_model_NS = Model(x_global_input, output)

    # Compile the model
    initial_learning_rate = 0.001
    optimizer = keras.optimizers.Adam(learning_rate=initial_learning_rate)
    sub_model_NS.compile(optimizer=optimizer, loss="mse")
    sub_model_NS.summary()

    #train sub-model for NS
    params_train_pre, params_test_pre, X_global_train_pre, X_global_test_pre = params_train_normalized[response_train==1], params_test_normalized[response_test==1], SS_0_train_normalized_neural[response_train==1], SS_0_test_normalized_neural[response_test==1]
    ES_NS =keras.callbacks.EarlyStopping("val_loss", patience=15, verbose = 1, restore_best_weights=True)

    history = sub_model_NS.fit(X_global_train_pre, params_train_pre, epochs=500, batch_size=100, verbose = 1, validation_split=0.1, callbacks=[ES_NS])

    plt.figure()
    plt.plot(history.history["loss"], label = "Loss")
    plt.plot(history.history["val_loss"], label = "Val_loss")
    plt.legend()
    #save the plot
    plt.savefig("neural_networks/sub_model_NS_training_curve.png")
    plt.close()

    Y_pred = sub_model_NS.predict(X_global_test_pre)
    for dim in range(len(l_bounds_NS)):
        plt.figure()
        plt.title(f"{col_names_params_NS[dim]}, MSE: {mean_squared_error(params_test_pre[:, dim], Y_pred[:, dim]):.4f}")
        plt.scatter(Y_pred[:, dim], params_test_pre[:, dim])
        plt.plot((np.linspace(l_bounds_NS_test[dim], u_bounds_NS_test[dim]) - params_mean[dim])/params_std[dim], (np.linspace(l_bounds_NS_test[dim], u_bounds_NS_test[dim]) - params_mean[dim])/params_std[dim], linestyle = "--", color = "grey")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(f"neural_networks/sub_model_NS_prediction_{dim}.png")
        plt.close()

    #Define full model for NS, which takes both summary statistics and parameters as input and outputs the probability of response = 1. The sub-model is included in the full model and is not trainable during training of the full model.
    sub_model_clone_NS  = keras.models.clone_model(sub_model_NS)
    sub_model_clone_NS.set_weights(sub_model_NS.get_weights())
    sub_model_clone_NS_cutted = Model(sub_model_clone_NS.inputs, sub_model_clone_NS.layers[-1].output)

    for layer in sub_model_clone_NS_cutted.layers:
        layer.trainable = False

    Input_summary_statistics = Input(shape=(gs,))
    Input_parameters = Input(shape=(len(u_bounds_NS),))

    #sub_model_output = sub_model_clone_cutted(Input_summary_statistics)
    sub_model_NS_output = Sequential(sub_model_clone_NS_cutted.layers)(Input_summary_statistics)

    concatenated = Concatenate(name = "Combine")([Input_summary_statistics, Input_parameters, sub_model_NS_output])

    x = Dense(32, activation='tanh', name = "First_dense")(concatenated)
    x = Dense(16, activation='tanh')(x)
    x = Dense(64, activation='tanh')(x)
    x = Dense(32, activation='tanh')(x)

    output = Dense(1, activation='linear')(x)  

    # Model definition
    classification_NN_NS = Model([Input_summary_statistics, Input_parameters], output)

    # Compile the model
    initial_learning_rate = 0.0001

    optimizer = keras.optimizers.Adam(learning_rate=initial_learning_rate)
    classification_NN_NS.compile(optimizer=optimizer, loss=keras.losses.BinaryCrossentropy(from_logits=True))
    classification_NN_NS.summary()

    #classification_NN_NS.optimizer.learning_rate.assign(0.001)
    ES =keras.callbacks.EarlyStopping("val_loss", patience=20, verbose = 1, restore_best_weights=True)
    history = classification_NN_NS.fit([SS_0_train_normalized_neural, params_train_normalized], response_train, epochs=500, batch_size=32, verbose = 1, validation_split=0.1, callbacks=[ES])#sample_weight=d_e)

    response_test_pred_logits = classification_NN_NS.predict([SS_0_test_normalized_neural, params_test_normalized]).reshape((-1))
    response_test_pred_probs = 1 / (1 + np.exp(-response_test_pred_logits))

    fpr, tpr, _ = roc_curve(response_test, response_test_pred_probs)
    roc_auc = auc(fpr, tpr)

    # Confusion matrix
    response_test_pred = (response_test_pred_probs >= 0.5).astype(int)
    cm = confusion_matrix(response_test, response_test_pred)

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.ravel()

    # Loss plot
    axs[0].plot(history.history["loss"], label="Loss")
    axs[0].plot(history.history["val_loss"], label="Val_loss")
    axs[0].set_title("Training and Validation Loss")
    axs[0].legend()

    # ROC curve
    axs[1].plot(fpr, tpr, color='b', label=f'ROC Curve (AUC = {roc_auc:.5f})')
    axs[1].plot([0, 1], [0, 1], 'k--')
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive Rate')
    axs[1].set_title('ROC Curve')
    axs[1].legend()
    axs[1].grid()

    # Histogram of predicted probabilities
    sns.histplot(response_test_pred_probs[response_test == 1], color='b', label='True pair', kde=True, bins=100, ax=axs[2])
    sns.histplot(response_test_pred_probs[response_test == 0], color='r', label='Not pair', kde=True, bins=100, ax=axs[2])
    axs[2].set_xlabel('Predicted Probability')
    axs[2].set_ylabel('Density')
    axs[2].set_title('Predicted Probability Distribution by Class')
    axs[2].legend()

    # Confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(ax=axs[3], cmap=plt.cm.Blues, colorbar=False)
    axs[3].set_title("Confusion Matrix")

    plt.tight_layout()
    plt.savefig("neural_networks/NCL_NS_performance_test.png")
    plt.close()

    response_train_pred_logits = classification_NN_NS.predict([SS_0_train_normalized_neural, params_train_normalized]).reshape((-1))

    response_train_pred_probs = 1 / (1 + np.exp(-response_train_pred_logits))
    fpr, tpr, _ = roc_curve(response_train, response_train_pred_probs)
    roc_auc = auc(fpr, tpr)
    # Confusion matrix
    response_train_pred = (response_train_pred_probs >= 0.5).astype(int)
    cm = confusion_matrix(response_train, response_train_pred)
    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.ravel()
    # Loss plot
    axs[0].plot(history.history["loss"], label="Loss")
    axs[0].plot(history.history["val_loss"], label="Val_loss")
    axs[0].set_title("Training and Validation Loss")
    axs[0].legend()
    # ROC curve
    axs[1].plot(fpr, tpr, color='b', label=f'ROC Curve (AUC = {roc_auc:.5f})')
    axs[1].plot([0, 1], [0, 1], 'k--')
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive Rate')
    axs[1].set_title('ROC Curve')
    axs[1].legend()
    axs[1].grid()
    # Histogram of predicted probabilities
    sns.histplot(response_train_pred_probs[response_train == 1], color='b', label='True pair', kde=True, bins=100, ax=axs[2])
    sns.histplot(response_train_pred_probs[response_train == 0], color='r', label='Not pair', kde=True, bins=100, ax=axs[2])
    axs[2].set_xlabel('Predicted Probability')
    axs[2].set_ylabel('Density')
    axs[2].set_title('Predicted Probability Distribution by Class')
    axs[2].legend()
    # Confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(ax=axs[3], cmap=plt.cm.Blues, colorbar=False)
    axs[3].set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("neural_networks/NCL_NS_performance_train.png")
    plt.close()

    #save full model
    classification_NN_NS.save("neural_networks/classification_NN_NS.h5")

    np.random.seed(42)
    n_samples = response_test_pred_probs.shape[0]

    logit_h = response_test_pred_logits

    X_platt = logit_h.reshape(-1, 1)
    model_calibration = LogisticRegression()
    model_calibration.fit(X_platt, response_test) 
    platt_scaler = model_calibration.coef_[0][0]
    #save the Platt scaling factor
    np.save("neural_networks/platt_scaler_NS.npy", platt_scaler)
    print(f"Model Coefficients: β0 = {model_calibration.intercept_[0]:.4f}, β1 = {model_calibration.coef_[0][0]:.4f}")

main()