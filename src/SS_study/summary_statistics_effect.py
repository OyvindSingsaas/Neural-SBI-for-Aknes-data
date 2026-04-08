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
    data = np.load('data/NS_data_SS_study.npz', allow_pickle=True)
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


    #Loop over different combinations of RK and evaluate the performance of the NCL model for each combination.
    #For each combination, we will train a sub-model to predict the parameters from the summary statistics, and then use the predictions from this sub-model as input to the full model for classification.
    #We will evaluate the performance of the full model using ROC AUC and confusion matrix, and we will also evaluate the performance of the sub-model using mean squared error.

    gs_total = len(SS_0_train_normalized_neural[0])  # Dimension of summary statistics

    RK_n = len(cluster_bins) #number of RK evaluations in summary statistics
    #Indexes corresponing ro RK evaluations in summary statistics (the RK_n entries in the summary statistics vector leading up to but excluding the last element)
    RK_indexes = np.arange(gs_total - RK_n - 1, gs_total - 1) 
    RK_range = cluster_bins[-1] - cluster_bins[0]

    print("Total dimension of summary statistics:", gs_total)
    print("RK indexes in summary statistics:", RK_indexes)
    print("Number of RK evaluetions in summary statistics:", RK_n)
    print("Cluster_bins:", cluster_bins)
    mse_list_NCL = [] #un-normalized mse
    mse_list_sub_model = [] #un-normalized mse
    N_test = 100
    year = [2023]

    N_rk = np.linspace(0, RK_n-2, 10, dtype=int) #number of RK evaluations to drop in the summary statistics, evenly spaced between the minimum and maximum cluster bin values
    print("Dropping plan. N_rk:", N_rk)
    #N_runs different summary statistic configurations with increasing resolution
    for k, n_rk in enumerate(N_rk):
        gs = gs_total - n_rk  # Dimension of summary statistics for the current run, starting with all RK evaluations and removing one RK evaluation at a time
        #select n_rk RK evaulations to drop from summary statistics, evenly spaced between the minimum and maximum cluster bin values
        cluster_bin_indexes = np.round(np.linspace(0, RK_n-1, n_rk+2)).astype(int)[1:-1]
        dropped_RK_indexes = RK_indexes[cluster_bin_indexes]
        print("########")
        print(f"\n Iteration {k}: Dropping {n_rk} RK evaluations from summary statistics.")
        try:print("Dropped RK evaluations for summary statistics:", cluster_bins[cluster_bin_indexes])
        except:print("Dropped RK evaluations for summary statistics: None")
        print("Dropped gs indecies", dropped_RK_indexes)
        print("Dropped cluster_bind indecies", cluster_bin_indexes)

        #Create new summary statistics by dropping the selected RK evaluations from the original summary statistics
        SS_0_test_normalized_neural_temp = np.delete(SS_0_test_normalized_neural, dropped_RK_indexes, axis=1)
        SS_0_train_normalized_neural_temp = np.delete(SS_0_train_normalized_neural, dropped_RK_indexes, axis=1)
        SS_mean_temp = np.delete(SS_mean, dropped_RK_indexes, axis=0)
        SS_std_temp = np.delete(SS_std, dropped_RK_indexes, axis=0)

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
        params_train_pre, params_test_pre, X_global_train_pre, X_global_test_pre = params_train_normalized[response_train==1], params_test_normalized[response_test==1], SS_0_train_normalized_neural_temp[response_train==1], SS_0_test_normalized_neural_temp[response_test==1]
        ES_NS =keras.callbacks.EarlyStopping("val_loss", patience=15, verbose = 1, restore_best_weights=True)

        history = sub_model_NS.fit(X_global_train_pre, params_train_pre, epochs=500, batch_size=100, verbose = 1, validation_split=0.1, callbacks=[ES_NS])

        plt.figure()
        plt.plot(history.history["loss"], label = "Loss")
        plt.plot(history.history["val_loss"], label = "Val_loss")
        plt.legend()
        #save the plot
        try:
            plt.suptitle(f"RK evaluations included: {cluster_bins[cluster_bin_indexes]} and above")
        except:
            plt.suptitle(f"No RK evaluations included")

        #plt.savefig(f"neural_networks/SS_sensitivity_3/sub_model_NS_training_curve_{k}.png")
        plt.close()

        Y_pred = sub_model_NS.predict(X_global_test_pre)
        Y_pred_unnormalized = Y_pred * params_std + params_mean
        params_test_pre_unnormalized = params_test_pre * params_std + params_mean
        mse_list_sub_model.append(mean_squared_error(params_test_pre_unnormalized, Y_pred_unnormalized))
        n_params = len(l_bounds_NS)
        fig, axs = plt.subplots(1, n_params, figsize=(4*n_params, 4))
        if n_params == 1:
            axs = [axs]
        for dim in range(n_params):
            ax = axs[dim]
            ax.set_title(f"{col_names_params_NS[dim]}, MSE: {mean_squared_error(params_test_pre_unnormalized[:, dim], Y_pred_unnormalized[:, dim]):.4f}", fontsize=10)
            ax.scatter(Y_pred_unnormalized[:, dim], params_test_pre_unnormalized[:, dim], alpha=0.7)
            line_vals = np.linspace(l_bounds_NS_test[dim], u_bounds_NS_test[dim], 100)
            ax.plot(line_vals, line_vals, linestyle = "--", color = "grey")
            ax.set_xlabel("Predicted", fontsize=9)
            ax.set_ylabel("True", fontsize=9)
        plt.tight_layout()
        try:            plt.suptitle(f"RK evaluations included: {cluster_bins[cluster_bin_indexes]:.4f} and above")
        except:            plt.suptitle(f"No RK evaluations included")
        plt.savefig(f"neural_networks/SS_sensitivity_3/sub_model_NS_prediction_all_{k}.png")
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
        initial_learning_rate = 0.001

        optimizer = keras.optimizers.Adam(learning_rate=initial_learning_rate)
        classification_NN_NS.compile(optimizer=optimizer, loss=keras.losses.BinaryCrossentropy(from_logits=True))
        classification_NN_NS.summary()

        #classification_NN_NS.optimizer.learning_rate.assign(0.001)
        ES =keras.callbacks.EarlyStopping("val_loss", patience=20, verbose = 1, restore_best_weights=True)
        history = classification_NN_NS.fit([SS_0_train_normalized_neural_temp, params_train_normalized], response_train, epochs=500, batch_size=32, verbose = 1, validation_split=0.1, callbacks=[ES])#sample_weight=d_e)

        response_test_pred_logits = classification_NN_NS.predict([SS_0_test_normalized_neural_temp, params_test_normalized]).reshape((-1))
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
        try:            plt.suptitle(f"RK evaluations included: {cluster_bins[cluster_bin_indexes]} and above")
        except:            plt.suptitle(f"No RK evaluations included")
        #plt.savefig(f"neural_networks/SS_sensitivity_3/NCL_NS_performance_test_{k}.png")
        plt.close()

        response_train_pred_logits = classification_NN_NS.predict([SS_0_train_normalized_neural_temp, params_train_normalized]).reshape((-1))

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
        try:            plt.suptitle(f"RK evaluations included: {cluster_bins[cluster_bin_indexes]} and above")
        except:            plt.suptitle(f"No RK evaluations included")
        #plt.savefig(f"neural_networks/SS_sensitivity_3/NCL_NS_performance_train_{n_rk}.png")
        plt.close()

        N_grid = 1000
        grid = us.LHS(N_grid, l_bounds_NS, u_bounds_NS, year)[:,:-1]
        grid_norm = (grid - params_mean) / params_std
        mle_ncl = np.zeros((N_test, len(l_bounds_NS)))
        for i in range(N_test):
            print("\nPredicting parameters using the neural network model...")
            if i % (N_test//10) == 0:
                print(round(i/N_test*100), "% done")

            ll_grid = classification_NN_NS.predict([SS_0_test_normalized_neural_temp[response_test==1][i].reshape((1,-1)).repeat(N_grid, axis = 0), grid_norm]).reshape(-1)  
            start = grid_norm[np.argmax(ll_grid)]
            MLE_NCL, MLE_NCL_normalized, final_logit, _, end_state = us.numerical_optim(start, SS_0_test_normalized_neural_temp[response_test==1][i], l_bounds_NS, u_bounds_NS, classification_NN_NS, gs, params_mean=params_mean, params_std=params_std)
            mle_ncl[i] = MLE_NCL
        mle_ncl_normalized = (mle_ncl - params_mean) / params_std
        mse_list_NCL.append(mean_squared_error(params_test_pre_unnormalized[:N_test], mle_ncl))

        print("\nSaving ")
        np.savez(f"neural_networks/SS_sensitivity_3/ncl_{n_rk}.npz", ncl_mle=mle_ncl, ncl_mle_normalized=mle_ncl_normalized)
    
    #Plot mse vs number of RK evaluations included in summary statistics for both the sub-model and the full model


    fig, ax1 = plt.subplots()
    color1 = 'tab:blue'
    color2 = 'tab:orange'
    width = 0.35
    x = RK_n - N_rk
    ax1.bar(x - width/2, mse_list_sub_model, width, color=color1, label="Point estimate")
    ax1.set_xlabel("#RK evaluations")
    ax1.set_ylabel("MSE (Point estimate)", color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(int(val)) for val in x])

    ax2 = ax1.twinx()
    ax2.bar(x + width/2, mse_list_NCL, width, color=color2, label="NCL")
    ax2.set_ylabel("MSE (NCL)", color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Add legends
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='best')

    plt.tight_layout()
    plt.savefig("neural_networks/SS_sensitivity_3/MSE_vs_RK_evaluations.png")
    plt.close()

    #plot only ncl results
    plt.figure()
    plt.bar(x, mse_list_NCL, width, color=color2, label="NCL")
    plt.xlabel("#RK evaluations")
    plt.ylabel("MSE", color=color2)
    plt.xticks(x, [str(int(val)) for val in x])
    plt.legend()
    plt.tight_layout()
    plt.savefig("neural_networks/SS_sensitivity_3/MSE_NCL_vs_RK_evaluations.png")
    plt.close()

    print("MSE for sub-model:", mse_list_sub_model)
    print("MSE for NCL:", mse_list_NCL)

    print("All iterations completed.")


main()