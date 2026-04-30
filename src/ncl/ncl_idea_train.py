import keras
from keras import layers, ops
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

import src.ncl.ncl_idea_utils as ncl_utils

# Load the data
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

print("Number of training data points:", len(params_train_normalized))

#Define network architecture and training parameters
gs = len(SS_0_train_normalized_neural[0])  # Dimension of summary statistics
print("Dimension of summary statistics:", gs)
p = len(params_train_normalized[0])  # Dimension of parameters
print("Dimension of parameters:", p)


# Stage 1: pre-train F_point on (S, theta) pairs from your simulator.
params_train_pre, params_test_pre, X_global_train_pre, X_global_test_pre = params_train_normalized[response_train==1], params_test_normalized[response_test==1], SS_0_train_normalized_neural[response_train==1], SS_0_test_normalized_neural[response_test==1]
F_point = ncl_utils.build_point_predictor(gs, p)
F_point.compile(optimizer=keras.optimizers.Adam(1e-3, clipnorm=1.0), loss="mse")
ES_point = keras.callbacks.EarlyStopping("val_loss", patience=15, restore_best_weights=True, verbose=1)
historyF_point = F_point.fit(X_global_train_pre, params_train_pre, epochs=200, validation_split=0.1, callbacks=[ES_point])

# Stage 2: train the full classifier with BCE.
# freeze_point=False so F_point drifts during training, which forces the
# curvature head to develop a sharp Jacobian at u=0.  That sharpness is
# what produces a well-scaled Fisher information matrix.  Gradient clipping
# addresses the spike instability without suppressing curvature growth.
model, _, delta_net, curvature_head = ncl_utils.build_neural_likelihood(
    gs, p,
    point_predictor=F_point,
    freeze_point=False,
    use_mode_correction=True,
)
model.compile(
    optimizer=keras.optimizers.Adam(1e-4, clipnorm=1.0),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
)
ES_class = keras.callbacks.EarlyStopping("val_loss", patience=20, restore_best_weights=True, verbose=1)
history_class = model.fit(
    [params_train_normalized, SS_0_train_normalized_neural], response_train,
    epochs=300, validation_split=0.1, callbacks=[ES_class],
)


#Test idea
response_test_pred_logits = model.predict([params_test_normalized, SS_0_test_normalized_neural])
response_test_pred_probs = tf.sigmoid(response_test_pred_logits).numpy().flatten()


fpr, tpr, _ = roc_curve(response_test, response_test_pred_probs)
roc_auc = auc(fpr, tpr)

# Confusion matrix
response_test_pred = (response_test_pred_probs >= 0.5).astype(int)
cm = confusion_matrix(response_test, response_test_pred)

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.ravel()

# Loss plot
axs[0].plot(history_class.history["loss"], label="Loss")
axs[0].plot(history_class.history["val_loss"], label="Val_loss")
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
plt.savefig("results/idea_network/idea_performance.png")
plt.close()

print("\nSaving trained models...")
model.save("neural_networks/idea_network/model_small.keras")
F_point.save("neural_networks/idea_network/F_point_small.keras")
if delta_net is not None:
    delta_net.save("neural_networks/idea_network/delta_net_small.keras")
print("Models saved to neural_networks/idea_network/")

