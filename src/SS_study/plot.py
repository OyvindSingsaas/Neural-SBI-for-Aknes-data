

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

data = np.load('data/NS_data_SS_study.npz', allow_pickle=True)
params_std = data['params_std']
params_mean = data['params_mean']
#mse = np.array([0.043368027293873276, 0.0420326122889147, 0.049438600007539094, 0.050306046088488515, 0.0461735203465074, 0.04508248895234812, 0.05400431718241135, 0.07900731985533094, 0.07661735680946181, 0.17328884908887748])

path = "neural_networks/SS_sensitivity_4/rmse_results.npz"
results = np.load(path, allow_pickle=True)
rmse_list_sub_model = results["rmse_list_sub_model"]
rmse_list_NCL = results["rmse_list_NCL"]
rmse_list_abc = results["rmse_list_abc"]
rmse_list_nf = results["rmse_list_nf"]
x_vals = results["x_vals"]



#plot mse vs rk evaluations for both normalized and unnormalized parameters in two separate figures
#RK_n = 20
#N_rk = np.linspace(0, RK_n-2, 10, dtype=int ) #number of RK evaluations to drop in the summary statistics, evenly spaced between the minimum and maximum cluster bin values
#x_vals = RK_n - N_rk
plt.figure()
plt.plot(x_vals,rmse_list_NCL, marker = "o")
plt.plot(x_vals,rmse_list_nf, marker = "o")
plt.xlabel("Number of K-function evaluations included in summary statistics")
plt.ylabel("MSE")
plt.grid()
plt.legend()
plt.xticks(x_vals)
plt.savefig("neural_networks/SS_sensitivity_4/MSE_vs_RK_unnormalized.png")
plt.close()
