import utils.utils_surface_NS as us
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils.utils_abc as abc
import time
from tensorflow.keras.models import load_model
from scipy.stats import norm
import tensorflow as tf
import keras
import src.ncl.ncl_idea_utils as ncl_utils
import os
from scipy.stats import gaussian_kde

col_names_params_NS = [
    r"$\log(\delta)$",
    r"$\log(\sigma^2)$",
    r"$\beta_0$",
    r"$\beta_1$",
    r"$\beta_2$",
    r"$\beta_3$",
    r"$\beta_4$",
    r"$\beta_5$",
    r"$\beta_6$"
]

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

def godambe_G_NS(N_G, theta_0, params_mean, params_std, gs, year, p, T, cluster_bins, percentiles, df_metro, classification_NN_NS, SS_mean, SS_std, plot = False):

    theta_0_repeated = np.repeat(np.reshape(theta_0, (1,-1)), N_G, axis = 0)
    theta_0_repeated_shape = np.zeros((theta_0_repeated.shape[0], theta_0_repeated.shape[1]+1))
    theta_0_repeated_shape[:, :-1] = theta_0_repeated
    theta_0_repeated_shape[:, -1] = np.repeat(year, theta_0_repeated.shape[0])

    SS_observed_bootstrap =  np.zeros((N_G, gs))
    X_boot, Y_boot, metro_year_list_train_boot, invalid_index_train = us.simulate_given_params(J=N_G, K=1, T=T, p=p, df_metro = df_metro, params=theta_0_repeated_shape, error = 0)

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



def plot_example(NF_samples, abc_samples, MLE_NCL_0_NS, G_inv_NS, H_neg_NS,
                  platt_scaler, theta_0_NS, col_names_params_NS,
                    params_mean, params_std, l_bounds_NS, u_bounds_NS,
                    mle_ncl_idea = None, fisher_info_ncl_idea = None, plot_idea = False):
    plt.rcParams.update({
        "font.size": 16,        # default text
        "axes.titlesize": 18,   # title
        "axes.labelsize": 18,   # x and y labels
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 14
    })

    fig, axs = plt.subplots(2, 3, figsize=(12, 10))#, gridspec_kw={'hspace': 0.4, 'wspace': 0.4}
    axs = axs.ravel()

    w = 5
    k = 0.5
    resolution = 10000
    alpha = 0.1
    shade = 0.1
    for dim in range(len(MLE_NCL_0_NS)):

        dim_vary = np.linspace(MLE_NCL_0_NS[dim] - w, MLE_NCL_0_NS[dim] + w, resolution)
        #dim_vary = np.linspace(l_bounds[dim], u_bounds[dim], resolution)
        full_dim = MLE_NCL_0_NS.reshape((1, -1)).repeat(resolution, axis = 0)
        full_dim[:, dim] = dim_vary
        full_dim_norm = (full_dim-params_mean)/params_std 
        ax = axs[dim] if len(MLE_NCL_0_NS) > 1 else axs
        
        #-----------
        samples_NF_temp = NF_samples['samples'][:, dim]
        kde = gaussian_kde(samples_NF_temp)

        low, high = np.percentile(samples_NF_temp, [5, 95])
        y_kde_NF = kde(dim_vary)

        ax.plot(dim_vary, y_kde_NF, color='green', lw=2, label = "NF")
        ax.axvline(x=NF_samples["NF_map"][dim], color='green', linestyle='--', linewidth=2)
        mask = (dim_vary >= low) & (dim_vary <= high)
        ax.fill_between(dim_vary[mask], y_kde_NF[mask], color='green', alpha=shade)
        #-----------

        #ABC kde
        abc_samples_dim = abc_samples[:, dim]
        kde = gaussian_kde(abc_samples_dim)
        y_kde = kde(dim_vary)
        ax.plot(dim_vary, y_kde, color='grey', lw=2, label = "ABC")
        ax.axvline(x=np.mean(abc_samples_dim), color='grey', linestyle='--', linewidth=2)
        low_abc, high_abc = np.percentile(abc_samples_dim, [5, 95])
        mask_abc = (dim_vary >= low_abc) & (dim_vary <= high_abc)
        ax.fill_between(dim_vary[mask_abc], y_kde[mask_abc], color='grey', alpha=shade)

        #density histogram for abc samples
        #ax.hist(abc_samples_dim, bins=30, density=True, alpha=0.5, color='grey', label='ABC Samples')

        #Gaussian pdf
        mu = MLE_NCL_0_NS[dim]
        std_G = np.sqrt(G_inv_NS[dim][dim])*params_std[dim]
        std = np.sqrt(np.linalg.inv(H_neg_NS)[dim][dim])*params_std[dim]
        std_Platt = np.sqrt(np.linalg.inv(H_neg_NS*platt_scaler)[dim][dim])*params_std[dim]

        pdf_G = norm.pdf(dim_vary, loc = mu, scale = std_G)
        pdf = norm.pdf(dim_vary, loc=mu, scale=std)
        pdf_Platt = norm.pdf(dim_vary, loc=mu, scale=std_Platt)

        ymin = 0
        ymax = np.max([np.max(pdf), np.max(pdf_G), np.max(pdf_Platt), np.max(y_kde_NF), np.max(y_kde)]) 

        # CI
        z = norm.ppf(1 - alpha/2)
        lower, upper =  mu- z*std, mu + z*std
        inside = (dim_vary >= lower) & (dim_vary <= upper)

        lower, upper = mu - z*std_G, mu + z*std_G
        inside_G = (dim_vary >= lower) & (dim_vary <= upper)

        lower, upper = mu - z*std_Platt, mu + z*std_Platt
        inside_platt = (dim_vary >= lower) & (dim_vary <= upper)

        
        #ax.axvline(NF_map[dim], color="green", linestyle="--", label = "NF MAP")
        ax.vlines(x=MLE_NCL_0_NS[dim], ymax=ymax*1.1, ymin=ymin, linestyles="--", color="Blue")

        #ax.fill_between(dim_vary[mask], y_kde[mask], color='green', alpha=shade)

        #ax.set_xlabel(f"{col_names_params_NS[dim]}")
        ax.set_title(f"{col_names_params_NS[dim]}")
        ax.plot(dim_vary, pdf, label="Uncalibrated", color="purple")
        ax.plot(dim_vary, pdf_Platt, label="Platt", color="orange")
        ax.plot(dim_vary, pdf_G, label="Godambe", color="blue")

        #ax.fill_between(dim_vary, Z_0, y2=ymin, where=inside, color='purple', alpha=0.3, label=f'{alpha*100:.0f}% CI')
        ax.fill_between(dim_vary, pdf_G, y2=ymin, where=inside_G, color='blue', alpha=shade)
        ax.fill_between(dim_vary, pdf_Platt, y2=ymin, where=inside_platt, color='orange', alpha=shade)
        ax.fill_between(dim_vary, pdf, y2=ymin, where=inside, color='purple', alpha=shade)


        if mle_ncl_idea is not None and fisher_info_ncl_idea is not None and plot_idea:
            print(dim_vary.shape)
            std_idea = np.sqrt(np.linalg.inv(fisher_info_ncl_idea)[dim][dim])*params_std[dim]
            pdf_idea = norm.pdf(dim_vary, loc = mle_ncl_idea[dim], scale = std_idea)
            ax.plot(dim_vary, pdf_idea, label="NCL idea", color="cyan")
            lower, upper = mle_ncl_idea[dim] - z*std_idea, mle_ncl_idea[dim] + z*std_idea
            inside_idea = (dim_vary >= lower) & (dim_vary <= upper)
            ax.fill_between(dim_vary, pdf_idea, y2=ymin, where=inside_idea, color='cyan', alpha=shade)

        ncl_lower = mu - z * np.max([std, std_Platt, std_G])
        ncl_upper = mu + z * np.max([std, std_Platt, std_G])
        xmin = min(ncl_lower, low, low_abc)
        xmax = max(ncl_upper, high, high_abc)
        margin = (xmax - xmin) * 0.05
        ax.set_xlim(xmin - margin, xmax + margin)
        ax.set_ylim((ymin, ymax*1.1))
        ax.vlines(x=theta_0_NS[dim], ymax=ymax*1.1, ymin=ymin, linestyles="--", color="red", label="True")
        ax.set_yticks([])

    handles, labels = axs[0].get_legend_handles_labels()

    ax = axs[-1]
    #Remove the last subplot (empty)
    fig.delaxes(ax)
    #Add legend to the last subplot
    fig.legend(handles, labels, bbox_to_anchor=(0.85, 0.4), ncol=1)#, 
    #fig.suptitle("Your Figure Title", fontsize=18, y=0.98)
    plt.tight_layout()
    plt.savefig("plots/NS_example.png")

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
    p = len(l_bounds_NS)

    df_metro = pd.DataFrame(data['df_metro_values'],
                        columns=data['df_metro_columns'],
                        index=data['df_metro_index'])
    
    print("Data loaded successfully.")

    print("\nLoad neural network model...")
    classification_NN_NS = load_model("neural_networks/classification_NN_NS.h5")
    print("Model loaded successfully.")
    """
    #Define true
    theta_true = np.array([np.log(2), np.log(0.1), -0.2, 0.2])
    theta_normalized = (theta_true - params_mean) / params_std
    year = [2023]
    gs = len(SS_mean)
    theta_true_with_year = np.hstack([theta_true, year]).reshape((1, -1))
    print(theta_true_with_year)
    X_obs, Y_obs, metro_year_list_obs, _ = us.simulate_given_params(J=1, K=1, T=T, p=p, df_metro = df_metro, params=theta_true_with_year, verbose=False)
    """
    #Load true parameter and observed data from example_config
    print("\nLoading true parameter and observed data from example_config...")
    example_config = np.load("results/example4_2/theta_true.npz", allow_pickle=True)
    X_obs = example_config["X_obs"]
    Y_obs = example_config["Y_obs"]
    metro_year_list_obs = example_config["metro_year_list_obs"]
    theta_true = example_config["theta_true"]
    theta_normalized = (theta_true - params_mean) / params_std
    SS_obs = example_config["SS_obs"]
    SS_obs_normalized = (SS_obs - SS_mean) / SS_std
    print("True parameter and observed data loaded successfully.")
    year = [2023]
    gs = len(SS_mean)
    print("SS_obs_normalized = ", SS_obs_normalized,
          "theta_true = ", theta_true)
    print("\n Number of events in observed data:", len(X_obs[0]))

    #Prediction from NCL method
    print("\nPredicting parameters using the neural network model...")
    N_grid = 10000
    grid = us.LHS(N_grid, l_bounds_NS, u_bounds_NS, year)[:,:-1]
    grid_norm = (grid - params_mean) / params_std
    ll_grid = classification_NN_NS.predict([SS_obs_normalized.reshape((1,-1)).repeat(N_grid, axis = 0), grid_norm]).reshape(-1)  
    start = grid_norm[np.argmax(ll_grid)]
    MLE_NCL, MLE_NCL_normalized, final_logit, _, end_state = us.numerical_optim(start, SS_obs_normalized, l_bounds_NS, u_bounds_NS, classification_NN_NS, gs, params_mean=params_mean, params_std=params_std)

    #Platt scalar
    print("\nLoading Platt scaling factor...")
    platt_scaler = np.load("neural_networks/platt_scaler_NS.npy")
    print(f"Platt scaling factor: {platt_scaler}")

    #Godambe
    print("\nComputing Godambe information matrix and adjusted confidence intervals...")
    N_G = 70
    G_inv_NS, H_neg_NS_mean = godambe_G_NS(N_G, MLE_NCL.reshape((1,-1)), params_mean, params_std, year = year[0], gs = gs, p = p, T = T, cluster_bins = cluster_bins, percentiles = percentiles, df_metro = df_metro, classification_NN_NS = classification_NN_NS, SS_mean = SS_mean, SS_std = SS_std, plot = False)

    #Hessian
    print("\nComputing negative Hessian at the MLE...")
    H_neg_NS, _ = get_neg_hessian(MLE_NCL_normalized, SS_obs_normalized, gs, classification_NN_NS, p)

    #Load ABC posterior samples
    print("\nLoading ABC posterior samples...")
    ABC_samples = np.load("results/NS_ABC_example.npz", allow_pickle=True)
    posterior_samples_theta = ABC_samples["posterior_samples_theta"]
    posterior_SS = ABC_samples["posterior_samples_SS"]
    print("Mean number of events in ABC simulated data:", np.mean(np.exp(posterior_SS[:,0])))


    # NCL idea
    print("\nLoading trained models...")
    model = keras.models.load_model("neural_networks/idea_network/model.keras")
    F_point = keras.models.load_model("neural_networks/idea_network/F_point.keras")
    delta_net_path = "neural_networks/idea_network/delta_net.keras"
    delta_net = keras.models.load_model(delta_net_path) if os.path.exists(delta_net_path) else None
    curvature_head = model.get_layer("curvature_head")
    print("Models loaded.")

    mle_ncl_idea_normalized = ncl_utils.mle(S_obs=SS_obs_normalized.reshape((1,-1)), F_point=F_point, delta_net=delta_net).numpy()[0]  # squeeze batch dim
    mle_ncl_idea = mle_ncl_idea_normalized * params_std + params_mean  # denormalize to original space
    fisher_info_ncl_idea = ncl_utils.fisher_information(curvature_head=curvature_head, S_obs=SS_obs_normalized.reshape((1,-1)), p=p).numpy()[0]  # squeeze batch dim, shape (p, p)


    plt.figure()
    plt.hist(np.exp(posterior_SS[:,0]), bins=30, density=True)
    plt.title("Distribution of number of events in ABC simulated data")
    plt.xlabel("Number of events")
    plt.ylabel("Density")
    plt.savefig("plots/NS_ABC_simulated_events_example.png")
    plt.close()

    #Load NF posterior samples
    print("\nLoading normalizing flow posterior samples...")
    NF_samples = np.load("results/NF_samples_NS.npz", allow_pickle=True)
 
    #Plot 
    print("\nPlotting the results...")
    plot_example(NF_samples, posterior_samples_theta, MLE_NCL,
                  G_inv_NS, H_neg_NS, platt_scaler = platt_scaler,
                    theta_0_NS = theta_true, col_names_params_NS = col_names_params_NS,
                      params_mean = params_mean, params_std = params_std,
                        l_bounds_NS = l_bounds_NS, u_bounds_NS = u_bounds_NS,
                        mle_ncl_idea = mle_ncl_idea, fisher_info_ncl_idea = fisher_info_ncl_idea)

main()