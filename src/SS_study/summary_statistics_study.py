import numpy as np
import pandas as pd
import utils.utils_surface_NS as us
import matplotlib.pyplot as plt

#1. Investigate the behaviour of the K-function for varying sigma

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

def main():

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
    J = 20
    K = 10
    T = 365
    error = 0.0
    #log(delta), log(sigma^2), beta_0, beta_1, beta_2, beta_3, beta_4, beta_5, beta_6
    l_bounds_NS = np.array([np.log(0.1), np.log(0.001), -2, -2])
    u_bounds_NS = np.array([np.log(10),  np.log(7),  2, 2])


    p = len(l_bounds_NS)
    years = df_metro.index.year.unique().values[1:-1]
    years = years[years!=2019]
    years = [2023]

    #define the parameter data with varying sigma within bounds and fixed delta and betas
    sigma_values = np.linspace(1, 3, J)
    print("selected sigma values for simulation:", sigma_values)
    delta_fixed = np.log(10)
    betas_fixed = np.array([0.5, 0])

    parameter_data = []
    for sigma in sigma_values:
        theta = np.array([delta_fixed, np.log(sigma), *betas_fixed, 2023])
        parameter_data.append(theta)
    parameter_data = np.array(parameter_data)

    print("Simulating data ...")
    X, Y, metro_year_list_train, invalid_index_train = us.simulate_given_params(J=J, K=K, T=T, p=p, df_metro = df_metro, params=parameter_data, error = error, verbose=False)

    print("Mean number of events in the simulated data: ", np.mean([len(x) for x in X]))
    #log-distributed cluster_bins for the K-function
    cluster_bins = np.linspace(0.01, 10, 100)

    RK_values = np.zeros((J, len(cluster_bins)))
    for j in range(J):
        rk = np.zeros((K, len(cluster_bins)))
        for k in range(K):
            rk[k] = us.ripley_K(X[j*K + k], cluster_bins, T)
            rk[k] = rk[k] /(2*cluster_bins)-1
        RK_values[j] = np.mean(rk, axis=0)

    #Simulate a Poisson process with the same number of events as the mean number of events in the simulated data, and compute the K-function for it
    mean_num_events = int(np.mean([len(x) for x in X]))
    #take mean over each cosequtive K simulations to get the mean number of events for each sigma value
    mean_num_events_per_sigma = []
    for j in range(0, J*K, K):
        mean_num_events_per_sigma.append(int(np.mean([len(x) for x in X[j:j+K]])))

    RK_poisson = np.zeros((J, len(cluster_bins)))
    for i in range(J):
        x_poisson = np.random.uniform(0, T, mean_num_events_per_sigma[i])
        rk = us.ripley_K(x_poisson, cluster_bins, T)
        rk = rk /(2*cluster_bins)-1
        RK_poisson[i] = rk

    #Find the maximum deviation between the RK values for each sigma and the RK values for the Poisson process, and the index of the cluster bin where this maximum deviation occurs
    RK_max_deviation = np.zeros(J)
    RK_max_deviation_index = np.zeros(J, dtype=int)
    for i in range(J):
        deviation = RK_values[i] #np.abs(RK_values[i] - RK_poisson[i])
        RK_max_deviation[i] = np.max(deviation)
        RK_max_deviation_index[i] = np.argmax(deviation)

    plt.figure(figsize=(10, 6))
    for i in range(J):
        plt.plot(cluster_bins, RK_values[i], label=f"$\sigma^2$: {sigma_values[i]:.3f}")
        plt.vlines(cluster_bins[RK_max_deviation_index[i]], ymin=0, ymax=RK_values[i][RK_max_deviation_index[i]], color='red', linestyle='dashed', linewidth=1)
    #plt.plot(cluster_bins, RK_poisson[0], label="Poisson process", color='black', linestyle='dashed')
    plt.xscale('log')
    plt.xlabel("Distance")
    plt.legend()
    plt.grid()

    plt.savefig("plots/test/Ripley_K_varying_sigma.png")
    plt.close()

    #Plot max deviatoion vs sigma and the cluster bin where the max deviation occurs vs sigma
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(sigma_values, RK_max_deviation, marker='o')
    plt.xscale('log')
    plt.xlabel("$\sigma^2$")
    plt.ylabel("Max deviation from Poisson RK")
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(sigma_values, cluster_bins[RK_max_deviation_index], marker='o')
    plt.xlabel("$\sigma^2$")
    plt.ylabel("Cluster bin of max deviation")
    plt.grid()
    plt.tight_layout()
    plt.savefig("plots/test/Ripley_K_max_deviation_varying_sigma.png")
    plt.close()

main()