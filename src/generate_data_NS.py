import numpy as np
import pandas as pd
import utils.utils_surface_NS as us

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
    """
    catalogue_path = "Surface_Catalogues"
    df_events = us.read_cataloge(catalogue_path)
    df_events.sort_index(inplace=True) #Sortere etter tidspunktet for observasjon i tilfelle det ble feil under nedlastning
    df_events.index = df_events.index.tz_localize(None)

    df_events = df_events.loc[df_events["Type"] == "Slope_HF"]

    timestamps = df_events.index
    timestamps = timestamps[(timestamps < df_metro.index.max()) & (timestamps > df_metro.index.min())]
    print("Date range:", np.min(df_events.index), "  to  ", np.max(df_events.index))
    year_dict = {}
    """
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
    J_train = 10000
    J_test = 100
    K = 3
    T = 1*365
    error = 0.0
    print("Generating data for NS model with J_train =", J_train, ", J_test =", J_test, ", K =", K, ", T =", T, ", error =", error)
    #log(delta), log(sigma^2), beta_0, beta_1, beta_2, beta_3, beta_4, beta_5, beta_6
    l_bounds_NS = np.array([np.log(0.1), np.log(0.001), -2.5, -2.5, -2.5])
    u_bounds_NS = np.array([np.log(10),  np.log(3),  2.5, 2.5, 2.5])

    l_bounds_NS_test = np.array([np.log(1), np.log(0.01), -1, -1, -1])
    u_bounds_NS_test = np.array([np.log(3),  np.log(2),  1, 1, 1])

    p = len(l_bounds_NS)
    years = df_metro.index.year.unique().values[1:-1]
    years = years[years!=2019]
    years = [2023]

    params_sample_train_NS = us.LHS(J_train, l_bounds_NS, u_bounds_NS, years)
    params_sample_test_NS = us.LHS(J_test, l_bounds_NS_test, u_bounds_NS_test, years)

    print("Simulating data for training and test sets...")
    X_train, Y_train, metro_year_list_train, invalid_index_train = us.simulate_given_params(J=J_train, K=K, T=T, p=p, df_metro = df_metro, params=params_sample_train_NS, error = error, verbose=False)
    X_test, Y_test, metro_year_list_test, invalid_index_test = us.simulate_given_params(J=J_test, K=K, T=T, p=p, df_metro = df_metro, params=params_sample_test_NS, error = error, verbose=False)

    print("Max number of events in a year = ", np.max([x.shape[0] for x in X_train]))
    print("Min number of events in a year = ", np.min([x.shape[0] for x in X_train]))
    print("Data shape = ", len(X_train))

    percentiles = [10, 50, 90]
    cluster_bins = np.logspace(np.log10(0.001), np.log10(10), 20)
    #cluster_bins = [0.001, 0.01, 0.1, 0.5, 1., 5., 10.]

    gs = len(l_bounds_NS) + len(cluster_bins) + len(percentiles) - 1 
    SS_0_train = np.zeros((len(X_train), gs))
    SS_0_test = np.zeros((len(X_test), gs))

    print("Computing summary statistics for training set...")
    for i,x in enumerate(X_train):
        SS_0_train[i,:] = us.summary_statistics(x, T, cluster_bins, percentiles, df_metro, metro_year_list_train[i], verbose=False, p = p)
        #print every 10 percent completed
        if i % (len(X_train) // 10) == 0:
            print(round(i/len(X_train)*100), "% done")
    print("Computing summary statistics for test set...")
    for i,x in enumerate(X_test):
        SS_0_test[i,:] = us.summary_statistics(x, T, cluster_bins, percentiles, df_metro, metro_year_list_test[i], verbose=False, p = p)
        if i % (len(X_test) // 10) == 0:
            print(round(i/len(X_test)*100), "% done")

    SS_mean = np.mean(SS_0_train, axis=0)
    SS_std = np.std(SS_0_train, axis=0)

    SS_0_train_normalized = (SS_0_train - SS_mean) / SS_std
    SS_0_test_normalized = (SS_0_test - SS_mean) / SS_std

    Y_shuffled_train = params_sample_train_NS[:, :-1]
    np.random.shuffle(Y_shuffled_train)
    Y_shuffled_train = Y_shuffled_train.repeat(K, axis = 0)
    params_train = np.concatenate([Y_train, Y_shuffled_train])
    params_train_unshuffled_copy = params_train.copy()
    response_train = np.concatenate([np.repeat(1, len(Y_train)), np.repeat(0, len(Y_train))])

    SS_0_train_normalized_neural = np.concatenate([SS_0_train_normalized, SS_0_train_normalized])

    print(SS_0_train_normalized_neural.shape)

    #--------------------

    Y_shuffled_test = params_sample_test_NS[:, :-1]
    np.random.shuffle(Y_shuffled_test)
    Y_shuffled_test = Y_shuffled_test.repeat(K, axis = 0)
    params_test = np.concatenate([Y_test, Y_shuffled_test])
    response_test = np.concatenate([np.repeat(1, len(Y_test)), np.repeat(0, len(Y_test))])

    SS_0_test_normalized_neural = np.concatenate([SS_0_test_normalized, SS_0_test_normalized])

    print(SS_0_test_normalized_neural.shape)

    #--------------------
    params_train_not_shuffled = params_sample_train_NS[:, :-1].copy()
    SS_0_train_normalized_not_shuffled = SS_0_train_normalized.copy()

    params_mean = np.mean(params_train, axis=0)
    params_std = np.std(params_train, axis=0)


    response_train, params_train, SS_0_train_normalized_neural = us.shuffle_data(response_train, params_train, SS_0_train_normalized_neural)
    response_test, params_test, SS_0_test_normalized_neural = us.shuffle_data(response_test, params_test, SS_0_test_normalized_neural)

    #X_global_mean = np.mean(X_global_train, axis=0)
    #X_global_std = np.std(X_global_train, axis=0)

    # Normalize the training sets
    params_train_normalized = (params_train - params_mean) / params_std

    # Normalize the test sets using the training set statistics
    params_test_normalized = (params_test - params_mean) / params_std

    print("Saving data...")
    #Data to save: params_train_normalized, SS_0_train_normalized_neural, response_train,
    #  params_test_normalized, SS_0_test_normalized_neural, response_test,
    #  params_mean, params_std, SS_mean, SS_std
    np.savez('data/NS_data_SS_study.npz', params_train_normalized=params_train_normalized, SS_0_train_normalized_neural=SS_0_train_normalized_neural, response_train=response_train,
             params_test_normalized=params_test_normalized, SS_0_test_normalized_neural=SS_0_test_normalized_neural, response_test=response_test,
             params_mean=params_mean, params_std=params_std, SS_mean=SS_mean, SS_std=SS_std, l_bounds_NS = l_bounds_NS, u_bounds_NS = u_bounds_NS,
             l_bounds_NS_test = l_bounds_NS_test, u_bounds_NS_test = u_bounds_NS_test, T = T, cluster_bins = cluster_bins, percentiles = percentiles, col_names_params_NS = col_names_params_NS,
            df_metro_values=df_metro.values, df_metro_columns=df_metro.columns.values, df_metro_index=df_metro.index.values,)
    print("Done generating data for NS model.")


main()