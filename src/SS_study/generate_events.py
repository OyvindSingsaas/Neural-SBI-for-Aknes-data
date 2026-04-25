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
    J_train = 5000
    J_test = 50
    K = 3
    T = 1*365
    error = 0.0
    print("Generating data for NS model with J_train =", J_train, ", J_test =", J_test, ", K =", K, ", T =", T, ", error =", error)
    #log(delta), log(sigma^2), beta_0, beta_1, beta_2, beta_3, beta_4, beta_5, beta_6
    l_bounds_NS = np.array([np.log(0.1), np.log(0.001), -1., -1., -1.])
    u_bounds_NS = np.array([np.log(10),  np.log(3),  1., 1., 1.])

    l_bounds_NS_test = np.array([np.log(1), np.log(0.01), -0.5, -0.5, -0.5])
    u_bounds_NS_test = np.array([np.log(3),  np.log(1),  0.5, 0.5, 0.5])

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


    np.savez(
        "data/NS_events.npz",
        X_train=np.array(X_train, dtype=object),
        Y_train=np.array(Y_train, dtype=object),
        metro_year_list_train=np.array(metro_year_list_train, dtype=object),
        invalid_index_train=np.array(invalid_index_train, dtype=object),
        X_test=np.array(X_test, dtype=object),
        Y_test=np.array(Y_test, dtype=object),
        metro_year_list_test=np.array(metro_year_list_test, dtype=object),
        invalid_index_test=np.array(invalid_index_test, dtype=object),
        l_bounds_NS=l_bounds_NS,
        u_bounds_NS=u_bounds_NS,
        l_bounds_NS_test=l_bounds_NS_test,
        u_bounds_NS_test=u_bounds_NS_test,
        params_sample_test_NS=params_sample_test_NS,
        params_sample_train_NS=params_sample_train_NS
    )
    print("Done generating data for NS model.")


main()