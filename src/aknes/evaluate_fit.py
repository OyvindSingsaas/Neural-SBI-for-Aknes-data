import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils.utils_surface_NS as us
from scipy.stats import gaussian_kde


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
catalogue_path = "Surface_Catalogues"
df_events = us.read_cataloge(catalogue_path)
df_events.sort_index(inplace=True) #Sortere etter tidspunktet for observasjon i tilfelle det ble feil under nedlastning
df_events.index = df_events.index.tz_localize(None)
df_events = df_events.loc[df_events["Type"] == "Slope_HF"]
timestamps = df_events.index
timestamps = timestamps[(timestamps < df_metro.index.max()) & (timestamps > df_metro.index.min())]
print("Date range:", np.min(df_events.index), "  to  ", np.max(df_events.index))
years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
event_times = timestamps[timestamps.year.isin(years)]
event_times = event_times.to_numpy()
T = len(years)*365
p = 3

#NHPP model with covariates----------------------------
print("Fitting NHPP model with covariates...")
df_design = pd.DataFrame({
    "time": df_metro.index,               # assumes DateTimeIndex
    "temperature": df_metro.temperature,
    "wp": df_metro.wp,
    "N_geophones": df_metro["N_geophones"]
})
print("Data loaded and preprocessed successfully.")
def intensity(params, X):
    return np.exp(np.dot(X, params))

mle_nhpp = np.array([0.121, -0.82, -0.52, 1.5])
CI_nhpp = np.array([[-0.06,0.19], [-0.88,-0.76], [-0.58,-0.47], [1.43,1.57]])

t_grid = df_design["time"]
X = df_design.drop(columns="time").values  # all covariates

#Hawkes process model
mle_hawkes = np.array([-1.01, -0.44, -0.59, 1.51, 0.84, 1.26])
CI_hawkes = np.array([[-1.28,-0.75], [-0.68,-0.19], [-0.82,-0.36], [1.25,1.78], [0.74,0.94], [1.08,1.44]])
X_hawkes = df_design.drop(columns=["time", "intercept"]).values[:T]
hawkes_intensity = np.exp(mle_hawkes[0] + X_hawkes@mle_hawkes[1:-2])
hawkes_intensity = hawkes_intensity*(1 / (1 - (mle_hawkes[-2]/mle_hawkes[-1])))

#Loading NCL results
print("Loading NCL results...")
mle_ncl = []
CI_ncl = []
SS_observed = []
mask = timestamps.year.isin(years)
X_observed = timestamps[mask]
t0 = X_observed.min()
X_observed = (X_observed - t0).total_seconds() / 86400.0
X_observed = X_observed.to_numpy()

#Loading NF results
print("Loading NF results...")
mle_nf = []
CI_nf = []

#Loading abc results
print("Loading ABC results...")
mle_abc = []
CI_abc = []


# --- KDE-based intensity estimate
event_sec = event_times.astype("int64")/1e9/ 86400
t_sec = t_grid.astype("int64")/1e9/ 86400

kde = gaussian_kde(event_sec, bw_method=0.03)
kde_vals = kde(t_sec) * len(event_sec)  # scale to event rate

#year_str = str(year_observed)
X_cov = us.covariate_formater(df_metro, years, x_p = p - 3, T = T)

lambda_fitted_ncl = np.array([us.lambda_intensity(t, X_cov, mle_ncl[0][2:], error = 0) for t in range(T)]) * np.exp(mle_ncl[0][0])
lambda_fitted_NF = np.array([us.lambda_intensity(t, X_cov, mle_nf[:-1][2:], error = 0) for t in range(T)]) * np.exp(mle_nf[0])
lambda_fitted_abc = np.array([us.lambda_intensity(t, X_cov, mle_abc[:-1][2:], error = 0) for t in range(T)]) * np.exp(mle_abc[0])
kernel_density_estimate = us.kernel_intensity_estimate_reflect(X_observed, 21, domain= (0, T))
lambda_fitted_nhpp = intensity(mle_nhpp, df_design[df_design.index.year.isin(years)].drop(columns="time").values)

tt = np.arange(0, T, 1)
tt = pd.date_range(start=timestamps[timestamps.year.isin(years)][0], periods=T, freq="D")
fig, ax1 = plt.subplots(figsize=(16, 5))  # width=16, height=5

shade = 0.6
ax1.plot(tt, kernel_density_estimate, 'r-', label='KDE', linestyle = "--")
ax1.plot(tt, lambda_fitted_nhpp[:T], 'g-', label='NHPP', alpha = shade)
ax1.plot(tt, hawkes_intensity, 'orange', label='Hawkes ', alpha = shade)
ax1.plot(tt, lambda_fitted_ncl, 'b-', label='NSP (NCL)', alpha =shade)
ax1.plot(tt, lambda_fitted_NF, 'purple', label='NSP (NF)', alpha = shade)
ax1.set_ylabel('λ_fitted')
ax1.tick_params(axis='y')
"""
ax2 = ax1.twinx()
ax2.plot(tt, X_cov[0, :], 'purple', label='wp')
ax2.plot(tt, X_cov[1, :], 'pink', label='temp')
ax2.plot(tt, X_cov[2, :], 'yellow', label='N_geo')
ax2.set_ylabel('Cov value')
ax2.tick_params(axis='y')
"""
ax1.set_ylabel('Expected number of events per day')
plt.legend()
plt.xlabel('Time')
plt.title(f'KDE vs. Fitted Intensity Functions')
plt.show()
