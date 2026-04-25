import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils.utils_surface_NS as us
from scipy.stats import gaussian_kde

# Load the data
print("Loading data...")
data = np.load('data/aknes_training_data.npz', allow_pickle=True)
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
"""
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
catalogue_path = "data/Surface_Catalogues"
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
p = 6

print("Event times range:", np.min(event_times), " to ", np.max(event_times))

mask = timestamps.year.isin(years)
X_observed = timestamps[mask]
t0 = X_observed.min()
X_observed = (X_observed - t0).total_seconds() / 86400.0
X_observed = X_observed.to_numpy()
print("Event times range after normalization:", np.min(X_observed), " to ", np.max(X_observed))

percentiles = [10, 50, 90]
cluster_bins = np.logspace(np.log10(0.001), np.log10(10), 10)  # Logarithmically spaced bins from 0.001 to 10 days
SS_observed = us.summary_statistics(X_observed, T, cluster_bins, percentiles, df_metro, years, verbose=False, p = p)
print("Summary statistics for observed data:", SS_observed)
SS_observed_normalized = (SS_observed - SS_mean) / SS_std
print("Normalized summary statistics for observed data:", SS_observed_normalized)

"""
mle = np.array([1.65, -0.12, -1.53, -0.76, -0.36, 1.5])
N_sim = 100
mle_params = np.concatenate((mle, []))
"""
SS_train_true_pairs = SS_0_train_normalized_neural[response_train == 1]
#remove samples with low event coutns
SS_train_true_pairs_reduced = SS_train_true_pairs[np.exp(SS_train_true_pairs[:, 0]*SS_std[0] + SS_mean[0]) > 1000]
print("Number of training samples with true events and sufficient counts:", len(SS_train_true_pairs_reduced))

#set plotting parameters to make plot clear
plt.rcParams.update({'font.size': 14, 'lines.linewidth': 1.5, 'axes.linewidth': 1.2})

SS_names = [
    r"log(N)",
    r"$\hat{Q}_{0.1}(\Delta)$",
    r"$\hat{Q}_{0.5}(\Delta)$",
    r"$\hat{Q}_{0.9}(\Delta)$",
    r"$\bar \Delta / \hat{Q}_{0.5}(\Delta)$",
    r"$K_{r_1}(t)$",
    r"$K_{r_2}(t)$",
    r"$K_{r_3}(t)$",
    r"$K_{r_4}(t)$",
    r"$K_{r_5}(t)$",
    r"$K_{r_6}(t)$",
    r"$K_{r_7}(t)$",
    r"$K_{r_8}(t)$",
    r"$K_{r_9}(t)$",
    r"$K_{r_{10}}(t)$",
    r"$\sum_i wp(t_i)$",
    r"$\sum_i temp(t_i)$",
    r"$\sum_i Geo(t_i)$"
]

# ── High-dimensional placement analysis ──────────────────────────────────────
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from scipy.spatial import ConvexHull

obs = SS_observed_normalized.reshape(1, -1)  # (1, D)
train = SS_train_true_pairs_reduced               # (N, D)

# Subsample training data if large (keeps plots readable and t-SNE tractable)
N_max = 2000
rng = np.random.default_rng(42)
if len(train) > N_max:
    idx = rng.choice(len(train), N_max, replace=False)
    train_plot = train[idx]
else:
    train_plot = train

combined = np.vstack([train_plot, obs])  # obs is the last row
labels_train = np.zeros(len(train_plot))
label_obs = np.array([1])

# ── 1. PCA ───────────────────────────────────────────────────────────────────
pca = PCA(n_components=4, random_state=42)
pca_all = pca.fit_transform(combined)
pca_train = pca_all[:-1]
pca_obs   = pca_all[-1]
ev = pca.explained_variance_ratio_

from itertools import combinations
pairs = list(combinations(range(4), 2))  # 6 pairs
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(f"PCA  (total var explained: {ev.sum():.1%})", fontsize=13)

for ax, (i, j) in zip(axes.flatten(), pairs):
    ax.scatter(pca_train[:, i], pca_train[:, j], s=6, alpha=0.4,
               color="steelblue", label="Training")
    ax.scatter(pca_obs[i], pca_obs[j], s=120, marker="*",
               color="red", zorder=5, label="Observed")
    ax.set_xlabel(f"PC{i+1} ({ev[i]:.1%})")
    ax.set_ylabel(f"PC{j+1} ({ev[j]:.1%})")
    ax.set_title(f"PC{i+1} vs PC{j+1}")
    ax.legend(fontsize=14)
"""    
    try:
        hull = ConvexHull(pca_train[:, [i, j]])
        for simplex in hull.simplices:
            ax.plot(pca_train[simplex, i], pca_train[simplex, j],
                    "k-", lw=0.6, alpha=0.5)
    except Exception:
        pass
 """   


plt.tight_layout()
plt.savefig("results/aknes/SS_placement_pca.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plot saved to results/aknes/SS_placement_pca.png")

# ── 2. MDS ───────────────────────────────────────────────────────────────────
mds = MDS(n_components=2, random_state=42, normalized_stress="auto")
mds_all = mds.fit_transform(combined)
mds_train = mds_all[:-1]
mds_obs   = mds_all[-1]

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(mds_train[:, 0], mds_train[:, 1], s=6, alpha=0.4,
           color="steelblue", label="Training")
ax.scatter(*mds_obs, s=120, marker="*", color="red", zorder=5, label="Observed")
"""
try:
    hull = ConvexHull(mds_train)
    for simplex in hull.simplices:
        ax.plot(mds_train[simplex, 0], mds_train[simplex, 1],
                "k-", lw=0.6, alpha=0.5)
except Exception:
    pass
"""
ax.set_xlabel("Dim 1")
ax.set_ylabel("Dim 2")
ax.legend(fontsize=14)
plt.tight_layout()
plt.savefig("results/aknes/SS_placement_mds.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plot saved to results/aknes/SS_placement_mds.png")

# ── 3. t-SNE ─────────────────────────────────────────────────────────────────
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(train_plot) - 1))
tsne_all = tsne.fit_transform(combined)
tsne_train = tsne_all[:-1]
tsne_obs   = tsne_all[-1]

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(tsne_train[:, 0], tsne_train[:, 1], s=6, alpha=0.4,
           color="steelblue", label="Training")
ax.scatter(*tsne_obs, s=120, marker="*", color="red", zorder=5, label="Observed")
#ax.set_title("t-SNE")
ax.set_xlabel("Dim 1")
ax.set_ylabel("Dim 2")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("results/aknes/SS_placement_tsne.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plot saved to results/aknes/SS_placement_tsne.png")

# ── 4. Distance summary ───────────────────────────────────────────────────────
dists = np.linalg.norm(train - obs, axis=1)

try:
    from scipy.spatial import Delaunay
    tri = Delaunay(pca_train)
    inside = tri.find_simplex(pca_obs) >= 0
    hull_msg = "INSIDE convex hull (PCA)" if inside else "OUTSIDE convex hull (PCA)"
except Exception:
    hull_msg = "Convex hull check failed"

# ── LP-based exact convex hull membership in full D-dimensional space ─────────
# x is in conv(X) iff  X^T λ = x,  sum(λ) = 1,  λ >= 0  is feasible.
from scipy.optimize import linprog

def in_convex_hull_lp(X, x):
    """Return True if x lies inside the convex hull of rows of X."""
    N = X.shape[0]
    # Variables: λ (N,) + slack s (D,) for equality residual (split +/-)
    # Minimise 0 (feasibility LP).  Equalities: X^T λ = x, 1^T λ = 1.
    # We reformulate as an LP with bounds λ >= 0, no explicit slack needed
    # because linprog handles equality constraints directly.
    c = np.zeros(N)
    A_eq = np.vstack([X.T, np.ones((1, N))])   # (D+1, N)
    b_eq = np.append(x, 1.0)                   # (D+1,)
    bounds = [(0, None)] * N
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    return res.status == 0  # status 0 means optimal (feasible)

print("Running LP convex hull membership test in full space (may take a moment)...")
lp_inside = in_convex_hull_lp(SS_train_true_pairs, obs[0])
lp_hull_msg = "INSIDE convex hull (full D-dim, LP)" if lp_inside else "OUTSIDE convex hull (full D-dim, LP)"
print(f"LP result: {lp_hull_msg}")

summary_lines = [
    f"Training samples used: {len(train_plot):,}  (of {len(train):,})",
    f"SS dimensionality: {train.shape[1]}",
    "",
    f"Nearest-neighbour dist (orig. space): {dists.min():.4f}",
    f"Median dist to training: {np.median(dists):.4f}",
    f"Mean dist to training:   {np.mean(dists):.4f}",
    "",
    f"PCA hull (2D proxy):   {hull_msg}",
    f"LP  hull (exact, {train.shape[1]}D): {lp_hull_msg}",
    "",
    "PCA loadings of observed SS:",
    f"  PC1={pca_obs[0]:.3f},  PC2={pca_obs[1]:.3f}",
    "",
    "PCA variance explained per component:",
]
for i, v in enumerate(pca.explained_variance_ratio_):
    summary_lines.append(f"  PC{i+1}: {v:.1%}")
print("\n".join(summary_lines))

# ── Per-dimension histograms ──────────────────────────────────────────────────
D = train.shape[1]
ncols = 3
nrows = int(np.ceil(D / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows))
axes = axes.flatten()

for d in range(D):
    ax = axes[d]
    if d != 0 and d != 17 and d != 16 and d != 15:
        lower = np.percentile(train[:, d], 5)
        upper = np.percentile(train[:, d], 80)
        train_temp = train[:, d][(train[:, d] >= lower) & (train[:, d] <= upper)]
        ax.hist(train_temp, bins=50, density=True, color="steelblue",
        alpha=0.7, label="Training")
        ax.axvline(obs[0, d], color="red", linewidth=2, label="Observed")
        pct = np.mean(train[:, d] < obs[0, d]) * 100
        ax.set_title(f"{SS_names[d]}  [{pct:.0f}th pct]", fontsize=14)
        ax.tick_params(labelsize=7)
        #set xlim based on percentiles of training data

    else:
        ax.hist(train[:, d], bins=50, density=True, color="steelblue",
                alpha=0.7, label="Training")
        ax.axvline(obs[0, d], color="red", linewidth=2, label="Observed")
        pct = np.mean(train[:, d] < obs[0, d]) * 100
        ax.set_title(f"{SS_names[d]}  [{pct:.0f}th pct]", fontsize=14)
        ax.tick_params(labelsize=7)
        #lower = np.percentile(train[:, d], 5)
        #upper = np.percentile(train[:, d], 95)
        #ax.set_xlim(lower, upper)
    if d == 0:
        ax.legend(fontsize=7)

for d in range(D, len(axes)):
    axes[d].set_visible(False)

plt.tight_layout()
plt.savefig("results/aknes/SS_histograms.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plot saved to results/aknes/SS_histograms.png")
