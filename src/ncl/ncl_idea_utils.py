import numpy as np
import tensorflow as tf
from scipy.stats import norm, chi2
from keras import layers, ops
import keras



def build_point_predictor(summary_dim, theta_dim, hidden=128):
    """F_point: S -> theta_hat. Pre-train with MSE."""
    S = layers.Input(shape=(summary_dim,), name="S")
    x = layers.Dense(hidden, activation="silu")(S)
    x = layers.Dense(hidden, activation="silu")(x)
    theta_hat = layers.Dense(theta_dim, name="theta_hat")(x)
    return keras.Model(S, theta_hat, name="F_point")


@keras.saving.register_keras_serializable(package="ncl")
class CurvatureHead(layers.Layer):
    """
    g(u, S) = -||h(u, S) - h(0, S)||^2
    Guarantees argmax_u g(u, S) = 0 and g(0, S) = 0.
    The overall scale/offset is absorbed into a learned bias b(S)
    so the classifier logit is g(u, S) + b(S).
    """
    def __init__(self, hidden=128, feat_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.hidden = hidden
        self.feat_dim = feat_dim
        self.h1 = layers.Dense(hidden, activation="silu")
        self.h2 = layers.Dense(hidden, activation="silu")
        self.h_out = layers.Dense(feat_dim)
        self.b1 = layers.Dense(hidden, activation="silu")
        self.b_out = layers.Dense(1)

    def _h(self, u, S):
        x = ops.concatenate([u, S], axis=-1)
        x = self.h1(x)
        x = self.h2(x)
        return self.h_out(x)

    def call(self, inputs):
        u, S = inputs                           # u = theta_hat - theta
        h_u = self._h(u, S)
        h_0 = self._h(ops.zeros_like(u), S)
        g = -ops.sum(ops.square(h_u - h_0), axis=-1, keepdims=True)
        b = self.b_out(self.b1(S))              # log-likelihood scale, depends on S
        return g + b                            # logit = log L(theta; S) + const

    def get_config(self):
        config = super().get_config()
        config.update({"hidden": self.hidden, "feat_dim": self.feat_dim})
        return config


def build_delta_net(summary_dim, theta_dim, hidden=128):
    """Mode correction: S -> small shift added to theta_hat."""
    S = layers.Input(shape=(summary_dim,), name="S")
    x = layers.Dense(hidden, activation="silu")(S)
    delta = layers.Dense(theta_dim, kernel_initializer="zeros", name="delta")(x)
    return keras.Model(S, delta, name="delta_net")


def build_neural_likelihood(summary_dim, theta_dim,
                            point_predictor=None,
                            delta_net=None,
                            freeze_point=False,
                            use_mode_correction=False):
    theta = layers.Input(shape=(theta_dim,), name="theta")
    S = layers.Input(shape=(summary_dim,), name="S")

    if point_predictor is None:
        point_predictor = build_point_predictor(summary_dim, theta_dim)
    point_predictor.trainable = not freeze_point
    theta_hat = point_predictor(S)

    if use_mode_correction:
        if delta_net is None:
            delta_net = build_delta_net(summary_dim, theta_dim)
        theta_hat = layers.Add()([theta_hat, delta_net(S)])

    u = layers.Subtract()([theta_hat, theta])
    curvature_head = CurvatureHead()           # keep a handle
    logit = curvature_head([u, S])

    model = keras.Model([theta, S], logit, name="neural_likelihood")
    return model, point_predictor, delta_net, curvature_head

def mle(S_obs, F_point, delta_net=None):
    result = F_point(S_obs, training=False)
    if delta_net is not None:
        result = result + delta_net(S_obs, training=False)
    return result

# Inference: closed-form MLE
def mle_point(S_obs, F_point, delta_net=None):
    result = F_point(S_obs, training=False)
    if delta_net is not None:
        result = result + delta_net(S_obs, training=False)
    return result

@tf.function
def fisher_information(curvature_head, S_obs, p):
    """Closed-form observed Fisher info: I(S) = 2 J(S)^T J(S)."""
    batch_size = tf.shape(S_obs)[0]
    theta_dim = p
    u0 = tf.zeros((batch_size, theta_dim), dtype=S_obs.dtype)
    
    with tf.GradientTape() as tape:
        tape.watch(u0)
        h_val = curvature_head._h(u0, S_obs)   # (batch, feat_dim)
    
    J = tape.batch_jacobian(h_val, u0)         # (batch, feat_dim, theta_dim)
    fisher = 2.0 * tf.einsum('bfi,bfj->bij', J, J)
    return fisher


def mle_with_covariance(F_point, delta_net, curvature_head, S_obs):
    theta_star = F_point(S_obs, training=False)
    if delta_net is not None:
        theta_star = theta_star + delta_net(S_obs, training=False)
    p = theta_star.shape[-1]
    fisher = fisher_information(curvature_head, S_obs, p)
    cov = tf.linalg.inv(fisher)
    return theta_star, cov


def empirical_coverage(F_point, delta_net, curvature_head, S_test, theta_test,
                       params_mean=None, params_std=None,
                       levels=(0.50, 0.80, 0.90, 0.95, 0.99)):
    """
    Empirical coverage of Wald-type confidence regions from the curvature head.

    Parameters
    ----------
    F_point : Keras model
    delta_net : Keras model or None
    curvature_head : CurvatureHead layer
    S_test : array, shape (N, summary_dim)
        Normalized test summary statistics.
    theta_test : array, shape (N, theta_dim)
        Normalized true parameters.
    params_mean : array, shape (theta_dim,) or None
        Mean used for normalization. If provided, RMSE is computed on the
        original scale.
    params_std : array, shape (theta_dim,) or None
        Std used for normalization.
    levels : tuple of float
        Nominal confidence levels to evaluate.

    Returns
    -------
    dict with keys:
        'marginal'      : (n_levels, theta_dim)  per-coordinate coverage
        'joint'         : (n_levels,)            joint elliptical-region coverage
        'mle'           : (N, theta_dim)         normalized point estimates
        'cov'           : (N, theta_dim, theta_dim) covariance matrices
        'z_scores'      : (N, theta_dim)         standardized residuals
        'mahal_sq'      : (N,)                   squared Mahalanobis distances
        'rmse_per_param': (theta_dim,)           per-parameter RMSE (original scale if params_mean/std given)
        'rmse_total'    : float                  total RMSE across all parameters
    """
    S_test = tf.constant(S_test, dtype=tf.float32)
    theta_test = np.asarray(theta_test, dtype=np.float32)
    N, theta_dim = theta_test.shape

    # Forward pass: MLE and covariance for all test points
    theta_hat, cov = mle_with_covariance(F_point, delta_net, curvature_head, S_test)
    theta_hat = theta_hat.numpy()
    cov = cov.numpy()

    # Per-coordinate standardized residuals (normalized space)
    residual = theta_test - theta_hat                          # (N, d)
    std = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))          # (N, d)
    z = residual / std                                         # (N, d)

    # Joint Mahalanobis distance squared, ~ chi2(d) under correct calibration
    cov_inv = np.linalg.inv(cov)
    mahal_sq = np.einsum('ni,nij,nj->n', residual, cov_inv, residual)

    levels = np.asarray(levels)

    # Marginal coverage: per coordinate, |z| < z_{(1+level)/2}
    z_thresh = norm.ppf(0.5 + levels / 2)                      # (n_levels,)
    marginal = np.zeros((len(levels), theta_dim))
    for k, zt in enumerate(z_thresh):
        marginal[k] = (np.abs(z) < zt).mean(axis=0)

    # Joint coverage: mahal_sq < chi2_{d, level}
    chi2_thresh = chi2.ppf(levels, df=theta_dim)               # (n_levels,)
    joint = np.array([(mahal_sq < t).mean() for t in chi2_thresh])

    # RMSE on original scale if normalization stats provided, else normalized scale
    if params_mean is not None and params_std is not None:
        params_mean = np.asarray(params_mean, dtype=np.float32)
        params_std = np.asarray(params_std, dtype=np.float32)
        true_denorm = theta_test * params_std + params_mean
        hat_denorm  = theta_hat  * params_std + params_mean
    else:
        true_denorm = theta_test
        hat_denorm  = theta_hat

    per_sample_se = (true_denorm - hat_denorm) ** 2            # (N, d)
    rmse_per_param = np.sqrt(per_sample_se.mean(axis=0))       # (d,)
    rmse_total = np.sqrt(per_sample_se.mean())

    return {
        'levels': levels,
        'marginal': marginal,
        'joint': joint,
        'mle': theta_hat,
        'cov': cov,
        'z_scores': z,
        'mahal_sq': mahal_sq,
        'rmse_per_param': rmse_per_param,
        'rmse_total': rmse_total,
    }


def coverage_report(results, param_names=None):
    """Pretty-print a coverage table and RMSE summary."""
    levels = results['levels']
    marg = results['marginal']
    joint = results['joint']
    rmse_per_param = results['rmse_per_param']
    rmse_total = results['rmse_total']
    d = marg.shape[1]
    if param_names is None:
        param_names = [f"θ_{i}" for i in range(d)]

    header = "Level   " + "  ".join(f"{n:>8}" for n in param_names) + "    Joint"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for k, lvl in enumerate(levels):
        marg_str = "  ".join(f"{marg[k, i]:>8.3f}" for i in range(d))
        print(f"{lvl:.2f}    {marg_str}    {joint[k]:.3f}")
    print(sep)
    rmse_str = "  ".join(f"{rmse_per_param[i]:>8.4f}" for i in range(d))
    print(f"RMSE    {rmse_str}    {rmse_total:.4f} (total)")
    print(sep)


def coverage_diagnostics(results):
    """
    Additional diagnostics: full coverage curve and chi2 calibration.
    Useful for plots in the paper.
    """
    z = results['z_scores']
    mahal_sq = results['mahal_sq']
    d = z.shape[1]

    # Smooth coverage curves: nominal vs empirical, fine grid
    nominal = np.linspace(0.01, 0.99, 99)
    z_thresh = norm.ppf(0.5 + nominal / 2)
    chi2_thresh = chi2.ppf(nominal, df=d)

    marginal_curve = np.stack([
        (np.abs(z) < zt).mean(axis=0) for zt in z_thresh
    ], axis=0)                                                 # (99, d)
    joint_curve = np.array([(mahal_sq < t).mean() for t in chi2_thresh])

    return {
        'nominal': nominal,
        'marginal_curve': marginal_curve,
        'joint_curve': joint_curve,
    }