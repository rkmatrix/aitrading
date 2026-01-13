import numpy as np


def project_simplex(w: np.ndarray, min_w: float = 0.0) -> np.ndarray:
    """Project vector onto probability simplex (Duchi et al., 2008)."""
    w = np.asarray(w, float)
    n = len(w)
    if min_w > 0:
        w -= min_w
    u = np.sort(w)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    w = np.maximum(w - theta, 0)
    if min_w > 0:
        w += min_w
    w /= w.sum() + 1e-12
    return w


def clamp_trust_region(w_new: np.ndarray, w_prev: np.ndarray, l1_limit: float) -> np.ndarray:
    """Ensure L1 change ≤ limit by interpolation."""
    w_new = np.asarray(w_new, float)
    w_prev = np.asarray(w_prev, float)
    l1 = np.abs(w_new - w_prev).sum()
    if l1 <= l1_limit:
        return w_new
    alpha = l1_limit / (l1 + 1e-12)
    w = alpha * w_new + (1 - alpha) * w_prev
    return project_simplex(w)
