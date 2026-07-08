"""
EnKF.py
-------
Stochastic (perturbed-observations) Ensemble Kalman Filter for the 2D QG model.

The state is a vorticity field flattened to a vector of length ``n = N*N``. The
analysis is computed in the efficient ensemble/observation-space form, so the
full ``n x n`` covariance is never built:

    Pf H^T ~ (1/(m-1)) Xp (H Xp)^T          (n x p)
    H Pf H^T ~ (1/(m-1)) (H Xp)(H Xp)^T      (p x p)
    K = Pf H^T (H Pf H^T + R)^-1
    x_a^(j) = x_f^(j) + K (y + eps_j - H x_f^(j))     (perturbed obs -> stochastic)

with ``m`` the ensemble size and ``p`` the number of observations.

Two ingredients that make the filter work at small ensemble size:

* **Multiplicative inflation** (``gamma``): the forecast perturbations are scaled
  by ``sqrt(gamma)`` before the update, counteracting the systematic
  underestimation of spread by a small ensemble.
* **Localization** (``loc``): the sampling noise in far-field covariances is
  suppressed by a Gaspari-Cohn taper applied (Schur product) to ``Pf H^T`` and
  ``H Pf H^T``. Distances are computed on the doubly-periodic torus. ``loc`` is
  the taper length scale (the weight reaches zero at ``2*loc``).
"""

import numpy as np


def gaspari_cohn(dist, c):
    """Gaspari-Cohn (1999) 5th-order compactly-supported correlation.

    :param dist: array of distances
    :param c: length scale; the taper is positive on [0, 2c) and zero beyond
    :return: taper weights in [0, 1], same shape as ``dist``
    """
    a = np.asarray(dist, dtype=float) / c
    w = np.zeros_like(a)
    m1 = a <= 1.0
    m2 = (a > 1.0) & (a <= 2.0)
    a1 = a[m1]
    w[m1] = (((-0.25 * a1 + 0.5) * a1 + 0.625) * a1 - 5.0 / 3.0) * a1 ** 2 + 1.0
    a2 = a[m2]
    w[m2] = ((1.0 / 12.0) * a2 ** 5 - 0.5 * a2 ** 4 + 0.625 * a2 ** 3
             + 5.0 / 3.0 * a2 ** 2 - 5.0 * a2 + 4.0 - (2.0 / 3.0) / a2)
    return w


class EnKF:
    """
    Stochastic EnKF with inflation and Gaspari-Cohn localization.

    :param coords: (n, 2) physical coordinates of each state element
    :param L: domain size (for periodic distances)
    :param gamma: multiplicative covariance inflation factor (1.0 = none)
    :param loc: localization length scale in physical units (None = no localization)
    """

    def __init__(self, coords, L, gamma=1.0, loc=None):
        self.coords = np.asarray(coords, dtype=float)
        self.L = float(L)
        self.gamma = float(gamma)
        self.loc = loc
        self._obs_idx = None
        self._rho_xy = None      # (n, p) localization state<->obs
        self._rho_yy = None      # (p, p) localization obs<->obs

    # ------------------------------------------------------------------ #
    def _periodic_dist(self, A, B):
        """Pairwise torus distance between point sets A (a,2) and B (b,2) -> (a,b)."""
        dx = np.abs(A[:, 0:1] - B[None, :, 0]); dx = np.minimum(dx, self.L - dx)
        dy = np.abs(A[:, 1:2] - B[None, :, 1]); dy = np.minimum(dy, self.L - dy)
        return np.sqrt(dx ** 2 + dy ** 2)

    def set_observation_network(self, obs_idx):
        """
        Fix which state indices are observed and precompute localization matrices.
        Call once whenever the observation network changes (here: fixed in time).
        """
        self._obs_idx = np.asarray(obs_idx, dtype=int)
        if self.loc is not None:
            obs_c = self.coords[self._obs_idx]
            self._rho_xy = gaspari_cohn(self._periodic_dist(self.coords, obs_c), self.loc)
            self._rho_yy = gaspari_cohn(self._periodic_dist(obs_c, obs_c), self.loc)
        else:
            self._rho_xy = self._rho_yy = None

    # ------------------------------------------------------------------ #
    def analysis(self, Xf, y, r):
        """
        One stochastic EnKF analysis step.

        :param Xf: forecast ensemble, shape (n, m)
        :param y: observation vector at the observed indices, shape (p,)
        :param r: observation error standard deviation
        :return: analysis ensemble, shape (n, m)
        """
        if self._obs_idx is None:
            raise RuntimeError("call set_observation_network(obs_idx) first")
        n, m = Xf.shape
        idx = self._obs_idx
        p = idx.size

        # multiplicative inflation of the forecast ensemble
        xbar = Xf.mean(axis=1, keepdims=True)
        if self.gamma != 1.0:
            Xf = xbar + np.sqrt(self.gamma) * (Xf - xbar)

        Xp = Xf - Xf.mean(axis=1, keepdims=True)      # (n, m)
        HXp = Xp[idx]                                  # (p, m)

        PfHt = (Xp @ HXp.T) / (m - 1)                  # (n, p)
        HPfHt = (HXp @ HXp.T) / (m - 1)                # (p, p)
        if self._rho_xy is not None:
            PfHt = PfHt * self._rho_xy
            HPfHt = HPfHt * self._rho_yy

        S = HPfHt + (r ** 2) * np.eye(p)               # (p, p)

        # perturbed observations -> stochastic update
        D = y[:, None] + np.random.normal(0.0, r, size=(p, m))   # (p, m)
        innov = D - Xf[idx]                            # (p, m)
        Xa = Xf + PfHt @ np.linalg.solve(S, innov)     # (n, m)
        return Xa
