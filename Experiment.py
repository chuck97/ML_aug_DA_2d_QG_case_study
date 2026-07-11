"""
Experiment.py
-------------
Drives a data-assimilation experiment on the 2D QG model, in the spirit of the
Lorenz-96 reference Experiment: it holds the settings and results, builds an
initial ensemble, and cycles forecast -> analysis over the observation times.

Assimilation is delegated to a stochastic EnKF (EnKF.py) or, optionally, to an
ML analysis operator (``ml_model`` with an ``.assimilate(xf_mean, y, obs_idx)``
method) -- mirroring the reference, where the network can drive the update on
alternate cycles.

Observations come from ``02_sample_observations.ipynb`` (``obs_full.nc`` /
``obs_sparse.nc``): variable ``obs(time, y, x)`` with unobserved points NaN, an
integer ``mask(y, x)``, and attributes ``sigma_obs`` and ``dt_obs``. The truth
trajectory is used only for verification (analysis RMSE).
"""

import os

import numpy as np
import xarray as xr
from tqdm.auto import tqdm

from QG_2D import QG2D, _save_netcdf
from EnKF import EnKF


def _save_run(ds, path):
    """
    Save an EnKF-run Dataset to netCDF under (typically) da_data/.

    The 3-D/4-D fields are zlib-compressed to keep the (large) ensemble history
    manageable. Falls back to the lock-free scipy backend if the HDF5 backend is
    rejected (e.g. OneDrive), in which case compression is skipped.
    """
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass
    enc = {v: {"zlib": True, "complevel": 4}
           for v in ds.data_vars if ds[v].ndim >= 3}
    try:
        ds.to_netcdf(path, encoding=enc)                 # compressed netCDF4/HDF5
    except (PermissionError, OSError, RuntimeError, ValueError):
        _save_netcdf(ds, path)                           # scipy fallback (no zlib)


class Experiment:
    """
    :param model_params: dict of QG2D parameters (N, beta, mu, nu, p, A, k_f, L)
    :param truth: xr.Dataset truth trajectory (variable 'q'); verification target
    :param obs: xr.Dataset of observations (obs, mask, attrs sigma_obs/dt_obs)
    :param settings: dict with
        nens   : ensemble size
        dt     : integration step (MTU)
        gamma  : multiplicative inflation factor
        loc    : localization length scale (physical units; None = off)
        init_spread : initial ensemble std as a fraction of climatology
    """

    def __init__(self, model_params, truth, obs, settings):
        self.mp = dict(model_params)
        self.truth = truth
        self.obs = obs
        self.s = dict(settings)

        self.model = QG2D(**self.mp)
        self.N = self.model.N
        self.L = self.model.L
        self.dt = self.s["dt"]
        self.nens = self.s["nens"]
        self.r = float(obs.attrs["sigma_obs"])
        self.dt_obs = float(obs.attrs["dt_obs"])
        self.obs_every = int(round(self.dt_obs / self.dt))

        # observed indices (flattened, consistent with field.ravel())
        self.mask = np.asarray(obs["mask"].values, dtype=bool)
        self.obs_idx = np.flatnonzero(self.mask.ravel())

        # coordinates of every state element, for localization (torus distances)
        x = np.arange(self.N) * (self.L / self.N)
        X, Y = np.meshgrid(x, x, indexing="ij")
        self.coords = np.column_stack([X.ravel(), Y.ravel()])

        self.enkf = EnKF(self.coords, self.L,
                         gamma=self.s.get("gamma", 1.0),
                         loc=self.s.get("loc", None))
        self.enkf.set_observation_network(self.obs_idx)

    # ------------------------------------------------------------------ #
    def make_ensemble(self, x0, spread):
        """Initial ensemble: x0 + smooth (band-limited) perturbations."""
        clim = float(self.truth["q"].std())
        X = np.empty((self.N * self.N, self.nens))
        for j in range(self.nens):
            d = self.model.random_ic(seed=1000 + j, amp=1.0)
            d *= (spread * clim) / d.std()
            X[:, j] = (x0 + d).ravel()
        return X

    def _forecast_member(self, vec):
        """Integrate one flattened state forward by one observation interval."""
        qh = np.fft.fft2(vec.reshape(self.N, self.N))
        for _ in range(self.obs_every):
            qh = self.model.step(qh, self.dt)
        return np.real(np.fft.ifft2(qh)).ravel()

    # ------------------------------------------------------------------ #
    def assimilate(self, ncycles=None, method="enkf", ml_model=None, x0=None,
                   schedule=None, ml_obs=None, save_path=None, progress=True):
        """
        Run the cycled assimilation.

        :param ncycles: number of observation cycles to assimilate (default: all)
        :param method: 'enkf' or 'ml' (used when schedule is None)
        :param schedule: optional list applied per cycle, cycled with the cycle
                   index, of 'enkf' / 'ml' / 'skip' (skip = forecast only, no
                   analysis). Enables sequences like ['enkf','ml','skip'].
        :param ml_obs: optional observation Dataset used only on 'ml' steps
                   (defaults to the main obs). Lets an ML step read dense
                   observations while the EnKF steps use sparse ones.
        :param ml_model: object with .assimilate(xf_mean, y, obs_idx) -> analysis mean
        :param x0: background field the initial ensemble is centred on
                   (default: the truth at time 0; pass the control state for a
                   realistic offset background)
        :param save_path: if given, write the result Dataset to this path (e.g.
                   'da_data/enkf_run.nc'), compressed, with a scipy fallback.
        :return: xr.Dataset with, for both forecast and analysis, the ensemble
                 mean field and the per-grid-point ensemble spread field
                 (std across members) -- 'xf_mean'/'xa_mean' and
                 'xf_spread'/'xa_spread', dims (time, y, x) -- plus the RMSE and
                 domain-mean spread series. The full ensemble is not stored.
        """
        n = self.N * self.N
        times = self.obs["time"].values
        ntot = len(times) - 1
        nc = ntot if ncycles is None else min(ncycles, ntot)

        truth = self.truth["q"].values.reshape(len(times), n)
        obs_flat = self.obs["obs"].values.reshape(len(times), n)

        # optional separate observation set for ML steps (e.g. dense obs the
        # sparse EnKF cannot afford every cycle -- the Howard et al. idea)
        if ml_obs is not None:
            ml_obs_flat = ml_obs["obs"].values.reshape(len(times), n)
            ml_idx = np.flatnonzero(np.asarray(ml_obs["mask"].values, bool).ravel())
        else:
            ml_obs_flat, ml_idx = obs_flat, self.obs_idx

        # initial ensemble centred on the background x0 (offset from truth)
        if x0 is None:
            x0 = self.truth["q"].isel(time=0).values
        Xa = self.make_ensemble(x0, self.s.get("init_spread", 0.1))

        xa_mean = np.zeros((nc + 1, n))
        xf_mean = np.zeros((nc + 1, n))
        xa_spread = np.zeros((nc + 1, n))       # per-grid-point ensemble std
        xf_spread = np.zeros((nc + 1, n))
        rmse_a = np.full(nc + 1, np.nan)
        rmse_f = np.full(nc + 1, np.nan)
        spread_a = np.full(nc + 1, np.nan)      # domain-mean analysis spread

        xa_mean[0] = Xa.mean(axis=1)
        xf_mean[0] = xa_mean[0]
        xa_spread[0] = Xa.std(axis=1, ddof=1)
        xf_spread[0] = xa_spread[0]
        rmse_a[0] = np.sqrt(np.mean((xa_mean[0] - truth[0]) ** 2))

        it = tqdm(range(1, nc + 1), desc="assimilating", disable=not progress)
        for i in it:
            # forecast every member one observation interval
            Xf = np.empty_like(Xa)
            for j in range(self.nens):
                Xf[:, j] = self._forecast_member(Xa[:, j])
            xf_mean[i] = Xf.mean(axis=1)
            xf_spread[i] = Xf.std(axis=1, ddof=1)
            rmse_f[i] = np.sqrt(np.mean((xf_mean[i] - truth[i]) ** 2))

            y = obs_flat[i][self.obs_idx]              # observed values this cycle

            step = schedule[(i - 1) % len(schedule)] if schedule else method
            if step == "skip":
                Xa = Xf                                     # forecast only, no update
            elif step == "ml" and ml_model is not None:
                # ML returns an analysis *mean*; shift every member by that
                # increment (recenter the ensemble on the ML analysis). The ML
                # step may read a different (e.g. dense) observation set.
                y_ml = ml_obs_flat[i][ml_idx]
                xa = ml_model.assimilate(xf_mean[i], y_ml, ml_idx)
                Xa = Xf + (xa - xf_mean[i])[:, None]
            else:
                Xa = self.enkf.analysis(Xf, y, self.r)

            xa_mean[i] = Xa.mean(axis=1)
            xa_spread[i] = Xa.std(axis=1, ddof=1)
            rmse_a[i] = np.sqrt(np.mean((xa_mean[i] - truth[i]) ** 2))
            spread_a[i] = np.sqrt(np.mean(xa_spread[i] ** 2))   # domain-RMS spread

        yv = np.arange(self.N) * (self.L / self.N)
        dims3 = ("time", "y", "x")

        def _f32(a):
            return a.reshape(nc + 1, self.N, self.N).astype(np.float32)

        data = {
            "xa_mean": (dims3, _f32(xa_mean)),
            "xf_mean": (dims3, _f32(xf_mean)),
            "xa_spread": (dims3, _f32(xa_spread)),
            "xf_spread": (dims3, _f32(xf_spread)),
            "rmse_a": (("time",), rmse_a),
            "rmse_f": (("time",), rmse_f),
            "spread_a": (("time",), spread_a),
        }
        coords = {"time": times[: nc + 1], "y": yv, "x": yv}
        out = xr.Dataset(
            data, coords=coords,
            attrs={**self.mp, "nens": self.nens, "gamma": self.s.get("gamma", 1.0),
                   "loc": self.s.get("loc", None) if self.s.get("loc") is not None else -1,
                   "sigma_obs": self.r, "dt_obs": self.dt_obs,
                   "coverage": float(self.obs.attrs.get("coverage", self.mask.mean())),
                   "method": method},
        )
        if save_path is not None:
            _save_run(out, save_path)
        return out

    # ------------------------------------------------------------------ #
    def free_run_rmse(self, control, ncycles=None):
        """Baseline: RMSE of the undisturbed control vs truth (no assimilation)."""
        n = self.N * self.N
        times = self.obs["time"].values
        nc = (len(times) - 1) if ncycles is None else ncycles
        t = truth = self.truth["q"].values.reshape(len(times), n)[: nc + 1]
        c = control["q"].values.reshape(len(control["time"]), n)[: nc + 1]
        return np.sqrt(((c - t) ** 2).mean(axis=1))
