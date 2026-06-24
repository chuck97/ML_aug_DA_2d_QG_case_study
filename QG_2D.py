import numpy as np


class QG2D:
    """
    Single-layer, doubly-periodic quasi-geostrophic vorticity model.

    Prognostic variable: relative vorticity q = laplacian(psi) on an N x N grid.
    Dynamics (beta-plane, forced-dissipative):

        dq/dt + J(psi, q) + beta * dpsi/dx = F - mu * q - nu * (-lap)^p q

      psi   : streamfunction, with q = lap(psi)  ->  psi_hat = -q_hat / K^2
      J     : Jacobian / advection u . grad(q),  u = (-dpsi/dy, dpsi/dx)
      beta  : planetary vorticity gradient (Rossby waves)
      F     : forcing (default: Kolmogorov forcing  A*cos(k_f * y))
      mu    : linear (Ekman) drag, removes energy at large scales
      nu, p : hyperviscosity coefficient and order, removes enstrophy at the grid scale

    Solved pseudospectrally (FFT) with 2/3-rule dealiasing and fixed-step RK4.
    The API mirrors L96.py: build with settings, then `derivative` / `integrate`.
    """

    def __init__(self, N=64, L=2 * np.pi, beta=0.0, mu=0.1,
                 nu=1.0e-3, p=2, A=4.0, k_f=4):
        self.N = N
        self.L = L
        self.beta = beta
        self.mu = mu
        self.nu = nu
        self.p = p
        self.A = A
        self.k_f = k_f

        # --- spectral grid ---------------------------------------------------
        k = 2 * np.pi * np.fft.fftfreq(N, d=L / N)          # wavenumbers
        self.kx, self.ky = np.meshgrid(k, k, indexing="ij")
        self.K2 = self.kx ** 2 + self.ky ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            self.K2_inv = np.where(self.K2 == 0, 0.0, 1.0 / self.K2)  # invert lap, skip mean mode

        # linear operator (diagonal in spectral space): drag + hyperviscosity
        self.lin = -self.mu - self.nu * self.K2 ** self.p

        # 2/3 dealiasing mask
        kmax = np.abs(k).max()
        self.dealias = (np.abs(self.kx) <= (2.0 / 3.0) * kmax) & \
                       (np.abs(self.ky) <= (2.0 / 3.0) * kmax)

        # physical-space coordinates and forcing field
        x = np.arange(N) * (L / N)
        self.X, self.Y = np.meshgrid(x, x, indexing="ij")
        F = self.A * np.cos(self.k_f * self.Y)
        self.F_hat = np.fft.fft2(F)

    # ----------------------------------------------------------------------- #
    # diagnostics / helpers
    # ----------------------------------------------------------------------- #
    def streamfunction(self, q):
        """psi field from a vorticity field q."""
        return np.real(np.fft.ifft2(-np.fft.fft2(q) * self.K2_inv))

    def velocity(self, q):
        """(u, v) velocity fields from a vorticity field q."""
        psi_hat = -np.fft.fft2(q) * self.K2_inv
        u = np.real(np.fft.ifft2(-1j * self.ky * psi_hat))
        v = np.real(np.fft.ifft2(1j * self.kx * psi_hat))
        return u, v

    def energy(self, q):
        """Domain-mean kinetic energy 0.5 <u^2 + v^2>."""
        u, v = self.velocity(q)
        return 0.5 * np.mean(u ** 2 + v ** 2)

    def enstrophy(self, q):
        """Domain-mean enstrophy 0.5 <q^2>."""
        return 0.5 * np.mean(q ** 2)

    def random_ic(self, seed=None, amp=1.0, kpeak=6):
        """Smooth random vorticity field peaked near wavenumber kpeak."""
        rng = np.random.default_rng(seed)
        ph = rng.uniform(0, 2 * np.pi, (self.N, self.N))
        Kmag = np.sqrt(self.K2)
        spec = (Kmag ** 2) * np.exp(-(Kmag / kpeak) ** 2)
        q_hat = spec * np.exp(1j * ph)
        q = np.real(np.fft.ifft2(q_hat))
        q -= q.mean()
        return amp * q / np.std(q)

    # ----------------------------------------------------------------------- #
    # dynamics
    # ----------------------------------------------------------------------- #
    def _N_hat(self, q_hat):
        """Nonlinear + forcing + beta part of the spectral tendency (no damping)."""
        psi_hat = -q_hat * self.K2_inv

        # velocities and vorticity gradients in physical space
        u = np.real(np.fft.ifft2(-1j * self.ky * psi_hat))
        v = np.real(np.fft.ifft2(1j * self.kx * psi_hat))
        qx = np.real(np.fft.ifft2(1j * self.kx * q_hat))
        qy = np.real(np.fft.ifft2(1j * self.ky * q_hat))

        # advection J(psi, q) = u.grad(q), dealiased
        adv_hat = np.fft.fft2(u * qx + v * qy) * self.dealias

        beta_hat = self.beta * (1j * self.kx * psi_hat)        # beta * dpsi/dx

        return -adv_hat - beta_hat + self.F_hat

    def _rhs_hat(self, q_hat):
        """Full spectral tendency (nonlinear + linear damping)."""
        return self._N_hat(q_hat) + self.lin * q_hat

    def derivative(self, t, q):
        """
        Tendency dq/dt for a flattened real vorticity field (length N*N).
        Mirrors L96.derivative; usable with scipy.solve_ivp (stiff solver).
        """
        q_hat = np.fft.fft2(q.reshape(self.N, self.N))
        dq = np.real(np.fft.ifft2(self._rhs_hat(q_hat)))
        return dq.ravel()

    def step(self, q_hat, dt):
        """
        One integrating-factor RK4 step. The diagonal linear damping
        (drag + hyperviscosity) is integrated exactly, so stability is set by
        the advective CFL condition rather than the stiff diffusion term.
        """
        E2 = np.exp(0.5 * dt * self.lin)
        E = E2 * E2
        n1 = self._N_hat(q_hat)
        n2 = self._N_hat(E2 * (q_hat + 0.5 * dt * n1))
        n3 = self._N_hat(E2 * q_hat + 0.5 * dt * n2)
        n4 = self._N_hat(E * q_hat + dt * E2 * n3)
        return E * q_hat + (dt / 6.0) * (E * n1 + 2 * E2 * (n2 + n3) + n4)

    def integrate(self, q0, t, store_every=1):
        """
        Integrate from initial vorticity field q0 over the time array t
        (assumed uniformly spaced). Returns the trajectory.

        :param q0: initial vorticity, shape (N, N)
        :param t:  1-D array of output times (uniform spacing)
        :param store_every: keep every k-th requested step
        :return: (times, traj) with traj shape (n_stored, N, N)
        """
        dt = t[1] - t[0]
        q_hat = np.fft.fft2(q0)
        frames, times = [], []
        for i in range(len(t)):
            if i % store_every == 0:
                frames.append(np.real(np.fft.ifft2(q_hat)))
                times.append(t[i])
            q_hat = self.step(q_hat, dt)
        return np.array(times), np.array(frames)

    def settings(self):
        """Dict of model settings (stored as netCDF attributes)."""
        return dict(N=self.N, L=self.L, beta=self.beta, mu=self.mu,
                    nu=self.nu, p=self.p, A=self.A, k_f=self.k_f)

    def run(self, q0, tf, dt=0.01, spinup=0.0, store_every=1, path=None):
        """
        Integrate and return an xarray Dataset (and optionally save to netCDF),
        in the spirit of Experiment.get_true. Spinup is integrated but discarded.

        :param q0: initial vorticity field (N, N)
        :param tf: final time (after spinup)
        :param dt: time step
        :param spinup: time to discard before recording
        :param store_every: keep every k-th step in the output
        :param path: if given, save the Dataset to this netCDF file
        :return: xarray.Dataset with variable 'q' on dims (time, y, x)
        """
        import xarray as xr

        q_hat = np.fft.fft2(q0)
        for _ in range(int(round(spinup / dt))):          # discard spinup
            q_hat = self.step(q_hat, dt)

        q_start = np.real(np.fft.ifft2(q_hat))
        t = np.arange(0, tf + dt, dt)
        times, traj = self.integrate(q_start, t, store_every=store_every)

        x = np.arange(self.N) * (self.L / self.N)
        ds = xr.Dataset(
            {"q": (("time", "y", "x"), traj)},
            coords={"time": times, "y": x, "x": x},
            attrs=self.settings(),
        )
        if path is not None:
            _save_netcdf(ds, path)
        return ds


def _save_netcdf(ds, path):
    """
    Write a Dataset to netCDF, robust to Windows/OneDrive file locking.

    The default netCDF4/HDF5 backend uses file locks that synced folders
    (OneDrive, Dropbox) and open file viewers frequently reject with a
    PermissionError. We remove any stale file first and, if the HDF5 backend
    still fails, fall back to the lock-free scipy/NETCDF3 backend.
    """
    import os

    if os.path.exists(path):
        try:
            os.remove(path)                       # clear stale/locked file
        except OSError:
            pass
    try:
        ds.to_netcdf(path)                        # default (netCDF4/HDF5)
    except (PermissionError, OSError, RuntimeError):
        ds.to_netcdf(path, engine="scipy", format="NETCDF3_64BIT")