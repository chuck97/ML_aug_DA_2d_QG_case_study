"""
visualization.py
----------------
Plotting and spectral-diagnostic helpers for the 2D QG model.

They operate on the xarray Dataset returned by ``QG2D.run`` (variable ``q``
on dimensions ``(time, y, x)``, with the square-domain length stored in
``ds.attrs["L"]``).  Thus they work on an in-memory run or on one reloaded
from netCDF; no ``QG2D`` instance is required.

Examples
--------
>>> from visualization import plot_snapshot, animate, plot_energy_spectra
>>> plot_snapshot(ds)
>>> plot_energy_spectra([ds_base, ds_low_drag], labels=["base", "low drag"])
"""

import numpy as np


# --------------------------------------------------------------------------- #
# Streamfunction reconstruction (spectral Laplacian inversion, from q alone)
# --------------------------------------------------------------------------- #

def _k2_inv(N, L):
    """Per-mode ``1 / K^2`` multiplier for inverting the Laplacian.

    The zero mode is set to zero because the streamfunction is only defined up
    to an arbitrary additive constant.
    """
    k = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    K2 = kx**2 + ky**2
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(K2 == 0, 0.0, 1.0 / K2)


def _streamfunction(q, K2_inv):
    """Return ``psi = inverse-Laplacian(q)`` for one vorticity field."""
    return np.real(np.fft.ifft2(-np.fft.fft2(q) * K2_inv))


def _domain_length(ds):
    """Return the (square) domain length stored in a QG run Dataset."""
    return float(ds.attrs.get("L", 2 * np.pi))


# --------------------------------------------------------------------------- #
# Spectral diagnostics
# --------------------------------------------------------------------------- #

def kinetic_energy_spectrum(ds, time_start=None, time_end=None, max_frames=None):
    """Compute a time-mean, isotropic kinetic-energy spectrum.

    Parameters
    ----------
    ds : xarray.Dataset
        A dataset produced by :meth:`QG2D.run`, containing ``q(time, y, x)``.
    time_start, time_end : float or None, optional
        Limits of the averaging window in the Dataset's ``time`` coordinate.
        ``None`` retains the first or last stored time, respectively.
    max_frames : int or None, optional
        Maximum number of equally spaced stored snapshots to use.  This is
        useful for very long runs; ``None`` uses all snapshots in the selected
        time window.

    Returns
    -------
    k : numpy.ndarray
        Positive, physical isotropic wavenumbers.  On the default domain
        ``L = 2*pi``, these are simply the integer Fourier modes 1, 2, ... .
    E_k : numpy.ndarray
        Time-mean kinetic energy in each radial wavenumber shell.  With the
        present FFT normalization, ``E_k.sum()`` equals the time-mean
        domain-mean kinetic energy (apart from round-off error).

    Notes
    -----
    For a vorticity field ``q = laplacian(psi)``, the kinetic energy carried
    by Fourier coefficient ``(kx, ky)`` is

    ``0.5 * |q_hat|^2 / (K^2 * (Nx * Ny)^2)``,

    where ``K^2 = kx^2 + ky^2``.  The function sums these modal contributions
    into unit-width isotropic shells.  The zero mode is excluded because it
    has no velocity and the inverse Laplacian is undefined there.
    """
    if "q" not in ds:
        raise KeyError("Dataset must contain a vorticity variable named 'q'.")

    q_da = ds["q"]
    if "time" not in q_da.dims:
        raise ValueError("The vorticity variable must have a 'time' dimension.")

    if time_start is not None or time_end is not None:
        q_da = q_da.sel(time=slice(time_start, time_end))

    ntime = q_da.sizes["time"]
    if ntime == 0:
        raise ValueError("The selected time window contains no stored frames.")

    if max_frames is not None:
        if not isinstance(max_frames, (int, np.integer)) or max_frames < 1:
            raise ValueError("max_frames must be a positive integer or None.")
        if ntime > max_frames:
            frame_index = np.linspace(0, ntime - 1, max_frames, dtype=int)
            q_da = q_da.isel(time=frame_index)

    q = np.asarray(q_da.values, dtype=float)
    if q.ndim != 3:
        raise ValueError(
            "Expected q to have shape (time, y, x); "
            f"received shape {q.shape}."
        )

    ny, nx = q.shape[-2:]
    if nx != ny:
        raise ValueError(
            "The current QG solver assumes a square grid; received "
            f"ny={ny}, nx={nx}."
        )

    L = _domain_length(ds)
    if L <= 0:
        raise ValueError("Dataset attribute 'L' must be positive.")

    # The axes are named (y, x) in the Dataset.  For an isotropic spectrum the
    # ordering is immaterial, but using FFT frequencies for both axes preserves
    # the exact pseudo-spectral grid used by QG2D.
    k_1d = 2 * np.pi * np.fft.fftfreq(nx, d=L / nx)
    kx, ky = np.meshgrid(k_1d, k_1d, indexing="ij")
    K2 = kx**2 + ky**2
    K = np.sqrt(K2)

    q_hat = np.fft.fft2(q, axes=(-2, -1))
    n_grid = nx * ny

    # Parseval-consistent modal kinetic energy, averaged across selected times.
    modal_energy = np.zeros_like(K2, dtype=float)
    nonzero = K2 > 0.0
    modal_energy[nonzero] = (
        0.5
        * np.mean(np.abs(q_hat[..., nonzero]) ** 2, axis=0)
        / (K2[nonzero] * n_grid**2)
    )

    # Radial shells have width delta_k = 2*pi/L.  Rounding assigns a mode at
    # radius sqrt(kx^2 + ky^2) to its nearest integer shell.
    delta_k = 2 * np.pi / L
    shell = np.rint(K / delta_k).astype(int)
    n_shells = int(shell.max()) + 1
    E_shell = np.bincount(
        shell.ravel(), weights=modal_energy.ravel(), minlength=n_shells
    )
    k_shell = np.arange(n_shells, dtype=float) * delta_k

    # Omit the zero mode.  It has no kinetic-energy contribution by definition.
    return k_shell[1:], E_shell[1:]


def plot_energy_spectra(
    datasets,
    labels=None,
    time_start=None,
    time_end=None,
    max_frames=None,
    ax=None,
    show_forcing=True,
    title="Time-mean kinetic-energy spectra",
    savepath=None,
):
    """Plot time-mean kinetic-energy spectra from several QG runs.

    Parameters
    ----------
    datasets : sequence of xarray.Dataset
        QG runs to compare.  Each must contain ``q(time, y, x)`` and should
        normally have the same domain size and grid resolution.
    labels : sequence of str or None, optional
        Legend labels.  Defaults to ``run 1``, ``run 2``, ... .
    time_start, time_end, max_frames : optional
        Passed unchanged to :func:`kinetic_energy_spectrum` for every run.
        Set ``time_start`` after the transient to compare statistically steady
        states rather than the initial spin-up.
    ax : matplotlib.axes.Axes or None, optional
        Existing axes to draw on.  A new figure and axes are created by
        default.
    show_forcing : bool, default True
        Draw a vertical dashed line at the forcing wavenumber when all runs
        use the same ``k_f``.  This makes the injection scale explicit without
        cluttering the legend for sensitivity studies that vary ``k_f``.
    title : str or None, optional
        Plot title.  Pass ``None`` to omit it.
    savepath : str or path-like or None, optional
        Write the resulting figure to this path when supplied.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    spectra : list of tuple
        ``[(k_1, E_1), (k_2, E_2), ...]`` in the same order as ``datasets``.

    Example
    -------
    >>> fig, ax, spectra = plot_energy_spectra(
    ...     [ds_base, ds_low_drag, ds_high_drag],
    ...     labels=["base", r"low $\\mu$", r"high $\\mu$"],
    ...     time_start=20.0,
    ... )
    """
    import matplotlib.pyplot as plt

    # Be permissive for a single Dataset while retaining the intended list API.
    if hasattr(datasets, "data_vars") and "q" in datasets:
        datasets = [datasets]
    else:
        datasets = list(datasets)

    if not datasets:
        raise ValueError("datasets must contain at least one QG run.")

    if labels is None:
        labels = [f"run {i + 1}" for i in range(len(datasets))]
    else:
        labels = list(labels)
        if len(labels) != len(datasets):
            raise ValueError("labels must have the same length as datasets.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
    else:
        fig = ax.figure

    spectra = []
    forcing_wavenumbers = []

    for ds, label in zip(datasets, labels):
        k, E_k = kinetic_energy_spectrum(
            ds,
            time_start=time_start,
            time_end=time_end,
            max_frames=max_frames,
        )
        spectra.append((k, E_k))

        # Log axes cannot display exact zeros.  Values below a machine-precision
        # floor relative to this spectrum's peak are numerical FFT noise rather
        # than resolved kinetic energy.
        finite_E = E_k[np.isfinite(E_k)]
        energy_floor = (
            np.finfo(float).eps * finite_E.max() if finite_E.size else 0.0
        )
        valid = (
            np.isfinite(k) & np.isfinite(E_k) & (k > 0.0)
            & (E_k > energy_floor)
        )
        ax.loglog(
            k[valid], E_k[valid], marker="o", markersize=3.0,
            linewidth=1.5, label=label,
        )

        k_f = ds.attrs.get("k_f")
        if k_f is not None:
            forcing_wavenumbers.append(float(k_f))

    # Only add one marker when the forcing scale is common to all curves.
    if show_forcing and len(forcing_wavenumbers) == len(datasets):
        k_f = forcing_wavenumbers[0]
        if np.allclose(forcing_wavenumbers, k_f):
            ax.axvline(
                k_f, color="0.25", linestyle="--", linewidth=1.0,
                label=rf"forcing $k_f={k_f:g}$",
            )

    ax.set_xlabel(r"isotropic wavenumber $k$")
    ax.set_ylabel(r"kinetic-energy spectrum $E(k)$")
    if title is not None:
        ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", alpha=0.45)
    ax.legend(frameon=True)

    if savepath is not None:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")

    return fig, ax, spectra


# --------------------------------------------------------------------------- #
# Existing visualisation helpers
# --------------------------------------------------------------------------- #

def plot_snapshot(ds, t_index=-1, savepath=None):
    """Plot vorticity and streamfunction at one stored model time.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset returned by :meth:`QG2D.run`.
    t_index : int, default -1
        Index along the time axis; ``-1`` selects the final stored frame.
    savepath : str or path-like or None
        Save the figure here when supplied.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.  Stored arrays are transposed for display so that
        x is horizontal and y is vertical.
    """
    import matplotlib.pyplot as plt

    q = ds["q"].isel(time=t_index).values
    L = _domain_length(ds)
    psi = _streamfunction(q, _k2_inv(q.shape[0], L))
    t = float(ds["time"].isel(time=t_index).values)

    extent = [0, L, 0, L]
    qmax = np.abs(q).max()

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    im0 = ax[0].imshow(
        q.T, cmap="RdBu_r", origin="lower", extent=extent,
        vmin=-qmax, vmax=qmax,
    )
    ax[0].set_title("vorticity q")
    fig.colorbar(im0, ax=ax[0], shrink=0.8)

    im1 = ax[1].imshow(psi.T, cmap="viridis", origin="lower", extent=extent)
    ax[1].set_title(r"streamfunction $\psi$")
    fig.colorbar(im1, ax=ax[1], shrink=0.8)

    for axes in ax:
        axes.set_xlabel("x")
        axes.set_ylabel("y")

    fig.suptitle("t = %.2f" % t)
    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=130, bbox_inches="tight")
        plt.close(fig)  # avoid duplicate auto-display in notebooks

    return fig


def animate(ds, savepath=None, fps=12, dpi=100):
    """Animate vorticity and streamfunction over a QG trajectory.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset returned by :meth:`QG2D.run`.
    savepath : str or path-like or None
        ``.mp4`` or ``.gif`` destination.  If ``None``, return the animation
        for inline display, for example ``HTML(anim.to_jshtml())``.
    fps : int, default 12
        Animation frame rate.
    dpi : int, default 100
        Resolution used when saving.

    Returns
    -------
    matplotlib.animation.FuncAnimation

    Notes
    -----
    Saving ``.mp4`` requires ffmpeg.  If it is unavailable, a ``.gif`` is
    written beside the requested path using Pillow.
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation

    q_all = ds["q"].values
    L = _domain_length(ds)
    K2_inv = _k2_inv(q_all.shape[-1], L)
    psi_all = np.array([_streamfunction(q, K2_inv) for q in q_all])
    times = ds["time"].values

    extent = [0, L, 0, L]
    qmax = np.abs(q_all).max()
    pmin, pmax = psi_all.min(), psi_all.max()

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    im0 = ax[0].imshow(
        q_all[0].T, cmap="RdBu_r", origin="lower", extent=extent,
        vmin=-qmax, vmax=qmax,
    )
    ax[0].set_title("vorticity q")
    fig.colorbar(im0, ax=ax[0], shrink=0.8)

    im1 = ax[1].imshow(
        psi_all[0].T, cmap="viridis", origin="lower", extent=extent,
        vmin=pmin, vmax=pmax,
    )
    ax[1].set_title(r"streamfunction $\psi$")
    fig.colorbar(im1, ax=ax[1], shrink=0.8)

    for axes in ax:
        axes.set_xlabel("x")
        axes.set_ylabel("y")

    title = fig.suptitle("t = %.2f" % times[0])
    fig.tight_layout()

    def update(frame):
        im0.set_data(q_all[frame].T)
        im1.set_data(psi_all[frame].T)
        title.set_text("t = %.2f" % times[frame])
        return im0, im1, title

    anim = animation.FuncAnimation(
        fig, update, frames=len(q_all), interval=1000.0 / fps, blit=False
    )

    if savepath is not None:
        if str(savepath).lower().endswith(".gif"):
            anim.save(savepath, writer=animation.PillowWriter(fps=fps), dpi=dpi)
        else:
            try:
                anim.save(savepath, writer=animation.FFMpegWriter(fps=fps), dpi=dpi)
            except (FileNotFoundError, RuntimeError, ValueError):
                alt = str(savepath).rsplit(".", 1)[0] + ".gif"
                anim.save(alt, writer=animation.PillowWriter(fps=fps), dpi=dpi)

    # Closing avoids a duplicate static frame in notebooks.  The animation
    # object still retains the figure for inline display or saved output.
    plt.close(fig)
    return anim