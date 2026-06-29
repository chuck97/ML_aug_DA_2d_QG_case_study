"""
visualization.py
----------------
Plotting helpers for the 2D QG model. They operate on the xarray Dataset
returned by ``QG2D.run`` (variable ``q`` on dims ``(time, y, x)``, with the
domain size stored in ``ds.attrs["L"]``), so they work on an in-memory run or
on one reloaded from netCDF -- no QG2D instance required:

    import xarray as xr
    from visualization import plot_snapshot, animate
    ds = xr.open_dataset("qg_default.nc")
    plot_snapshot(ds, savepath="snap.png")
    animate(ds, savepath="movie.mp4")
"""

import numpy as np


# --------------------------------------------------------------------------- #
# streamfunction reconstruction (spectral Laplacian inversion, from q alone)
# --------------------------------------------------------------------------- #
def _k2_inv(N, L):
    """Per-mode 1/K^2 multiplier for inverting the Laplacian; K=0 mode set to 0."""
    k = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    K2 = kx ** 2 + ky ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(K2 == 0, 0.0, 1.0 / K2)


def _streamfunction(q, K2_inv):
    """psi = inverse-Laplacian(q) for a single 2D vorticity field."""
    return np.real(np.fft.ifft2(-np.fft.fft2(q) * K2_inv))


def _domain_length(ds):
    return float(ds.attrs.get("L", 2 * np.pi))


# --------------------------------------------------------------------------- #
# public helpers
# --------------------------------------------------------------------------- #
def plot_snapshot(ds, t_index=-1, savepath=None):
    """
    Plot the vorticity and streamfunction fields at a single stored time.

    :param ds: Dataset returned by QG2D.run (variable 'q', dim 'time')
    :param t_index: index along the time axis (default -1, the last frame)
    :param savepath: if given, save the figure there; otherwise just draw it
    :return: the matplotlib Figure

    The stored arrays are indexed [x, y], so fields are transposed for display
    (x horizontal, y vertical).
    """
    import matplotlib.pyplot as plt

    q = ds["q"].isel(time=t_index).values
    L = _domain_length(ds)
    psi = _streamfunction(q, _k2_inv(q.shape[0], L))
    t = float(ds["time"].isel(time=t_index).values)
    extent = [0, L, 0, L]
    qmax = np.abs(q).max()

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    im0 = ax[0].imshow(q.T, cmap="RdBu_r", origin="lower",
                       extent=extent, vmin=-qmax, vmax=qmax)
    ax[0].set_title("vorticity  q")
    fig.colorbar(im0, ax=ax[0], shrink=0.8)
    im1 = ax[1].imshow(psi.T, cmap="viridis", origin="lower", extent=extent)
    ax[1].set_title(r"streamfunction  $\psi$")
    fig.colorbar(im1, ax=ax[1], shrink=0.8)
    for a in ax:
        a.set_xlabel("x")
        a.set_ylabel("y")
    fig.suptitle("t = %.2f" % t)
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=130, bbox_inches="tight")
    plt.close(fig)          # avoid the duplicate auto-display in notebooks
    return fig


def animate(ds, savepath=None, fps=12, dpi=100):
    """
    Animate the vorticity and streamfunction fields over the trajectory.

    :param ds: Dataset returned by QG2D.run
    :param savepath: '<name>.mp4' or '<name>.gif' to save. If None, the
                     FuncAnimation is returned for inline display, e.g.
                     `from IPython.display import HTML; HTML(anim.to_jshtml())`.
    :param fps: frames per second
    :param dpi: resolution when saving
    :return: matplotlib FuncAnimation

    Saving '.mp4' needs ffmpeg; if it is not found the movie is written as a
    '.gif' (via Pillow) next to the requested path instead.
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
    im0 = ax[0].imshow(q_all[0].T, cmap="RdBu_r", origin="lower",
                       extent=extent, vmin=-qmax, vmax=qmax)
    ax[0].set_title("vorticity  q")
    fig.colorbar(im0, ax=ax[0], shrink=0.8)
    im1 = ax[1].imshow(psi_all[0].T, cmap="viridis", origin="lower",
                       extent=extent, vmin=pmin, vmax=pmax)
    ax[1].set_title(r"streamfunction  $\psi$")
    fig.colorbar(im1, ax=ax[1], shrink=0.8)
    for a in ax:
        a.set_xlabel("x")
        a.set_ylabel("y")
    title = fig.suptitle("t = %.2f" % times[0])
    fig.tight_layout()

    def update(frame):
        im0.set_data(q_all[frame].T)
        im1.set_data(psi_all[frame].T)
        title.set_text("t = %.2f" % times[frame])
        return im0, im1, title

    anim = animation.FuncAnimation(fig, update, frames=len(q_all),
                                   interval=1000.0 / fps, blit=False)

    if savepath is not None:
        if savepath.lower().endswith(".gif"):
            anim.save(savepath, writer=animation.PillowWriter(fps=fps), dpi=dpi)
        else:
            try:
                anim.save(savepath, writer=animation.FFMpegWriter(fps=fps), dpi=dpi)
            except (FileNotFoundError, RuntimeError, ValueError):
                alt = savepath.rsplit(".", 1)[0] + ".gif"
                anim.save(alt, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    # Close the figure either way: when saving it is no longer needed, and when
    # returning the animation this stops the static first frame from being
    # auto-displayed (it would otherwise reappear at the end of later cells).
    plt.close(fig)
    return anim