# The Stochastic Ensemble Kalman Filter

This note describes the stochastic (perturbed-observations) Ensemble Kalman
Filter implemented in `EnKF.py` and driven by `Experiment.py`. The notation is
chosen to match the code one-to-one, so each equation maps directly onto a line
in `EnKF.analysis`.

## Symbols and code names

| symbol | meaning | code |
|--------|---------|------|
| $n$ | state dimension ($=N^2$, the flattened vorticity field) | `n` |
| $m$ | ensemble size | `m` / `nens` |
| $p$ | number of observations | `p` |
| $\mathbf{X}^f$ | forecast ensemble, $n\times m$ (columns are members) | `Xf` |
| $\overline{\mathbf{x}}^f$ | forecast ensemble mean, $n\times 1$ | `xbar` |
| $\mathbf{X}'$ | forecast perturbations $\mathbf{X}^f-\overline{\mathbf{x}}^f$, $n\times m$ | `Xp` |
| $\mathbf{H}$ | observation operator (selects observed points) | indexing by `obs_idx` |
| $\mathbf{H}\mathbf{X}'$ | perturbations at observed points, $p\times m$ | `HXp` |
| $\mathbf{P}^f\mathbf{H}^\top$ | state–obs covariance, $n\times p$ | `PfHt` |
| $\mathbf{H}\mathbf{P}^f\mathbf{H}^\top$ | obs–obs covariance, $p\times p$ | `HPfHt` |
| $r$ | observation error standard deviation | `r` / `sigma_obs` |
| $\mathbf{R}=r^2\mathbf{I}_p$ | observation error covariance | `r**2 * np.eye(p)` |
| $\mathbf{S}$ | innovation covariance $\mathbf{H}\mathbf{P}^f\mathbf{H}^\top+\mathbf{R}$ | `S` |
| $\gamma$ | multiplicative inflation factor | `gamma` |
| $\boldsymbol{\rho}_{xy},\ \boldsymbol{\rho}_{yy}$ | localization tapers, $n\times p$ and $p\times p$ | `_rho_xy`, `_rho_yy` |
| $\mathbf{D}$ | perturbed observations, $p\times m$ | `D` |
| $\mathbf{X}^a$ | analysis ensemble, $n\times m$ | `Xa` |

## 1. Setup

The state is the vorticity field on the $N\times N$ grid, flattened to a vector
of length $n=N^2$. Our knowledge of it is represented by an **ensemble** of $m$
members, stored as the columns of

$$\mathbf{X}^f = \big[\,\mathbf{x}^f_1,\ \mathbf{x}^f_2,\ \dots,\ \mathbf{x}^f_m\,\big] \in \mathbb{R}^{n\times m}.$$

The ensemble mean is the best estimate and the ensemble spread encodes the
uncertainty:

$$\overline{\mathbf{x}}^f = \frac{1}{m}\sum_{j=1}^{m}\mathbf{x}^f_j,
\qquad
\mathbf{X}' = \mathbf{X}^f - \overline{\mathbf{x}}^f ,$$

with the sample forecast covariance

$$\mathbf{P}^f = \frac{1}{m-1}\,\mathbf{X}'\mathbf{X}'^{\top} \in \mathbb{R}^{n\times n}.$$

At an observation time we have $p$ observations $\mathbf{y}\in\mathbb{R}^p$ with
error covariance $\mathbf{R}=r^2\mathbf{I}_p$. The **observation operator**
$\mathbf{H}$ maps a state to observation space; here it simply selects the
observed grid points, so in code $\mathbf{H}\mathbf{x} = $ `x[obs_idx]`.

## 2. The Kalman update

The Kalman filter combines the forecast and the observations into the analysis
by the **Kalman gain**

$$\mathbf{K} = \mathbf{P}^f\mathbf{H}^\top\big(\mathbf{H}\mathbf{P}^f\mathbf{H}^\top + \mathbf{R}\big)^{-1}
= \mathbf{P}^f\mathbf{H}^\top\,\mathbf{S}^{-1},
\qquad
\mathbf{S} = \mathbf{H}\mathbf{P}^f\mathbf{H}^\top + \mathbf{R}.$$

$\mathbf{K}$ weights the **innovation** (observation minus forecast) by the
relative confidence in each: where the forecast is uncertain and observations
are accurate, the gain is large.

## 3. Ensemble form (avoiding the $n\times n$ covariance)

With $n=N^2$ (e.g. $4096$ at $N=64$) the matrix $\mathbf{P}^f$ is far too large
to form. The EnKF never builds it: the gain only needs the two products

$$\mathbf{P}^f\mathbf{H}^\top = \frac{1}{m-1}\,\mathbf{X}'\,(\mathbf{H}\mathbf{X}')^{\top}
\quad (n\times p),
\qquad
\mathbf{H}\mathbf{P}^f\mathbf{H}^\top = \frac{1}{m-1}\,(\mathbf{H}\mathbf{X}')(\mathbf{H}\mathbf{X}')^{\top}
\quad (p\times p),$$

which are cheap because they run through the small ensemble dimension $m$. In
code, with `HXp = Xp[idx]`:

```
PfHt  = (Xp @ HXp.T) / (m - 1)      #  P^f H^T   (n, p)
HPfHt = (HXp @ HXp.T) / (m - 1)     #  H P^f H^T (p, p)
```

## 4. Stochastic (perturbed-observations) update

Updating every member with the *same* observation vector would collapse the
analysis spread and underestimate the true analysis uncertainty. The
**stochastic** EnKF fixes this by giving each member its own noisy copy of the
observations,

$$\mathbf{d}_j = \mathbf{y} + \boldsymbol{\varepsilon}_j,
\qquad \boldsymbol{\varepsilon}_j \sim \mathcal{N}(\mathbf{0}, \mathbf{R}),
\qquad j = 1,\dots,m,$$

collected as the columns of $\mathbf{D}\in\mathbb{R}^{p\times m}$. Each member is
then updated with its own innovation:

$$\mathbf{x}^a_j = \mathbf{x}^f_j + \mathbf{K}\big(\mathbf{d}_j - \mathbf{H}\mathbf{x}^f_j\big),
\qquad
\mathbf{X}^a = \mathbf{X}^f + \mathbf{K}\big(\mathbf{D} - \mathbf{H}\mathbf{X}^f\big).$$

Rather than forming $\mathbf{K}=\mathbf{P}^f\mathbf{H}^\top\mathbf{S}^{-1}$
explicitly, the code solves the linear system (more stable and never inverts
$\mathbf{S}$):

```
D     = y[:, None] + np.random.normal(0.0, r, size=(p, m))   #  D
innov = D - Xf[idx]                                          #  D - H X^f
Xa    = Xf + PfHt @ np.linalg.solve(S, innov)               #  X^a
```

so that $\mathbf{X}^a = \mathbf{X}^f + \mathbf{P}^f\mathbf{H}^\top\,\mathbf{S}^{-1}(\mathbf{D}-\mathbf{H}\mathbf{X}^f)$.

## 5. Inflation

A small ensemble systematically **under-estimates** its own spread (sampling
error plus unrepresented model error), which makes the filter trust the forecast
too much and can lead to *filter divergence* — the analysis ignores the
observations and drifts away from the truth. **Multiplicative inflation**
counteracts this by scaling the forecast perturbations about their mean before
the update,

$$\mathbf{X}^f \leftarrow \overline{\mathbf{x}}^f + \sqrt{\gamma}\,\big(\mathbf{X}^f - \overline{\mathbf{x}}^f\big),$$

which multiplies the forecast covariance by $\gamma$. The factor $\gamma\ge 1$
is `gamma` in the code; $\gamma=1$ means no inflation. Too little inflation
under-spreads and diverges; too much re-injects noise into the analysis, so
$\gamma$ is tuned (typically a few percent above 1).

```
xbar = Xf.mean(axis=1, keepdims=True)
if self.gamma != 1.0:
    Xf = xbar + np.sqrt(self.gamma) * (Xf - xbar)
```

## 6. Localization

With few members, sample covariances between **distant** points are dominated by
noise: spurious long-range correlations let an observation wrongly update remote
parts of the state. **Localization** suppresses these by tapering the covariances
with distance through a Schur (element-wise) product,

$$\mathbf{P}^f\mathbf{H}^\top \leftarrow \boldsymbol{\rho}_{xy}\circ \mathbf{P}^f\mathbf{H}^\top,
\qquad
\mathbf{H}\mathbf{P}^f\mathbf{H}^\top \leftarrow \boldsymbol{\rho}_{yy}\circ \mathbf{H}\mathbf{P}^f\mathbf{H}^\top,$$

where $\boldsymbol{\rho}_{xy}$ ($n\times p$) tapers state–observation pairs and
$\boldsymbol{\rho}_{yy}$ ($p\times p$) tapers observation–observation pairs.
Besides removing noise, localization raises the effective rank of the update,
letting an $m$-member ensemble correct many more than $m$ degrees of freedom.

The taper is the **Gaspari–Cohn** function $g$, a smooth, compactly supported
approximation to a Gaussian that reaches exactly zero at twice the length scale
$c$ (the code's `loc`). With the **periodic (torus) distance**
$d(\mathbf{s}_i,\mathbf{s}_k)$ between grid locations,

$$\big(\boldsymbol{\rho}_{xy}\big)_{ik} = g\!\big(d(\mathbf{s}_i,\mathbf{s}_k)\,;\,c\big),$$

$$
g(d; c) =
\begin{cases}
-\tfrac{1}{4}\alpha^5 + \tfrac{1}{2}\alpha^4 + \tfrac{5}{8}\alpha^3 - \tfrac{5}{3}\alpha^2 + 1, & 0 \le \alpha \le 1,\\[4pt]
\tfrac{1}{12}\alpha^5 - \tfrac{1}{2}\alpha^4 + \tfrac{5}{8}\alpha^3 + \tfrac{5}{3}\alpha^2 - 5\alpha + 4 - \tfrac{2}{3}\alpha^{-1}, & 1 < \alpha \le 2,\\[4pt]
0, & \alpha > 2,
\end{cases}
\qquad \alpha = d/c .
$$

Because the observation network is fixed in time, $\boldsymbol{\rho}_{xy}$ and
$\boldsymbol{\rho}_{yy}$ are computed **once** in `set_observation_network` and
reused every cycle:

```
obs_c   = coords[obs_idx]
_rho_xy = gaspari_cohn(_periodic_dist(coords, obs_c), loc)   # (n, p)
_rho_yy = gaspari_cohn(_periodic_dist(obs_c, obs_c), loc)    # (p, p)
...
PfHt  = PfHt  * _rho_xy
HPfHt = HPfHt * _rho_yy
```

Setting `loc = None` disables localization ($\boldsymbol{\rho}\equiv 1$).

## 7. The analysis step, end to end

Putting the pieces together, one call to `EnKF.analysis(Xf, y, r)` performs:

1. **inflate** the forecast ensemble: $\mathbf{X}^f \leftarrow \overline{\mathbf{x}}^f + \sqrt{\gamma}(\mathbf{X}^f-\overline{\mathbf{x}}^f)$;
2. form perturbations $\mathbf{X}' = \mathbf{X}^f-\overline{\mathbf{x}}^f$ and $\mathbf{H}\mathbf{X}' = $ `Xp[idx]`;
3. build $\mathbf{P}^f\mathbf{H}^\top$ and $\mathbf{H}\mathbf{P}^f\mathbf{H}^\top$ from the ensemble;
4. **localize** both by the Schur products with $\boldsymbol{\rho}_{xy},\boldsymbol{\rho}_{yy}$;
5. form $\mathbf{S} = \mathbf{H}\mathbf{P}^f\mathbf{H}^\top + r^2\mathbf{I}_p$;
6. draw perturbed observations $\mathbf{D}$ and solve $\mathbf{X}^a = \mathbf{X}^f + \mathbf{P}^f\mathbf{H}^\top\mathbf{S}^{-1}(\mathbf{D}-\mathbf{H}\mathbf{X}^f)$.

## 8. The assimilation cycle

`Experiment.assimilate` repeats, for each observation time $i=1,2,\dots$:

$$
\underbrace{\mathbf{x}^f_{j,i} = \mathcal{M}_{\Delta t_\text{obs}}\!\big(\mathbf{x}^a_{j,i-1}\big)}_{\textbf{forecast each member with QG2D}}
\qquad\longrightarrow\qquad
\underbrace{\mathbf{X}^a_i = \texttt{analysis}\big(\mathbf{X}^f_i,\ \mathbf{y}_i,\ r\big)}_{\textbf{EnKF update}},
$$

where $\mathcal{M}_{\Delta t_\text{obs}}$ integrates the QG model over one
observation interval (`obs_every` steps). The observation interval and error
$(\Delta t_\text{obs}, r)$ come from the observation dataset produced in
`02_sample_observations.ipynb`. If instead an ML analysis operator is supplied
(`method="ml"`), it replaces step 6 by predicting the analysis mean directly,
and the ensemble is recentred on that mean.

## References

- G. Evensen (1994); G. Burgers, P. J. van Leeuwen, G. Evensen (1998) — stochastic EnKF.
- G. Gaspari, S. E. Cohn (1999) — the compactly supported correlation function used for localization.
- P. L. Houtekamer, H. L. Mitchell (1998, 2001) — covariance localization and inflation in the EnKF.
