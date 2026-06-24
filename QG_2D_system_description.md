# A 2D Quasi-Geostrophic Model: System Description and Numerical Solver

## 1. Overview

The model integrates a single-layer, doubly-periodic quasi-geostrophic (QG)
vorticity equation on a square domain. It is the 2D analogue of the Lorenz-96
testbed: a spatially extended, forced–dissipative, chaotic system whose periodic
geometry makes it a natural target for convolutional emulators and
data-assimilation experiments. The implementation (`QG_2D.py`) is pseudospectral
and the prognostic state is the relative vorticity field on an $N \times N$ grid.

## 2. Prognostic variables

The single prognostic variable is the relative vorticity

$$q(x,y,t) = \nabla^2 \psi,$$

where $\psi(x,y,t)$ is the geostrophic streamfunction. All other fields are
*diagnostic*, recovered from $q$ at each step:

$$\text{streamfunction:}\quad \psi = \nabla^{-2} q,$$

$$\text{velocity:}\quad \mathbf{u} = (u,v) = \left(-\frac{\partial \psi}{\partial y},\; \frac{\partial \psi}{\partial x}\right).$$

The flow is non-divergent by construction ($\nabla \cdot \mathbf{u} = 0$).

## 3. Governing equation

The vorticity evolves on a $\beta$-plane under forcing and dissipation:

$$\frac{\partial q}{\partial t} + J(\psi, q) + \beta\,\frac{\partial \psi}{\partial x}
  = F - \mu\,q - \nu\,(-\nabla^2)^p\,q$$

with the Jacobian (advection) term

$$J(\psi, q) = \frac{\partial \psi}{\partial x}\frac{\partial q}{\partial y}
            - \frac{\partial \psi}{\partial y}\frac{\partial q}{\partial x}
            = \mathbf{u} \cdot \nabla q.$$

The terms on the right-hand side are, in order: external forcing $F$, linear
(Ekman) drag $-\mu q$ that removes energy at large scales, and a hyperviscous
sink $-\nu(-\nabla^2)^p q$ that removes enstrophy near the grid scale. The term
$\beta\,\partial \psi / \partial x = \beta v$ represents advection of planetary
vorticity and supports Rossby waves; with $\beta > 0$ the flow organises into
zonal (banded) structures.

**Default forcing.** Unless overridden, $F$ is a stationary Kolmogorov forcing,

$$F(x,y) = A\,\cos(k_f\, y),$$

which injects energy at a single meridional wavenumber $k_f$ and drives a shear
flow that is unstable to a sustained chaotic eddy field.

## 4. Parameters

| Symbol | Code | Meaning | Default |
|--------|------|---------|---------|
| $N$ | `N` | grid points per dimension ($N \times N$ state) | 64 |
| $L$ | `L` | domain side length | $2\pi$ |
| $\beta$ | `beta` | planetary vorticity gradient | 0 |
| $\mu$ | `mu` | linear (Ekman) drag coefficient | 0.1 |
| $\nu$ | `nu` | hyperviscosity coefficient | $10^{-3}$ |
| $p$ | `p` | hyperviscosity order $(-\nabla^2)^p$ | 2 |
| $A$ | `A` | forcing amplitude | 4.0 |
| $k_f$ | `k_f` | forcing wavenumber | 4 |
| $\Delta t$ | `dt` | time step | 0.01 |

The state dimension is $N^2$ (e.g. 4096 at $N = 64$). Physical roles: $A, k_f$
set the energy injection rate and scale; $\mu$ controls the large-scale energy
balance; $\nu, p$ control small-scale dissipation; $\beta$ sets the degree of
anisotropy / zonation; $N$ sets resolution and cost.

## 5. Boundary conditions

The domain is the square $[0,L) \times [0,L)$ with **doubly-periodic** boundary
conditions in both $x$ and $y$:

$$q(x+L,\,y,\,t) = q(x,\,y+L,\,t) = q(x,\,y,\,t),$$

and likewise for $\psi$ and $\mathbf{u}$. Periodicity is exact: it is enforced
implicitly by the Fourier (FFT) representation, so no explicit wall, inflow or
outflow conditions are required. The spatial mean of $q$ is conserved by the
advection and is taken to be zero; correspondingly the domain-mean streamfunction
(the $\mathbf{k} = \mathbf{0}$ Fourier mode) is undefined under the Laplacian
inversion and is set to zero.

## 6. Numerical solver

### 6.1 Collocation grid

The domain $[0,L) \times [0,L)$ is discretised on a uniform $N \times N$ grid,

$$x_j = j\,\Delta,\quad y_l = l\,\Delta,\qquad \Delta = \frac{L}{N},\qquad j,l = 0,1,\dots,N-1,$$

and the prognostic field is stored as the grid values $q_{jl}(t) = q(x_j, y_l, t)$.

### 6.2 Fourier basis and series expansion

On a periodic square the natural orthogonal basis is the set of complex
exponentials

$$\phi_{mn}(x,y) = e^{\,i(k_m x + k_n y)},\qquad k_m = \frac{2\pi}{L}\,m,\quad k_n = \frac{2\pi}{L}\,n,$$

where the integers $(m,n)$ index the modes. Each $\phi_{mn}$ has period $L$ in
both directions, so any field built from them automatically satisfies the
doubly-periodic boundary conditions of Section 5. On $N$ collocation points only
$N$ distinct wavenumbers are resolvable; following the FFT convention used in the
code the indices run over the symmetric set

$$m,n \in \left\{ -\tfrac{N}{2},\,\dots,\,-1,\,0,\,1,\,\dots,\,\tfrac{N}{2}-1 \right\},$$

so the largest resolved wavenumber is the Nyquist value
$k_{\max} = \tfrac{2\pi}{L}\tfrac{N}{2} = \pi/\Delta$.

The prognostic vorticity is represented by the corresponding truncated Fourier
series,

$$q(x,y,t) = \sum_{m,n=-N/2}^{N/2-1} \widehat{q}_{mn}(t)\; e^{\,i(k_m x + k_n y)},$$

and the time-dependent complex coefficients $\widehat{q}_{mn}(t)$ are the unknowns
actually advanced in spectral space. They are obtained from the grid values by
the forward 2D discrete Fourier transform (`fft2`) and the grid values are
recovered by the inverse transform (`ifft2`):

$$\widehat{q}_{mn} = \sum_{j,l=0}^{N-1} q_{jl}\, e^{-\,i(k_m x_j + k_n y_l)},$$

$$q_{jl} = \frac{1}{N^2}\sum_{m,n} \widehat{q}_{mn}\, e^{+\,i(k_m x_j + k_n y_l)}.$$

Because $k_m x_j = \tfrac{2\pi}{L}m \cdot j\Delta = 2\pi m j / N$, these reduce to
the standard DFT kernels $e^{\pm 2\pi i\, mj/N}$, which is what the FFT computes.
As $q$ is real its coefficients are Hermitian,
$\widehat{q}_{-m,-n} = \widehat{q}_{mn}^{*}$; the implementation carries the full
complex array and takes the real part after each inverse transform, enforcing
this automatically.

### 6.3 Spectral differentiation and inversion of the Laplacian

Every basis function is an eigenfunction of differentiation,

$$\frac{\partial}{\partial x}\phi_{mn} = i k_m\,\phi_{mn},\quad
  \frac{\partial}{\partial y}\phi_{mn} = i k_n\,\phi_{mn},\quad
  \nabla^2\,\phi_{mn} = -K_{mn}^2\,\phi_{mn},$$

with $K_{mn}^2 = k_m^2 + k_n^2$. Differential operators therefore become
*algebraic, mode-by-mode multipliers* acting on the coefficients:

$$\widehat{\partial_x q}_{mn} = i k_m\,\widehat{q}_{mn},\quad
  \widehat{\partial_y q}_{mn} = i k_n\,\widehat{q}_{mn},\quad
  \widehat{\nabla^2 q}_{mn} = -K_{mn}^2\,\widehat{q}_{mn}.$$

Inverting the Laplacian to recover the streamfunction from $q = \nabla^2 \psi$ is
then a per-mode division,

$$-K_{mn}^2\,\widehat{\psi}_{mn} = \widehat{q}_{mn}
  \quad\Longrightarrow\quad
  \widehat{\psi}_{mn} = -\frac{\widehat{q}_{mn}}{K_{mn}^2}\quad (K_{mn} \neq 0).$$

The mean mode $m = n = 0$ has $K_{00} = 0$: $\widehat{\psi}_{00}$ is undefined
(adding a constant to $\psi$ does not change the velocities) and is fixed to zero
by gauge choice, while consistency requires zero domain-mean vorticity,
$\widehat{q}_{00} = 0$. The velocities and vorticity gradients follow from the
same multipliers,

$$\widehat{u}_{mn} = -i k_n\,\widehat{\psi}_{mn},\quad
  \widehat{v}_{mn} = i k_m\,\widehat{\psi}_{mn},\quad
  \widehat{q_x}_{mn} = i k_m\,\widehat{q}_{mn},\quad
  \widehat{q_y}_{mn} = i k_n\,\widehat{q}_{mn}.$$

### 6.4 Nonlinear term and dealiasing (the pseudospectral step)

The advection $J(\psi, q) = u\,q_x + v\,q_y$ is a product of fields. Products are
cheap and local in physical space but correspond to a convolution
$\widehat{(fg)}_{\mathbf{k}} = \sum_{\mathbf{k}'} \widehat{f}_{\mathbf{k}'}\widehat{g}_{\mathbf{k}-\mathbf{k}'}$
in spectral space. The *pseudospectral* strategy therefore evaluates the product
where it is cheap: the factors $u, v, q_x, q_y$ are formed by the spectral
multipliers above and transformed to the grid, multiplied pointwise there, and
the result transformed back, at a cost of $O(N^2 \log N)$ rather than the
$O(N^4)$ of a direct convolution.

Multiplying two fields whose wavenumbers reach $k_{\max}$ produces content up to
$2k_{\max}$, beyond the resolved range; on the finite grid these high wavenumbers
are misrepresented as spurious low ones (*aliasing*). They are removed with the
**2/3 rule**: all coefficients with $|k_m|$ or $|k_n|$ exceeding
$\tfrac{2}{3}k_{\max}$ are set to zero, so the aliased contributions fall in the
discarded band. In the code this is a boolean mask applied to the transform of
the nonlinear product.

### 6.5 Semi-discrete form

Writing the spectral tendency as a nonlinear part $\mathcal{N}$ and a diagonal
linear part $\mathcal{L}$,

$$\frac{\partial \widehat{q}}{\partial t} = \mathcal{N}(\widehat{q}) + \mathcal{L}\,\widehat{q},$$

$$\mathcal{N}(\widehat{q}) = -\,\widehat{J(\psi,q)} - \beta\,i k_m\,\widehat{\psi} + \widehat{F},$$

$$\mathcal{L} = -\,\mu - \nu\,(K^2)^p.$$

### 6.6 Time integration (integrating-factor RK4)

The dissipative operator $\mathcal{L}$ is stiff: at $N = 64$ the fastest-damped
mode gives $|\mathcal{L}|\Delta t \gg 2.78$, so an explicit RK4 applied to the
*full* right-hand side is unstable. Because $\mathcal{L}$ is diagonal and
real-negative, it is integrated *exactly* via an integrating factor, while
$\mathcal{N}$ is advanced with RK4. With $E = e^{\mathcal{L}\Delta t}$ and
$E_2 = e^{\mathcal{L}\Delta t/2}$, one step from $\widehat{q}_n$ is

$$
\begin{aligned}
n_1 &= \mathcal{N}(\widehat{q}_n), \\
n_2 &= \mathcal{N}\!\big(E_2(\widehat{q}_n + \tfrac{1}{2}\Delta t\,n_1)\big), \\
n_3 &= \mathcal{N}\!\big(E_2\widehat{q}_n + \tfrac{1}{2}\Delta t\,n_2\big), \\
n_4 &= \mathcal{N}\!\big(E\,\widehat{q}_n + \Delta t\,E_2\,n_3\big), \\
\widehat{q}_{n+1} &= E\,\widehat{q}_n + \tfrac{\Delta t}{6}\big(E\,n_1 + 2E_2(n_2+n_3) + n_4\big).
\end{aligned}
$$

Since $E, E_2 \in (0,1]$, the damping is unconditionally stable; the time step is
then limited only by the advective CFL condition. The scheme is fourth-order
accurate in $\Delta t$ and was verified against a stiff implicit reference solver
(relative error $\sim 3 \times 10^{-5}$ at $\Delta t = 10^{-3}$).

### 6.7 Algorithm summary

1. Set initial vorticity $q_0$ (e.g. a smooth random field) and transform to $\widehat{q}$.
2. (Optional) integrate a spin-up interval and discard it.
3. For each step: invert the Laplacian for $\widehat{\psi}$, build velocities and
   gradients, evaluate the dealiased nonlinear term $\mathcal{N}$, and advance
   with the integrating-factor RK4 above.
4. Store $q = \mathcal{F}^{-1}\widehat{q}$ on the requested cadence; diagnose
   $\psi$, $\mathbf{u}$, energy $E = \tfrac{1}{2}\langle|\mathbf{u}|^2\rangle$, and
   enstrophy $Z = \tfrac{1}{2}\langle q^2\rangle$ as needed.

### 6.8 Conserved / diagnostic quantities

In the unforced, inviscid limit ($A = \mu = \nu = 0$) the system conserves domain
energy and enstrophy; with forcing and dissipation these reach a fluctuating
statistical balance, which serves as the working chaotic attractor for the
experiments.
