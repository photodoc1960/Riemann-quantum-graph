# Theorem 1: Neumann Rigidity on Cycle Graphs (Corrected)

*For insertion at the beginning of Section III.B of "Vertex Boundary Conditions Determine Spectral Correspondence in Quantum Graphs" (manuscript v3).*

*Version note (May 30, 2026): This file replaces an earlier version whose
spectrum formula and odd-$n$ "antiperiodic ladder" claim were incorrect.
The corrected result is mathematically cleaner: the Neumann cycle spectrum
is independent of $n$ for all $n$, the rigidity follows from the
reflectionlessness of the degree-2 Kirchhoff condition, and the
two-parameter TRS-broken ceiling has a precise algebraic explanation.
See the integration notes at the end of this file for the comparison to
the earlier version.*

---

## III.B Neumann ceiling

We first establish the algebraic origin of the Neumann correspondence
ceiling as a precise statement about the spectrum of $C_n$ under standard
Neumann boundary conditions.

**Theorem 1 (Neumann rigidity on cycle graphs).** *Let $C_n$ denote the
cycle graph on $n$ vertices with edge lengths $l_1, \ldots, l_n > 0$, and
let the Laplacian on $C_n$ be equipped with standard Neumann (Kirchhoff)
conditions at every degree-2 vertex. Then the spectrum of the resulting
self-adjoint operator depends on the edge lengths only through their sum
$L = \sum_{e=1}^{n} l_e$, and is*

$$
k_m = \frac{2\pi m}{L}, \qquad m \in \mathbb{Z}_{\geq 0},
$$

*with $k_0 = 0$ simple and each $k_m$ for $m \geq 1$ doubly degenerate.
The spectrum is independent of the partition of $L$ among individual edges
and of $n$, including its parity.*

**Proof.** At a degree-2 vertex the Kirchhoff conditions require continuity
of $\psi$ and continuity of $\psi'$. These conditions impose no constraint
on a $C^1$ function, so a degree-2 Kirchhoff vertex is spectrally removable
[10]. Hence $C_n$, viewed as a metric space carrying the operator
$-d^2/dx^2$ on each edge, is isometric to a single circle of circumference
$L = \sum_e l_e$. The Neumann graph Laplacian on $C_n$ is unitarily
equivalent to the Laplacian on this circle with periodic boundary
conditions, whose spectrum is $k_m = 2\pi m / L$ with the stated
degeneracies.

Equivalently in the bond picture: the Neumann scattering matrix at a
degree-2 vertex,

$$
\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix},
$$

is reflectionless ($S_{ii} = 0$). The bond-level scattering operator
$S \cdot U(k)$ therefore decouples into two independent channels —
forward-traversing and backward-traversing bond eigenfunctions — each of
which is a weighted $n$-cycle whose monodromy reduces to the scalar
$\prod_{v=1}^{n} e^{ik l_v} = e^{ikL}$. The eigenvalue condition
$e^{ikL} = 1$ in each channel gives $k_m = 2\pi m / L$. The double
degeneracy at $m \geq 1$ arises because forward and backward channels
yield the same eigenvalues; the simple eigenvalue at $m = 0$ corresponds
to the constant eigenfunction common to both channels. $\square$

**Corollary 1 (Neumann correspondence ceiling).** *Under Neumann
boundary conditions, the spectrum of $C_n$ is a single arithmetic
progression of spacing $2\pi / L$. The correspondence score $\mathcal{S}$
against any target sequence therefore depends on a single continuous
parameter $L > 0$, independent of $n$. Numerical maximization of
$\mathcal{S}$ over $L$ for the first $N = 100$ Riemann zeros under our
scoring (Sec. II.B) yields*

$$
\mathcal{S}_{\text{Neumann}}^{\max} \approx 0.092,
$$

*confirmed across $C_3$ through $C_{28}$ to a standard deviation
$< 0.003$.*

**Remark 1 (TRS-broken Neumann extension).** Introducing per-vertex
TRS-breaking phases via the similarity transform $D_v \sigma_x D_v^\dagger$
with $D_v = \mathrm{diag}(1, e^{i\theta_v})$ yields

$$
D_v \sigma_x D_v^\dagger = \begin{pmatrix} 0 & e^{-i\theta_v} \\ e^{i\theta_v} & 0 \end{pmatrix},
$$

which is *still reflectionless* (diagonal entries vanish): the conjugation
introduces direction-dependent transmission phases but no reflection
amplitude. The forward and backward channels remain decoupled and now
accumulate opposite monodromy phases $\pm \Phi$ with
$\Phi = \sum_v \theta_v$, so the spectrum splits into two interleaved
uniform ladders

$$
k_m^{\pm} = \frac{2\pi m \pm \Phi}{L}, \qquad m \in \mathbb{Z}_{\geq 0}.
$$

The TRS-broken Neumann spectrum is therefore a two-parameter rigid family
$(L, \Phi)$, with the achievable set identical for every $n$: $L$ ranges
over $(0, \infty)$ and $\Phi$ over the full circle of total phases. The
numerical maximum of $\mathcal{S}$ over $(L, \Phi)$ against the first
$N = 100$ Riemann zeros is $\mathcal{S}_{\text{Neumann+TRS}}^{\max}
\approx 0.72$. The numerical value $0.72$ is not derived from a closed
form, but its existence as a hard cap and its $n$-independence are
algebraically explicable: every TRS-broken Neumann cycle of any size $n$
is unitarily equivalent to one of this two-parameter family.

**Remark 2 (Reflection as the mechanism that breaks the ceiling).** The
ceiling in Corollary 1 and its TRS-broken extension in Remark 1 are both
consequences of a single structural property: $\sigma_x$ and
$D_v \sigma_x D_v^\dagger$ are both reflectionless at degree-2 vertices
($S_{ii} = 0$). The general $U(2)$ vertex scattering matrix

$$
U_v = e^{i\alpha} \begin{pmatrix} e^{i\beta}\cos\theta & -e^{-i\gamma}\sin\theta \\ e^{i\gamma}\sin\theta & e^{-i\beta}\cos\theta \end{pmatrix}
$$

has reflection magnitude $|U_v|_{ii} = |\cos\theta|$, and the Neumann case
corresponds to $\theta = \pi/2$ (zero reflection). The transition from
$\mathcal{S} \approx 0.72$ to $\mathcal{S} \approx 0.90$ reported in
Sec. III.C is therefore the consequence of allowing nonzero reflection at
the degree-2 vertices, not of unitary freedom in general. We elevate this
to a formal observation: it is reflection, not the additional phase
parameters, that lifts the spectrum out of the rigid two-parameter family.

The remainder of this section reports the empirical confirmation of
Theorem 1 and Corollary 1 across $C_3$ through $C_{28}$, and the
numerical determination of the TRS-broken ceiling at $\mathcal{S}_{\text{Neumann+TRS}}^{\max} \approx 0.72$.

---

## Notes for integration

1. **Replaces an earlier incorrect version.** The previous fragment
   stated the spectrum as $k_m^{\pm} = (2\pi m \pm \theta_n) / L$ with
   $\theta_n = 0$ for $n$ even and $\theta_n = \pi$ for $n$ odd, claiming
   an "antiperiodic ladder" for odd $n$. That claim is wrong: a degree-2
   Kirchhoff vertex is spectrally removable, so $C_n$ is isometric to a
   single circle of length $L$ regardless of $n$, and the spectrum is
   $k_m = 2\pi m / L$ in both parity cases. The error stemmed from
   applying $\det[I - \sigma_x^n e^{ikL}]$ in a $2 \times 2$ matrix space
   when the correct setting is the channel-decoupled scalar equation
   $\prod_v e^{ikl_v} = e^{ikL}$ in each of two independent channels.

2. **Stronger physical content.** The corrected version identifies
   reflectionlessness — not unitarity per se — as the structural property
   that constrains the Neumann (and TRS-broken Neumann) cycle to a
   low-dimensional rigid family. This connects directly to the new
   Remark 2: the U(2) → 0.90 transition is precisely the introduction
   of nonzero reflection at vertices, parameterized by $\theta \neq \pi/2$
   in the standard $U(2)$ form.

3. **Reference [10] in the proof** is Berkolaiko & Kuchment, *Introduction
   to Quantum Graphs* (AMS, 2013), where the spectral removability of
   degree-2 Kirchhoff vertices is standard textbook material.

4. **Remark 1 corrects the manuscript's overstatement.** The earlier
   version of Section III.B called the TRS-broken 0.72 ceiling "an
   analytically explicable consequence of the Neumann boundary condition
   structure," which overstated the rigor. The numerical value 0.72 is
   not derived analytically; what is derived analytically is the existence
   of a hard cap and its independence of $n$, because every TRS-broken
   Neumann cycle is unitarily equivalent to a member of the
   two-parameter $(L, \Phi)$ family. This is a stronger and more accurate
   claim than the original.

5. **Downstream implications for the manuscript.** The reflection
   observation in Remark 2 should be propagated into Section III.C and
   the Discussion. The "U(2) breakthrough" framing currently used in
   Section III.C and IV should become "reflection breakthrough" where the
   text supports it. The reflection-isolation control experiment proposed
   by the reviewer would directly test this: varying only $\theta_v$
   per vertex (other three U(2) parameters frozen at Neumann values) and
   observing whether most of the 0.72 → 0.90 gain returns.

6. **Acknowledgement.** The corrections in this version follow detailed
   mathematical feedback from a reviewer; the original spectrum formula
   error, the incorrect odd-$n$ antiperiodic claim, the overstated
   analytic-explicability sentence, and the reflection-as-mechanism
   observation are all due to that review. If the review constitutes
   intellectual contribution warranting co-authorship, the manuscript
   byline and acknowledgements must be updated accordingly before any
   further submission.
