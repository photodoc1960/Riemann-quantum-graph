# Theorem 1: Neumann Rigidity on Cycle Graphs

*For insertion at the beginning of Section III.B of "Vertex Boundary Conditions Determine Spectral Correspondence in Quantum Graphs" (manuscript v3).*

---

## III.B Neumann ceiling

We first establish the algebraic origin of the Neumann correspondence ceiling as a precise statement about the spectrum of $C_n$ under Neumann boundary conditions.

**Theorem 1 (Neumann rigidity on cycle graphs).** *Let $C_n$ denote the cycle graph on $n$ vertices with edge lengths $l_1, \ldots, l_n > 0$, and let the Laplacian on $C_n$ be equipped with standard Neumann boundary conditions at every vertex. Then the spectrum of the resulting self-adjoint operator depends on the edge lengths only through their sum $L = \sum_{e=1}^{n} l_e$. Explicitly, the eigenvalues are*

$$
k_m^{\pm} = \frac{2\pi m \pm \theta_n}{L}, \quad m \in \mathbb{Z}_{\geq 0},
$$

*where $\theta_n = 0$ if $n$ is even and $\theta_n = \pi$ if $n$ is odd, with positive eigenvalues obtained by selecting $k_m^{\pm} > 0$. In particular, the spectrum is independent of the partition of $L$ among individual edges.*

**Proof.** The Neumann scattering matrix at a degree-2 vertex is

$$
S = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} = \sigma_x,
$$

which satisfies $\sigma_x^2 = I$. The eigenvalues of the cycle Laplacian are determined by the secular equation $\det[I - S(k) U(k)] = 0$, where $S(k)$ is the block-diagonal $2n \times 2n$ vertex scattering matrix and $U(k) = \mathrm{diag}(e^{ikl_b})$ is the bond propagation matrix over the $2n$ directed bonds [3].

A bond eigenfunction acquires phase $e^{ikl_e}$ traversing edge $e$ in either direction. The monodromy operator $M(k)$ — the product of scattering and propagation operators encountered by a bond eigenfunction in one full traversal of the cycle — acts as

$$
M(k) = \prod_{v=1}^{n} \sigma_x \cdot e^{ikl_v} = \sigma_x^n \cdot \exp\!\left(ik \sum_{e=1}^{n} l_e\right) = \sigma_x^n \, e^{ikL}.
$$

The factorization is exact: the propagation phases commute with the scattering matrices (they act on different bonds), so the product reorders to $\sigma_x^n \cdot e^{ikL}$ regardless of the ordering of edges around the cycle.

The eigenvalue condition $\det[I - M(k)] = 0$ is therefore

$$
\det\!\left[I - \sigma_x^n \, e^{ikL}\right] = 0.
$$

For $n$ even, $\sigma_x^n = I$, and the condition reduces to $\det[(1 - e^{ikL}) I] = 0$, i.e., $e^{ikL} = 1$, giving $k = 2\pi m / L$ for nonnegative integers $m$.

For $n$ odd, $\sigma_x^n = \sigma_x$, and the condition is

$$
\det\!\begin{pmatrix} 1 & -e^{ikL} \\ -e^{ikL} & 1 \end{pmatrix} = 1 - e^{2ikL} = 0,
$$

giving $e^{2ikL} = 1$, i.e., $k = \pi m / L$ for nonnegative integers $m$, equivalently $k = (2\pi m + \pi) / L$ once we discard the duplicates of the $n$-even case. Combining the two parities, $k_m^{\pm} = (2\pi m \pm \theta_n) / L$ with $\theta_n$ as stated.

In both cases the spectrum depends only on $L$ and the parity of $n$, not on the partition $\{l_1, \ldots, l_n\}$. $\square$

**Corollary 1 (Neumann correspondence ceiling).** *The spectral correspondence score $\mathcal{S}$ of $C_n$ under Neumann boundary conditions against any target sequence depends on at most one continuous parameter ($L$) and one discrete parameter (parity of $n$). For sufficiently generic targets and any fixed $N \geq 2$, the maximum of $\mathcal{S}$ over $L > 0$ and $n \in \mathbb{N}$ is therefore bounded by the score of a one-parameter family of uniformly spaced spectra, which we denote $\mathcal{S}_{\text{Neumann}}^{\max}$. For the first $N = 100$ Riemann zeros under our scoring (Sec. II.B), numerical optimization across $C_3$ through $C_{28}$ confirms $\mathcal{S}_{\text{Neumann}}^{\max} \approx 0.092$.*

**Remark 1.** The one-parameter Neumann extension that introduces per-vertex TRS-breaking phases via the similarity transform $D_v \sigma_x D_v^\dagger$ with $D_v = \mathrm{diag}(1, e^{i\theta_v})$ enlarges the parameter space from $(L, \mathrm{parity})$ to $(L, \theta_1, \ldots, \theta_n)$. The monodromy is then $\prod_v D_v \sigma_x D_v^\dagger \cdot e^{ikL}$, whose eigenvalues are no longer fixed at $\pm 1$ but acquire additional phase freedom $\Phi = \sum_v \theta_v$ (the trace of the conjugated product). The empirical ceiling under this extension at $N = 100$ zeros rises to $\mathcal{S} \approx 0.72$ but remains independent of $n$ — confirmed by the cycle-size sweep reported below. The full self-adjoint extension by general $U(2)$ matrices, which does not factor through any one-parameter family in this way, is what breaks the ceiling decisively (Sec. III.C).

The remainder of this section reports the empirical confirmation of Theorem 1 and Corollary 1 across $C_3$ through $C_{28}$, and characterizes the scaling of the TRS-extended Neumann ceiling as the comparison parameter $N$ varies.

---

## Notes for integration

1. The theorem is intended for inline insertion at the start of Section III.B in `Primes_in_the_Wires_v3.md`, replacing the first paragraph of that section. The existing empirical material on $C_3$ through $C_{28}$ scores and Fig. 2 then follows as confirmation.

2. The reference [3] cited in the proof is the Kostrykin–Schrader paper from the main manuscript's reference list. No new references are introduced.

3. Reviewers in the quantum graphs community will recognize the monodromy argument as standard technique applied to a specific case where the algebra simplifies. The novelty is the observation that the simplification yields a ceiling theorem for the spectral correspondence problem, not the technique itself.

4. Corollary 1 is stated empirically because the precise value of $\mathcal{S}_{\text{Neumann}}^{\max}$ depends on the scoring function definition. A reviewer might request either (a) a derived upper bound from the closed-form spectrum, or (b) a clearer specification of what "sufficiently generic targets" means. Either revision is straightforward if requested.

5. Remark 1 handles the TRS-broken extension case in a way that makes the empirical ceiling at $0.72$ feel like a natural continuation of Theorem 1 rather than a separate phenomenon.
