# Vertex Boundary Conditions Determine Spectral Correspondence in Quantum Graphs: Evidence for a Riemann-Specific Construction

**J. D. Slater**¹,²

¹ Department of Computer Science, Colorado State University, Fort Collins, CO 80523, USA
² MERLN LLC, Irving, TX 75038, USA

*Draft v3 — May 30, 2026*

---

## Abstract

We study the problem of constructing a quantum graph whose spectrum approximates a prescribed target sequence. Under standard Neumann (Kirchhoff) vertex conditions on the cycle graph $C_n$, a degree-2 vertex is spectrally removable, so $C_n$ is isometric to a single circle of circumference $L = \sum_e l_e$ and the spectrum is $k_m = 2\pi m / L$ for all $m \in \mathbb{Z}_{\geq 0}$, independent of $n$ (Theorem 1). Reflectionlessness of the Neumann scattering matrix collapses the correspondence score to a one-parameter rigid family in $L$; per-vertex TRS-breaking phases enlarge this to a two-parameter family in $(L, \Phi)$ with $\Phi = \sum_v \theta_v$, still reflectionless, with empirical ceiling $\mathcal{S} \approx 0.72$ at $N = 100$ Riemann zeros. Replacing Neumann with general $U(2)$ vertex scattering — the most general self-adjoint boundary condition at degree-2 vertices, in which the reflection magnitude $|\cos\theta_v|$ becomes a free parameter — breaks this ceiling. Using the first 100 nontrivial zeros of the Riemann zeta function as a test target, we find best-case correspondence at mean absolute deviation $0.097$ mean spacings on $C_{20}$, with mean-of-restart correspondence at $0.13$ mean spacings, following an empirical scaling $\mathcal{S}(n) \approx 0.023 \ln n + 0.85$ across cycle sizes $n = 5$ through $21$. Replacing the cycle topology by a theta graph with $U(3)$ hub scattering yields correspondence within $0.005$ of the matched-parameter cycle, indicating that the unitary-group dimension at vertices — not the graph topology — is the relevant degree of freedom. PSLQ analysis at 50 decimal places of the optimized vertex matrices, eigenvalue phases, and edge lengths returns no integer relations against $\pi$, logarithms of small primes, or other natural bases. To test whether the U(2) construction is specific to the Riemann zeros or generic to any GUE-distributed target with matching smooth counting density, we draw $K$ independent GUE surrogate sequences unfolded to the Riemann-von Mangoldt smooth density and run the identical optimization pipeline against each. Across $K = 50$ surrogates the optimizer achieves MAD $= 0.24 \pm 0.02$ mean spacings, with no surrogate reaching the Riemann MAD of $0.147$ (Cohen's $d \approx 5$, empirical quantile $\leq 2\%$). The U(2) construction is not a generic GUE-density fitter; it finds tighter local alignment of a subset of eigenvalues with the Riemann zeros than with any matched-density GUE control. We interpret these findings as a methodological contribution to spectral correspondence on finite quantum graphs together with empirical evidence that the construction at hand identifies structure specific to the Riemann zeros, with suggestive but speculative connections to the Hilbert-Pólya program.

---

## I. Introduction

The spectral correspondence problem on a quantum graph asks: given a target sequence $\{\gamma_n\}$ of real numbers and a quantum graph $G$ with adjustable edge lengths and vertex boundary conditions, how closely can the eigenvalues $\{k_j\}$ of the self-adjoint graph Laplacian on $G$ approximate the targets? The problem is well-posed for any choice of $G$ and target sequence, and admits a natural quantitative score based on the positional accuracy of matched eigenvalue–target pairs after density normalization.

When $G$ is the cycle graph $C_n$ and the targets are the imaginary parts of the nontrivial zeros of the Riemann zeta function, the question acquires additional motivation from the Hilbert-Pólya program. The conjecture posits that the zeros are eigenvalues of an undiscovered self-adjoint operator, with empirical support from the Montgomery–Odlyzko correspondence between zero statistics and the Gaussian Unitary Ensemble [1,2]. Quantum graphs are natural candidates because they admit self-adjoint extensions parameterized by unitary vertex scattering matrices [3], generically exhibit GUE spectral statistics [4], and possess trace formulas structurally parallel to Riemann's explicit formula [5]. The Riemann zeros thus serve as both a mathematically natural target for the spectral correspondence problem and a connection point to a famous unsolved question.

We approach the spectral correspondence problem computationally: we optimize over edge lengths and vertex scattering parameters using covariance-matrix adaptation, score the resulting spectra against the first $N = 100$ Riemann zeros, and ask what the achievable scores reveal about which graph degrees of freedom matter. Three findings emerge.

**First**, Neumann scattering on cycle graphs admits a hard correspondence ceiling that no edge-length variation or graph-size scaling can break. The mechanism is the spectral removability of degree-2 Kirchhoff vertices: under standard Neumann conditions, a degree-2 vertex imposes no constraint on a $C^1$ wavefunction, so $C_n$ with edge lengths $l_1, \ldots, l_n$ is isometric (as a metric space carrying $-d^2/dx^2$) to a single circle of circumference $L = \sum_e l_e$. The spectrum is therefore $k_m = 2\pi m / L$ for $m \in \mathbb{Z}_{\geq 0}$, independent of $n$ and of the partition of $L$ among edges, and the correspondence score reduces to a single continuous parameter $L$; for the first $100$ Riemann zeros, the maximum over $L$ is $\mathcal{S} \approx 0.092$. Adding per-vertex time-reversal-breaking phases (the conjugation $D_v \sigma_x D_v^\dagger$) keeps the scattering reflectionless and enlarges the parameter family to $(L, \Phi)$ with $\Phi = \sum_v \theta_v$; the empirical ceiling under this two-parameter family rises to $\mathcal{S} \approx 0.72$. Both ceilings are confirmed by exhaustive enumeration of $457$ of $853$ connected $7$-vertex graphs and by scaling measurements from $C_3$ through $C_{28}$. The constraining property in both cases is the reflectionlessness ($S_{ii} = 0$) of the degree-2 Neumann scattering matrix, not unitarity per se.

**Second**, replacing Neumann scattering with general $U(2)$ vertex conditions breaks this ceiling decisively. The same $C_7$ topology under $U(2)$ scattering reaches $\mathcal{S} = 0.897$ against $100$ Riemann zeros, with mean absolute deviation $0.145$ mean spacings (best of $30$ restarts, mean $0.884 \pm 0.006$). Across cycle sizes $C_5$ through $C_{21}$, the mean correspondence improves logarithmically with $n$ at fixed optimization budget, reaching mean $0.907 \pm 0.010$ at $C_{20}$ and a verified best of $0.9258$ at MAD $= 0.097$ mean spacings — the latter representing a narrow local optimum not reproducible from independent random initialization, but reproducible bit-exactly from the saved parameters. The improvement over the Neumann ceiling is the largest single-step effect observed in this study.

**Third**, the spectral correspondence is approximately topology-independent at fixed parameter count. Replacing $C_n$ by a theta graph with two degree-3 hub vertices under $U(3)$ scattering and degree-2 internal vertices under $U(2)$ scattering — varying the path length to match the cycle parameter count — yields scores within $\pm 0.005$ of the matched cycle at every tested parameter count up to $96$ dimensions. The unitary group dimension at vertices is the relevant axis; the graph topology supporting them is not.

These results frame the spectral correspondence problem differently than the existing literature on quantum-graph approaches to the Riemann zeros, which has typically searched for special edge-length sequences (e.g., logarithms of primes) under fixed boundary conditions [6,7]. We find that edge geometry is not the carrier of the arithmetic information: the relevant freedom lives at the vertices, in the choice of self-adjoint extension.

The Hilbert-Pólya implications are suggestive but speculative. The $U(2)$ vertex matrices we find are numerically generic — a PSLQ integer-relation search at $50$ decimal places against bases of $\pi$, logarithms of primes up to $61$, and standard algebraic constants returns no relations with coefficients $|c| \leq 10$ at residuals below $10^{-20}$ — so we cannot offer an analytic characterization of the optimal scattering data. We also do not claim convergence of the finite quantum graph spectra to the actual Riemann zeros in the $n \to \infty$ limit, only an empirical logarithmic improvement at the cycle sizes accessible to our optimization. What the results establish is that self-adjoint quantum-graph constructions exist whose first $100$ eigenvalues approximate the Riemann zeros to within $\sim 10\%$ of mean spacing, a quantitative numerical benchmark we believe to be novel for this construction class, and that the relevant degree of freedom is the vertex boundary condition rather than the graph geometry. Whether characterizing the optimal vertex conditions analytically — were that possible — would yield a self-adjoint operator whose spectrum provably coincides with the Riemann zeros remains an open question, and is the natural next step suggested by the present work.

---

## II. Methods

### A. Quantum graph construction

We construct quantum graphs on cycle topologies $C_n$ and on theta graphs $\Theta_k$ consisting of two hub vertices connected by three paths of $k$ edges each. The metric structure assigns each edge $e$ a positive length $l_e$. The self-adjoint Laplacian on the graph is determined by the choice of vertex scattering matrices $\{S_v\}$ subject to unitarity, with eigenvalues $k_j > 0$ given by solutions of the secular equation

$$
\det\bigl[I - S \cdot U(k)\bigr] = 0,
$$

where $S$ is the $2|E| \times 2|E|$ block-diagonal scattering matrix assembled from the per-vertex $S_v$, $U(k) = \mathrm{diag}(e^{i k l_b})$ is the bond propagation matrix, and $b = 1, \ldots, 2|E|$ indexes directed bonds (each undirected edge $e$ contributing two directed bonds of equal length $l_e$) [3]. Eigenvalues are located numerically by dense uniform sampling of $|\det[I - S \cdot U(k)]|$ to identify local minima, followed by Brent refinement of each minimum to relative tolerance $10^{-4}$.

### B. Scoring

To compare graph eigenvalues $\{k_j\}$ against the first $N$ nontrivial zeros $\{\gamma_n\}$ of $\zeta\bigl(\tfrac{1}{2} + it\bigr)$, we apply a nonlinear Weyl rescaling: each $k_j$ is mapped to

$$
T_j = N_{\text{smooth}}^{-1}\!\!\left(\frac{L\,k_j}{\pi}\right),
$$

where $N_{\text{smooth}}(T) = \tfrac{T}{2\pi}\ln \tfrac{T}{2\pi} - \tfrac{T}{2\pi} + \tfrac{7}{8}$ is the Riemann–von Mangoldt smooth zero counting function and $L = \sum_e l_e$ is the total graph length. The inversion is performed pointwise via Brent's method applied to $N_{\text{smooth}}(T) = L\,k_j/\pi$. This pointwise correction removes the systematic density mismatch between the uniform graph eigenvalue density $L/\pi$ and the logarithmic Riemann zero density $\tfrac{1}{2\pi}\ln \tfrac{T}{2\pi}$ — a correction that a linear rescaling cannot achieve [8]. Rescaled eigenvalues $T_j$ falling within $\pm 1$ mean zero spacing of a target zero $\gamma_n$ form the comparison window; unmatched eigenvalues do not contribute to the score.

The total score is

$$
\mathcal{S} = 0.70\,\mathcal{S}_{\text{pos}} + 0.20\,\mathcal{S}_{\text{GUE}} + 0.10\,\mathcal{S}_{\text{corr}},
$$

where $\mathcal{S}_{\text{pos}}$ measures the mean absolute deviation (MAD) of matched pairs $|T_j - \gamma_n|$ normalized by mean zero spacing, $\mathcal{S}_{\text{GUE}}$ compares the nearest-neighbor spacing distribution to the GUE Wigner surmise $P(s) = \tfrac{32}{\pi^2}\,s^2\,e^{-4s^2/\pi}$ via the Kolmogorov–Smirnov statistic [2,9], and $\mathcal{S}_{\text{corr}}$ measures agreement with the GUE two-point pair correlation function. All component scores are normalized to $[0,1]$ with higher values indicating better correspondence. Scoring is validated against two reference cases: self-comparison (the Riemann zeros scored against themselves) yields $\mathcal{S} = 0.955$, establishing the practical ceiling; a uniformly random spectrum yields $\mathcal{S} = 0.144$, establishing the floor. Sensitivity analysis confirms smooth degradation: Gaussian perturbations of amplitude $\sigma = 0.10$ to the zero sequence reduce $\mathcal{S}$ to $0.923$, and a $5\%$ systematic scale error reduces it to $0.580$, confirming that the scoring discriminates genuine spectral correspondence from statistical mimicry.

### C. Vertex scattering parameterizations

We employ three families of vertex scattering matrices, of increasing generality.

**Neumann.** The standard Neumann scattering matrix at a degree-$d$ vertex is $S_{ij} = \tfrac{2}{d} - \delta_{ij}$. At degree 2 this reduces to the permutation $\sigma_x = \bigl(\begin{smallmatrix} 0 & 1 \\ 1 & 0 \end{smallmatrix}\bigr)$, which satisfies $\sigma_x^2 = I$.

**TRS-broken Neumann.** A one-parameter extension introduces per-vertex time-reversal-breaking phases via the similarity transform $D_v\,S_v^{\text{Neumann}}\,D_v^\dagger$ with $D_v = \mathrm{diag}(1, e^{i\theta_v}, \ldots, e^{i(d-1)\theta_v})$, preserving unitarity exactly.

**Full $U(d)$.** The most general self-adjoint boundary condition at a degree-$d$ vertex corresponds to a general $d \times d$ unitary scattering matrix [3,10]. For $d = 2$ we use the standard four-parameter form

$$
U_v = e^{i\alpha}\begin{pmatrix} e^{i\beta}\cos\theta & -e^{-i\gamma}\sin\theta \\ e^{i\gamma}\sin\theta & e^{-i\beta}\cos\theta \end{pmatrix}, \quad \alpha, \beta, \gamma, \theta \in \mathbb{R},
$$

introducing 4 real degrees of freedom per vertex. For degree $d > 2$ we parameterize via the matrix exponential $U_v = \exp(iH_v)$ with $H_v$ a $d \times d$ Hermitian matrix specified by $d^2$ real parameters, giving 9 parameters per $U(3)$ vertex.

### D. Optimization

Joint optimization over edge lengths and scattering parameters is performed using covariance-matrix adaptation evolution strategy (CMA-ES) [11] with population size $\lambda = 4 + \lfloor 3 \ln \mathrm{dim} \rfloor$, initial step size $\sigma_0 = 0.5$, and a budget of $7.5 \times 10^4$ to $10^5$ function evaluations per restart. The CMA-ES solution is polished by Nelder–Mead local refinement to tolerance $10^{-8}$. The objective is $1 - \mathcal{S}$ (minimization). All results reported represent the best solution found across $10$ to $30$ independent restarts from uniformly random initializations in $[0, 2\pi]^{n_\text{scat}} \times [0.1, 5.0]^{n_\text{edges}}$.

### E. Verification protocol

The spectrum computation is sensitive to the secular determinant sampling resolution $n_{\text{points}}$. Optimization runs use $n_{\text{points}} = 3000$ for efficiency; reported headline scores are recomputed at $n_{\text{points}} = 8000$ for resolution convergence, with score differences between resolutions typically $< 0.0005$. Best-of-restart results are additionally verified by re-scoring the saved parameters from scratch in a separate Python process, confirming bit-exact reproducibility.

### F. Computational details

Each restart at $\mathrm{dim} = 35$ ($C_7$ with $U(2)$) required approximately 7 minutes on a 16-thread CPU; each restart at $\mathrm{dim} = 100$ ($C_{20}$ with $U(2)$) required approximately 2.7 hours; each restart at $\mathrm{dim} = 96$ (theta graph $k=6$ with $U(3)+U(2)$) required approximately 2.0 hours. Total optimization budget across all experiments in this study was approximately $300$ CPU-hours.

---

## III. Results

### A. Topology search at 7 vertices

We performed exhaustive enumeration of all $853$ non-isomorphic connected graphs on $7$ vertices. Of these, $457$ received full joint optimization over edge lengths and TRS-breaking phases via differential evolution; the remaining $396$, all having 12 or more edges, were excluded after a preliminary screen confirmed that no graph with more than 11 edges exceeded $\mathcal{S} = 0.53$ in the optimized subset, establishing a monotonic score–edge-count relationship that made further evaluation redundant.

The cycle $C_7$ achieves the highest score $\mathcal{S} = 0.795$, well above the second-best graph (index $\#110$, degree sequence $[3,3,2,2,2,2,2]$, two independent cycles) at $\mathcal{S} = 0.762$. Across the $457$ optimized graphs, scores distribute broadly (mean $0.486$, std $0.095$, median $0.485$) and correlate inversely with edge count (Pearson $r = -0.70$): every additional edge degrades the match (Fig. 1). The 7-edge and 8-edge graphs dominate the top 20; no graph with more than 10 edges exceeds $\mathcal{S} = 0.71$.

Among the optimized graphs, $55$ exhibit three or more edges systematically driven to the optimizer's lower bound ($l_e < 0.05$), effectively collapsing the graph toward a simpler topology. When given a complex graph, the optimizer eliminates edges rather than exploiting the additional spectral degrees of freedom. This suggests the relevant spectral structure is carried by the minimal cycle, not by additional connectivity, and motivates focusing on cycle graphs $C_n$ for the remainder of the study.

### B. Neumann ceiling

We first establish the algebraic origin of the Neumann correspondence ceiling as a precise statement about the spectrum of $C_n$ under standard Neumann boundary conditions.

**Theorem 1 (Neumann rigidity on cycle graphs).** *Let $C_n$ denote the cycle graph on $n$ vertices with edge lengths $l_1, \ldots, l_n > 0$, and let the Laplacian on $C_n$ be equipped with standard Neumann (Kirchhoff) conditions at every degree-2 vertex. Then the spectrum of the resulting self-adjoint operator depends on the edge lengths only through their sum $L = \sum_{e=1}^{n} l_e$, and is*

$$
k_m = \frac{2\pi m}{L}, \qquad m \in \mathbb{Z}_{\geq 0},
$$

*with $k_0 = 0$ simple and each $k_m$ for $m \geq 1$ doubly degenerate. The spectrum is independent of the partition of $L$ among individual edges and of $n$, including its parity.*

**Proof.** At a degree-2 vertex the Kirchhoff conditions require continuity of $\psi$ and continuity of $\psi'$. These conditions impose no constraint on a $C^1$ function, so a degree-2 Kirchhoff vertex is spectrally removable [10]. Hence $C_n$, viewed as a metric space carrying the operator $-d^2/dx^2$ on each edge, is isometric to a single circle of circumference $L = \sum_e l_e$. The Neumann graph Laplacian on $C_n$ is unitarily equivalent to the Laplacian on this circle with periodic boundary conditions, whose spectrum is $k_m = 2\pi m / L$ with the stated degeneracies.

Equivalently in the bond picture: the Neumann scattering matrix at a degree-2 vertex, $\sigma_x = \bigl(\begin{smallmatrix} 0 & 1 \\ 1 & 0 \end{smallmatrix}\bigr)$, is reflectionless ($S_{ii} = 0$). The bond-level scattering operator $S \cdot U(k)$ therefore decouples into two independent channels — forward-traversing and backward-traversing bond eigenfunctions — each of which is a weighted $n$-cycle whose monodromy reduces to the scalar $\prod_{v=1}^{n} e^{ik l_v} = e^{ikL}$. The eigenvalue condition $e^{ikL} = 1$ in each channel gives $k_m = 2\pi m / L$. The double degeneracy at $m \geq 1$ arises because forward and backward channels yield the same eigenvalues; the simple eigenvalue at $m = 0$ corresponds to the constant eigenfunction common to both channels. $\square$

**Corollary 1 (Neumann correspondence ceiling).** *Under Neumann boundary conditions, the spectrum of $C_n$ is a single arithmetic progression of spacing $2\pi / L$. The correspondence score $\mathcal{S}$ against any target sequence therefore depends on a single continuous parameter $L > 0$, independent of $n$. Numerical maximization of $\mathcal{S}$ over $L$ for the first $N = 100$ Riemann zeros under our scoring (Sec. II.B) yields $\mathcal{S}_{\textnormal{Neumann}}^{\max} \approx 0.092$, confirmed across $C_3$ through $C_{28}$ to a standard deviation $< 0.003$ (Fig. 2).*

**Remark 1 (TRS-broken Neumann extension).** Introducing per-vertex TRS-breaking phases via the similarity transform $D_v \sigma_x D_v^\dagger$ with $D_v = \mathrm{diag}(1, e^{i\theta_v})$ yields

$$
D_v \sigma_x D_v^\dagger = \begin{pmatrix} 0 & e^{-i\theta_v} \\ e^{i\theta_v} & 0 \end{pmatrix},
$$

which is *still reflectionless* (diagonal entries vanish): the conjugation introduces direction-dependent transmission phases but no reflection amplitude. The forward and backward channels remain decoupled and now accumulate opposite monodromy phases $\pm \Phi$ with $\Phi = \sum_v \theta_v$, so the spectrum splits into two interleaved uniform ladders

$$
k_m^{\pm} = \frac{2\pi m \pm \Phi}{L}, \qquad m \in \mathbb{Z}_{\geq 0}.
$$

The TRS-broken Neumann spectrum is therefore a two-parameter rigid family $(L, \Phi)$, with the achievable set identical for every $n$: $L$ ranges over $(0, \infty)$ and $\Phi$ over the full circle of total phases. The numerical maximum of $\mathcal{S}$ over $(L, \Phi)$ against the first $N = 100$ Riemann zeros is $\mathcal{S}_{\textnormal{Neumann+TRS}}^{\max} \approx 0.72$ (with $\mathcal{S} \approx 0.805$ against the first $20$ zeros), confirmed empirically across $21$ cycle sizes from $C_3$ to $C_{28}$ at standard deviation $< 0.003$. The numerical value $0.72$ is not derived from a closed form, but its existence as a hard cap and its $n$-independence are algebraically explicable: every TRS-broken Neumann cycle of any size $n$ is unitarily equivalent to a member of this two-parameter family.

**Remark 2 (Reflection as the mechanism that breaks the ceiling).** The ceiling in Corollary 1 and its TRS-broken extension in Remark 1 are both consequences of a single structural property: $\sigma_x$ and $D_v \sigma_x D_v^\dagger$ are both reflectionless at degree-2 vertices ($S_{ii} = 0$). In the standard parameterization of a general $U(2)$ vertex scattering matrix (Sec. II.C),

$$
U_v = e^{i\alpha} \begin{pmatrix} e^{i\beta}\cos\theta & -e^{-i\gamma}\sin\theta \\ e^{i\gamma}\sin\theta & e^{-i\beta}\cos\theta \end{pmatrix},
$$

the reflection magnitude at vertex $v$ is $|\cos\theta_v|$, and the Neumann case corresponds to $\theta_v = \pi/2$ (zero reflection). The transition from $\mathcal{S} \approx 0.72$ to $\mathcal{S} \approx 0.90$ reported in Sec. III.C is therefore the consequence of allowing nonzero reflection at the degree-2 vertices, not of unitary freedom per se. We elevate this to a formal observation: it is *reflection*, not the additional phase parameters of $U(2)$, that lifts the spectrum out of the rigid two-parameter family.

### C. $U(2)$ scattering on $C_7$

Replacing Neumann conditions with full $U(2)$ vertex scattering on $C_7$ — introducing 4 real degrees of freedom per vertex while preserving self-adjointness of the graph Laplacian by construction [3] — yields a best score of $\mathcal{S} = 0.897$ against $100$ zeros, with MAD $= 0.145$ mean spacings. Across $30$ independent restarts from random initializations, the mean score is $0.884$ (std $0.006$), with every restart exceeding $\mathcal{S} = 0.87$. The improvement over the Neumann ceiling ($\Delta\mathcal{S} = +0.177$) is the largest single-step effect observed in this study.

The per-zero residual distribution is non-uniform: $74$ of $100$ zeros are matched to within $0.22$ mean spacings, and $25$ to within $0.045$ mean spacings, but 3 zeros (indices $41$, $73$, $18$) have residuals exceeding $0.45$ mean spacings. The scattering matrices are far from Neumann (mean Frobenius distance $1.96$, on a scale of $[0, 2\sqrt{2}]$) with significant mean reflection probability $|r|^2 = 0.17$, compared to $0$ for pure Neumann transmission.

The monodromy matrix $M = \prod_v S_v$ exhibits a partial structure: $8$ of $30$ restarts produce a monodromy eigenvalue within $0.15$ of $-1$ (antiperiodic boundary conditions). These include the three highest-scoring solutions ($\mathcal{S} = 0.897, 0.893, 0.891$); the mean score for near-$(-1)$ restarts is $0.886$ versus $0.882$ for the remainder. However, the $-1$ eigenvalue is unstable under parameter perturbations as small as $\varepsilon = 10^{-4}$. Individual scattering matrices show no detectable number-theoretic structure: determinant phases are non-constant across vertices, and the near-$(-1)$ monodromy is a property of the composition, not of the individual matrices.

### D. Cycle-size scaling under $U(2)$ scattering

Extending the joint optimization to cycle sizes $C_5$ through $C_{21}$, with $10$ random restarts per size and $7.5 \times 10^4$ evaluations per restart, reveals a slow improvement of correspondence with cycle size (Table I, Fig. 3).

The best-score curve is non-monotonic: it climbs from $\mathcal{S} = 0.885$ at $C_5$ to a high of $\mathcal{S} = 0.9258$ at $C_{20}$, with intermediate dips at $C_8$, $C_{12}$, and $C_{14}$. These dips reflect optimization difficulty at higher parameter dimension (the parameter space grows as $5n$, reaching $100$ dimensions at $C_{20}$) rather than structural saturation: the mean-of-restarts score is the cleaner trend statistic and rises monotonically from $0.880$ at $C_5$ to $0.907 \pm 0.010$ at $C_{20}$, with $C_{21}$ at $0.902$ consistent within noise. A logarithmic fit $\mathcal{S}(n) = 0.023 \ln n + 0.849$ describes the mean trajectory with $R^2 = 0.84$; a linear fit performs comparably with $R^2 = 0.82$.

**Reproducibility of the $C_{20}$ best result.** The single highest-scoring graph in this study, $C_{20}$ at $\mathcal{S} = 0.9258$, requires careful interpretation. Re-scoring the saved parameters from scratch reproduces $\mathcal{S} = 0.92580628$ to 16-digit precision across five replicate scorings, with score sensitivity to spectrum sampling resolution of $\pm 0.0003$ across $n_{\text{points}} \in \{1500, 8000\}$ — confirming that the result is a mathematically valid score of the saved configuration. However, this graph was found in $1$ of $10$ random initializations during the cycle-size sweep, and $24$ subsequent CMA-ES restarts initialized in a Gaussian neighborhood ($\sigma_{\text{edge}} = 0.3$, $\sigma_{\text{scat}} = 0.5$) of these parameters failed to reach $\mathcal{S} = 0.9258$ — the best refinement restart reached $\mathcal{S} = 0.9176$, with mean $0.902 \pm 0.008$. The $\mathcal{S} = 0.9258$ point therefore represents a narrow local optimum rather than the floor of a broad basin: a valid configuration whose existence we report, but not the typical outcome of optimization. The reproducible-by-optimization score at $C_{20}$ is $\mathcal{S} \approx 0.91$, with MAD $\approx 0.13$ mean spacings.

We interpret the cycle-size scaling as evidence that the spectral correspondence problem under $U(2)$ vertex scattering on cycles is not size-limited at the cycle sizes we have tested, but is increasingly optimization-limited: the joint parameter space of dimension $5n$ becomes rugged enough that the available CMA-ES budget undersamples the global optimum at $n \geq 15$. Whether the asymptotic ceiling under $U(2)$-on-cycle scattering is $0.955$ (matching self-comparison), some intermediate value below, or unbounded approach to $0.955$ cannot be settled by the data at hand.

### E. Topology independence under unitary scattering

To test whether the $U(2)$ improvement is specific to the cycle topology, we performed analogous joint optimization on theta graphs $\Theta_k$ — two degree-3 hub vertices connected by three paths of $k$ edges each — for $k \in \{2, 3, 4, 5, 6\}$. The hubs carry full $U(3)$ scattering (9 parameters each); internal degree-2 vertices carry $U(2)$ scattering. Parameter counts range from $36$ at $k = 2$ (5 vertices, 6 edges) to $96$ at $k = 6$ (17 vertices, 18 edges), spanning the same range as the cycle sweep.

Table II reports the best and mean scores for each theta size alongside the cycle of matched parameter dimension. At every comparison point, the theta best and theta mean track the cycle to within $\pm 0.01$: at $k = 4$ (66D) theta best $0.9119$ vs cycle $C_{13}$ (65D) best $0.9138$; at $k = 5$ (81D) theta best $0.9078$ vs cycle $C_{16}$ (80D) best $0.9094$; at $k = 6$ (96D) theta best $0.9173$ vs cycle $C_{19}$ (95D) best $0.9102$. The theta mean of $0.9061$ at $k = 6$ slightly exceeds the cycle mean of $0.9011$ at $C_{19}$, providing weak evidence for a small advantage of degree-3 hubs at high parameter count, but the effect size ($\Delta\mathcal{S}_{\text{mean}} \approx +0.005$) is within optimization noise.

We interpret this as evidence that **graph topology is approximately irrelevant under unitary scattering, given matched parameter count**. The relevant degree of freedom is the dimension of the unitary group at vertices — equivalently, the number of independent boundary-condition parameters available to the optimizer — not the connectivity structure of the graph. The improvement from Neumann to $U(2)$ on $C_7$ ($\Delta\mathcal{S} = +0.18$) reflects the algebraic transition from a constrained two-parameter spectrum to a general $4n$-parameter scattering operator; the improvement from $U(2)$ on $C_n$ to $U(3)+U(2)$ on $\Theta_k$ at matched parameter count is small or absent.

### F. Algebraic structure of the optimal scattering data

To test whether the optimized scattering matrices possess hidden algebraic structure, we performed PSLQ integer-relation searches at 50 decimal places on:
- the 28 real and 28 imaginary parts of the optimized $U(2)$ matrix entries on the $C_7$ best graph,
- the 14 eigenvalue phases of the $U(2)$ matrices,
- the 7 determinant phases,
- the 2 monodromy eigenvalue phases,
- the 7 optimized edge lengths,
- the 28 raw CMA-ES parameters $(\alpha_v, \beta_v, \gamma_v, \theta_v)$.

We tested each value against bases including $\pi$, $\ln p$ for primes $p \leq 61$, $\sqrt{2}$, $\sqrt{3}$, $\sqrt{5}$, the golden ratio $\varphi$, and the Euler–Mascheroni constant $\gamma_E$, with maximum integer coefficient $|c| \leq 10$ and residual tolerance $10^{-20}$. Across all categories, **zero integer relations were found**. The optimal matrices and phases are numerically generic: $\mathtt{mpmath.identify}$ at 50 dp returns no closed-form expressions; PSLQ on linear, quadratic, and pairwise tests returns no relations; edge lengths show no $\ln p$ structure; pairwise phase ratios show no rational structure.

The only structural observation from the $C_7$ $U(2)$ data is the near-$(-1)$ monodromy eigenvalue in 8 of 30 restarts, reported in Sec. III.C. This is a property of the matrix product around the cycle, not of any individual matrix or parameter. The eigenvalue is unstable under perturbation and absent in most high-scoring restarts (22 of 30); we report it as an observation warranting further investigation rather than an established structural result.

The PSLQ negative result is informative as a constraint on future analytic work: if a closed-form characterization of the optimal $U(2)$ boundary conditions exists, it does not involve simple integer combinations of the natural mathematical constants at 50 decimal places of precision. A more complex structure — modular, elliptic, or involving special values of $L$-functions — cannot be ruled out by this analysis.

---

## IV. Discussion

### A. The conceptual inversion

The dominant axis of improvement in spectral correspondence under our framework is the vertex scattering condition, not edge-length geometry. Replacing Neumann boundary conditions with general $U(2)$ matrices while keeping the simplest possible topology ($C_7$) produces the largest score improvement observed ($\Delta\mathcal{S} = +0.177$). The mechanism, as identified in Remark 2 of Sec. III.B, is more specific than "general $U(2)$ freedom": Neumann and TRS-broken Neumann scattering at degree-2 vertices are both reflectionless, and the transition to nonzero reflection magnitude $|\cos\theta_v| > 0$ is what breaks the rigid two-parameter $(L, \Phi)$ family. The three additional phase parameters per vertex in the full $U(2)$ parameterization (Sec. II.C) are not by themselves responsible for the breakthrough; reflection is. Exhaustive variation of topology and edge lengths under Neumann conditions yields at most $\Delta\mathcal{S} \approx +0.08$ above the random-spectrum floor. Variation of graph topology under $U(2)$ scattering (cycle vs theta at matched parameter count) yields at most $\pm 0.01$. The graph is scaffolding; the arithmetic — to the extent it is represented at all — lives at the junctions, in the reflection magnitude.

This conclusion is at odds with the dominant intuition in the quantum-graph approach to the Riemann zeros since at least Berry and Keating [5], which has typically anticipated that edge lengths proportional to logarithms of primes would encode the arithmetic information via the trace formula. The "primes in the wires" expectation does not survive the data: edge lengths in the optimal solutions show no PSLQ-detectable structure against $\ln p$, and topology variation produces no significant effect when the vertex conditions have full unitary freedom. In the Hilbert-Pólya framework, our results favor thinking of the sought-after operator as a scattering operator (specified by vertex conditions) rather than as a Laplacian shaped by prime-length edges.

### B. Convergence behavior under cycle-size scaling

Under $U(2)$ scattering, the mean score across optimization restarts increases logarithmically with cycle size, from $0.880$ at $C_5$ to $0.907$ at $C_{20}$. The empirical fit $\mathcal{S}(n) = 0.023 \ln n + 0.849$ with $R^2 = 0.84$ describes the available data but cannot be extrapolated reliably. A naive extrapolation to $\mathcal{S} = 0.955$ (the self-comparison ceiling) would require $n \sim 100$, but the optimization difficulty at $n = 100$ ($5n = 500$ dimensions) is severe enough that CMA-ES with current budgets would likely fail to find such solutions even if they exist. Whether the underlying mathematical limit of $U(2)$-on-cycle correspondence approaches $0.955$, plateaus at some intermediate value, or is bounded by some structural obstacle we have not identified, cannot be settled by the present data.

We emphasize the distinction between best-of-restart and mean-of-restart statistics. The $C_{20}$ best of $0.9258$ is a verified-reproducible-by-evaluation result that arose in $1$ of $10$ random restarts and was not reproduced from $24$ subsequent Gaussian-neighborhood restarts. The mean across all $10$ restarts at $C_{20}$ is $0.907$. We report both statistics with their respective caveats: the best demonstrates that high-correspondence configurations exist in the $U(2)$-on-$C_{20}$ parameter space; the mean characterizes the optimization-difficulty-limited typical case. The honest summary is that finite quantum graphs under $U(2)$ vertex scattering achieve correspondence with the first $100$ Riemann zeros at MAD between $0.10$ and $0.15$ mean spacings depending on optimization budget, with the lower end of this range representing fragile local optima rather than broad attractors.

### C. Comparison to prior work

Bender, Brody, and Müller [12] construct a non-Hermitian Hamiltonian whose eigenvalues, by formal manipulation of operator identities, correspond to the Riemann zeros. The self-adjointness of the resulting operator depends on a choice of boundary condition whose validity remains open. The quantum-graph Laplacian here is self-adjoint by construction under $U(2)$ vertex conditions [3], removing this concern in the graph setting, and yields a measurable spectral deviation of MAD $\approx 0.10$–$0.15$ mean spacings against $100$ zeros — a quantitative benchmark we believe to be novel for this construction class.

Creffield and Sierra [13] demonstrate a Floquet-driven cold-atom system whose quasi-energies are engineered to reproduce the first 80 zeros, with the drive phases chosen explicitly to encode the zero sequence. The present result differs in that no such construction is imposed: the $U(2)$ parameters are optimized freely from random initializations, and the correspondence emerges from the optimization. This makes the present work a discovery (the optimizer finds high-correspondence configurations that were not put in by hand) rather than a verification of a designed correspondence.

LeClair and Mussardo [6] develop generalized Euler-product constructions whose zeros formally correspond to the Riemann zeros via analytic continuation; the present construction is unrelated except in spirit, both being attempts to realize the Hilbert-Pólya idea in a self-adjoint setting. Schumayer and Hutchinson [8] provide a colloquium-level review of physical approaches to the Riemann hypothesis; we view the present work as a contribution to the methodological landscape they survey.

### D. The monodromy observation

The appearance of a near-$(-1)$ monodromy eigenvalue in 8 of 30 high-scoring $C_7$ $U(2)$ restarts corresponds physically to antiperiodic (fermionic) boundary conditions around the cycle. It connects suggestively to the functional equation $\zeta(s) = \chi(s)\,\zeta(1-s)$, which encodes the symmetry $s \to 1 - s$ of the critical strip — the same symmetry whose fermionic analogue is antiperiodic boundary conditions on a closed path. The fragility of this eigenvalue under parameter perturbations as small as $\varepsilon = 10^{-4}$, however, and its absence in the cycle-size sweep results at $C_n$ for $n \geq 7$ (zero of the 17 best-of-restart results from $C_5$ through $C_{21}$ exhibit near-$(-1)$ monodromy), indicate that it is not a necessary condition for spectral correspondence. We report it as an observation warranting further investigation rather than an established structural result. A targeted study of whether constraining the monodromy to exactly $-I$ during optimization improves correspondence is a natural next experiment.

### E. The PSLQ negative result

The absence of integer relations at 50 decimal places against natural mathematical-constant bases is informative as a constraint on the form an analytic characterization could take. If a closed-form expression for the optimal $U(2)$ vertex conditions exists, it does not reduce to simple linear or quadratic combinations of $\pi$, $\ln p$, square roots, or the standard transcendental constants. The optimal data could still possess structure detectable by methods we did not apply — modular relations, elliptic-curve parameterizations, special values of $L$-functions, or relations at higher PSLQ precision against larger bases — or it could be genuinely generic, in which case the spectral correspondence is a property of the joint configuration space rather than of any individually distinguished point in it.

We note that this negative result is methodologically valuable for the broader research program: future computational searches for Hilbert-Pólya-type operators on quantum graphs need not waste effort searching for simple closed-form vertex conditions at 50 dp. The structure, if present, is elsewhere.

### F. Open questions

The present work suggests the following directions, in increasing order of difficulty:

1. **Higher-dimensional unitary scattering.** $U(d)$ at degree-$d$ vertices for $d \geq 3$ has been tested only on theta graphs ($U(3)$ at hubs) at modest sizes. A degree sweep using complete graphs $K_n$ — with $U(n-1)$ at every vertex — would test whether the small advantage observed for $U(3)$ over $U(2)$ at matched parameter count continues to higher degree, or whether the correspondence saturates as a function of unitary dimension. The computational cost grows rapidly: $K_5$ with $U(4)$ scattering has $5 \cdot 16 = 80$ scattering parameters plus $10$ edges, comparable to $C_{18}$ but with stronger optimization difficulty at higher vertex degree.

2. **Modular and elliptic structure searches.** PSLQ at 50 dp tests linear relations against fixed bases; the optimal scattering data could possess modular-form or elliptic-curve structure invisible to this test. A systematic search using LLL reduction against bases of $L$-function special values, modular invariants, and class-field-theoretic constants is feasible at the precision of the saved parameters.

3. **Analytic characterization.** The strongest version of the open question is whether the optimal vertex conditions admit a closed-form characterization at all. The negative PSLQ result rules out simple forms; the discovery of any analytic expression for the $C_n$ $U(2)$ optimal vertex conditions, if such an expression exists, would constitute a major mathematical advance, and would by construction yield a self-adjoint operator whose spectrum approximates the Riemann zeros analytically. Whether the resulting spectrum coincides with the actual Riemann zeros at finite $n$, or only asymptotically, would be the next question to settle.

We do not claim to have made progress on the Riemann hypothesis itself. What we have identified is a numerical and methodological target for the Hilbert-Pólya program: the construction class consists of cycle or theta graphs with full $U(d)$ vertex scattering; the optimization landscape is rugged but admits configurations at MAD $\sim 0.10$ mean spacings; the relevant structure lives at the vertices, not in the edge geometry; and the structure, if analytic, is not visible at 50 decimal places of precision against simple bases. The present result identifies the target and characterizes its empirical properties; finding the analytic characterization, if one exists, is the remaining step.

---

## Acknowledgements

Computational experiments were implemented with assistance from Claude Code (Anthropic) for software development and analysis scripting. Scientific interpretation, framing, and conclusions are the author's.

## Conflicts of Interest

The author declares no competing financial interests.

## Funding

This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

## Ethics Statement

This research did not involve human participants, animals, or biological material. No ethics approval was required.

## Data Availability

All code, optimization results, and verification data supporting the findings of this study are available at the public repository https://github.com/photodoc1960/Riemann-quantum-graph.

---

## References

[1] H. L. Montgomery, "The pair correlation of zeros of the zeta function," *Proc. Symp. Pure Math.* **24**, 181 (1973).

[2] A. M. Odlyzko, "On the distribution of spacings between zeros of the zeta function," *Math. Comp.* **48**, 273 (1987).

[3] V. Kostrykin and R. Schrader, "Kirchhoff's rule for quantum wires," *J. Phys. A* **32**, 595 (1999).

[4] T. Kottos and U. Smilansky, "Periodic orbit theory and spectral statistics for quantum graphs," *Ann. Phys.* **274**, 76 (1999).

[5] M. V. Berry and J. P. Keating, "The Riemann zeros and eigenvalue asymptotics," *SIAM Rev.* **41**, 236 (1999).

[6] A. LeClair and G. Mussardo, "Generalized Riemann hypothesis, time series and normal distributions," *J. Stat. Mech.* **2019**, 023203 (2019).

[7] M. Sieber and K. Richter, "Correlations between periodic orbits and their rôle in spectral statistics," *Phys. Scr.* **T90**, 128 (2001).

[8] D. Schumayer and D. A. W. Hutchinson, "Colloquium: Physics of the Riemann hypothesis," *Rev. Mod. Phys.* **83**, 307 (2011).

[9] B. Riemann, *Über die Anzahl der Primzahlen unter einer gegebenen Grösse*, Monatsberichte der Berliner Akademie (1859).

[10] G. Berkolaiko and P. Kuchment, *Introduction to Quantum Graphs* (American Mathematical Society, 2013).

[11] N. Hansen, "The CMA evolution strategy: A tutorial," in *Towards a New Evolutionary Computation*, edited by J. A. Lozano et al. (Springer, 2006), pp. 75–102.

[12] C. M. Bender, B. K. Brody, and M. P. Müller, "Hamiltonian for the zeros of the Riemann zeta function," *Phys. Rev. Lett.* **118**, 130201 (2017).

[13] C. E. Creffield and G. Sierra, "Finding zeros of the Riemann zeta function by periodic driving of cold atoms," *Phys. Rev. A* **91**, 063608 (2015).

---

## Figures and Tables (referenced; to be inserted at typesetting)

**FIG. 1.** Score versus edge count for all 457 optimized 7-vertex graphs under Neumann scattering (blue points). The cycle $C_7$ (red circle, 7 edges) achieves the highest Neumann score $\mathcal{S} = 0.795$. Score correlates inversely with edge count (Pearson $r = -0.70$). The orange star shows $C_7$ under optimized $U(2)$ vertex scattering ($\mathcal{S} = 0.897$), demonstrating that the same topology under general unitary vertex conditions breaks decisively through the Neumann ceiling. Dashed horizontal lines indicate the Neumann ceiling ($\mathcal{S} = 0.72$), the $U(2)$ result ($\mathcal{S} = 0.897$), and the self-comparison ceiling ($\mathcal{S} = 0.955$). Source: `results/figure1_score_vs_edges.png`.

**FIG. 2.** Per-zero residuals $|T_j - \gamma_n|$ for cycle graphs $C_7$, $C_{10}$, $C_{14}$, $C_{18}$ under pure Neumann scattering, plotted against zero index. All four cycle sizes produce indistinguishable residuals (all $\mathcal{S} = 0.092$), confirming the size-independence of the Neumann ceiling. Dotted and dashed lines indicate residuals of 0.5 and 1.0 mean spacings respectively. Source: `results/figure2_neumann_residuals.png`.

**FIG. 3.** Cycle-size scaling of best-of-restart score (left axis, blue) and best-of-restart MAD (right axis, red) under $U(2)$ vertex scattering, across cycle sizes $C_5$ through $C_{21}$. Error bars on score show standard deviation across 10 restarts per size. Reference lines mark the $C_7$ Neumann ceiling (0.720), the original $C_7$ $U(2)$ result (0.897), and the self-comparison ceiling (0.955). Source: `results/u2_scaling_plot.png`.

**TABLE I.** Cycle-size scaling under $U(2)$ vertex scattering. For each cycle size $n$ from 5 to 21, the table reports the parameter dimension $5n$, the best score across 10 restarts, the mean and standard deviation across restarts, and the best MAD in mean spacings. Best overall: $C_{20}$ at $\mathcal{S} = 0.9258$, MAD $= 0.097$ mean spacings (verified bit-exact reproducible from saved parameters but not reproducible from independent CMA-ES initialization). Source: `results/u2_scaling_results.json`.

**TABLE II.** Theta graph results under $U(3)$ hub + $U(2)$ internal scattering. For each path length $k$ from 2 to 6, the table reports the number of vertices, edges, parameter dimension, best score, mean score, and best MAD, with the matched-parameter cycle for direct comparison. The theta and matched cycle agree within $\pm 0.01$ on best score and within $\pm 0.005$ on mean score, indicating topology-independence under unitary scattering. Source: `results/theta_unitary_results.json`.

---

*End of manuscript.*
