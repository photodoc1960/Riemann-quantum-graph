# Cover Letter — Communications in Number Theory and Physics

*For submission of: Primes_in_the_Wires_v3.md*

*Draft date: June 6, 2026*

---

**To**: Editorial Board, *Communications in Number Theory and Physics*

**Re**: Submission — "Vertex Boundary Conditions Determine Spectral Correspondence in Quantum Graphs: Evidence for a Riemann-Specific Construction"

Dear Editors,

I am submitting the enclosed manuscript, *"Vertex Boundary Conditions Determine Spectral Correspondence in Quantum Graphs: Evidence for a Riemann-Specific Construction,"* for consideration in *Communications in Number Theory and Physics*. The work is a computational study of the spectral correspondence problem on finite quantum graphs, with four principal contributions situated within the Hilbert-Pólya program.

**The first contribution is a small but novel theorem on the rigidity of Neumann boundary conditions.** Under standard Kirchhoff (Neumann) conditions, a degree-2 vertex is spectrally removable, so the cycle graph $C_n$ is isometric to a single circle of circumference $L = \sum_e l_e$ and its spectrum is $k_m = 2\pi m / L$ for all $n \in \mathbb{N}$, independent of the partition of $L$ among edges. The associated correspondence score depends on a single continuous parameter $L$ alone; under per-vertex TRS-breaking phases (still reflectionless) the family enlarges to two parameters $(L, \Phi)$ with $\Phi = \sum_v \theta_v$. The construction breaks through the resulting $\mathcal{S} \approx 0.72$ ceiling only when reflection at degree-2 vertices is allowed — that is, only when one moves to a vertex condition outside the reflectionless similarity class of Neumann. The argument is brief and, to my knowledge, has not been stated in this form in the literature.

**The second contribution is a numerical demonstration that general U(2) vertex scattering breaks the Neumann ceiling.** Using the first 100 nontrivial Riemann zeros as a test target, the optimized $C_n$ U(2) construction achieves mean absolute deviation MAD = 0.10–0.15 mean spacings across cycle sizes $n = 5$ to $21$, with a verified best of MAD = 0.097 on $C_{20}$. The self-adjointness of the resulting graph Laplacian is guaranteed by the Kostrykin–Schrader characterization of self-adjoint extensions.

**The third contribution is an empirical topology-independence result.** Replacing the cycle topology with theta graphs of matched parameter count produces spectral correspondence within ±0.005 of the matched cycle at every parameter count tested up to 96 dimensions. The relevant degree of freedom is the dimension of the unitary group at vertices — equivalently, the reflection amplitude that the boundary condition admits — not the connectivity structure of the graph. A PSLQ search at 50 decimal places confirms that the optimal vertex matrices contain no detectable integer relations against $\pi$, prime logarithms, or other natural mathematical constants.

**The fourth contribution is a direct empirical test of whether the U(2) construction finds structure specific to the Riemann zeros or instead represents a generic capacity to fit any GUE-distributed target of matching smooth density.** I drew $K = 50$ independent GUE surrogate target sequences, each unfolded to match the Riemann-von Mangoldt smooth counting density, and ran the identical CMA-ES + Nelder-Mead optimization pipeline against each. The surrogate distribution converged at $K = 50$ to MAD = 0.260 ± 0.035 mean spacings; *no* surrogate reached the Riemann MAD of 0.147 (empirical quantile 0/50, Cohen's $d = 3.21$, base-rate floor under the null hypothesis 1.96%). A complementary asymmetry emerged in matched-eigenvalue counts: the optimizer placed graph eigenvalues within the matching window of all 100 surrogate targets in every one of the 50 surrogates, while it matched only 74 of 100 Riemann targets — at substantially tighter precision on the matched subset. The result is not consistent with generic GUE-density fitting. The construction finds structure specific to the Riemann zeros, even if the PSLQ analysis above shows that this structure is not visible as integer relations at 50 decimal places.

These four results together support the manuscript's central methodological claim — that spectral correspondence on finite quantum graphs is carried by the vertex boundary condition rather than the edge geometry — and the empirical claim, established by the surrogate control, that the construction at hand identifies structure specific to the Riemann zeros. The manuscript does not claim convergence of the finite construction to the Hilbert-Pólya operator in any well-defined limit; the connections to the Hilbert-Pólya program are framed throughout as suggestive rather than established.

The principal results figure, *Figure 4* (the K = 50 surrogate MAD histogram with the Riemann baseline overlaid), is included with the manuscript. All code, optimization results, the Theorem 1 proof, all four manuscript figures, and the complete K = 50 surrogate data deposit are publicly available at [https://github.com/photodoc1960/Riemann-quantum-graph](https://github.com/photodoc1960/Riemann-quantum-graph). The reproducibility commands and runtime estimates are documented in the repository README.

This work has not been previously published and is not under consideration at any other journal. The author has no competing financial interests to declare.

I would suggest the following potential referees, all active in quantum graph spectral theory or related mathematical-physics areas. I have no co-authorship history with any of them:

- Gregory Berkolaiko, Texas A&M University
- Uzy Smilansky, Weizmann Institute of Science
- Germán Sierra, IFT Madrid
- Sven Gnutzmann, University of Nottingham

Thank you for considering this submission.

Sincerely,

J. D. Slater
Department of Computer Science (on leave)
Colorado State University
jdslater@colostate.edu
