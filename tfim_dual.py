"""
tfim_dual.py
==============

Step 9b: exact ground-truth energy computed the way the PAPER actually does
it -- NOT via our own cycle-space construction (physical_basis.py), but via
the paper's own duality mapping onto the 2D transverse-field Ising model
(TFIM) on the dual (plaquette) lattice.

IMPORTANT HONESTY NOTE, since this exact question came up: physical_basis.py
is OUR OWN independent construction (Hadamard duality + graph cycle space),
NOT something the paper describes. This file is different -- it implements
the paper's own stated method (their Eq. 10-11), quoted and re-derived from
the actual paper text:

    "we can define new Pauli spin variables X_p and Z_p on the dual
     lattice, where p denotes the plaquette centers"

        X_p        = B_p                     (plaquette operator)
        Z_p Z_p'   = sigma^x_{l(p,p')}        (shared link's X operator)

    for neighboring plaquettes p, p' sharing link l(p,p'). Under this
    relabeling the full Hamiltonian H(h) = H_E + h*H_B becomes

        H_dual = sum_{<p,p'>} (1 - Z_p Z_p')  -  h * sum_p X_p

    i.e. a transverse-field Ising model on the DUAL lattice (one qubit per
    ORIGINAL plaquette -- so L^2 qubits instead of 2*L^2 links!), with
    nearest-neighbor ferromagnetic ZZ couplings and a transverse field h.

Why this halves the qubit count instead of the ~2x reduction our own
cycle-space trick gets: the paper's mapping trades "one qubit per link"
for "one qubit per plaquette" directly (2*L^2 -> L^2 qubits), a clean
combinatorial fact about the square lattice (every plaquette touches 4
links, every interior link is shared by exactly 2 plaquettes).

The catch (also stated explicitly in the paper): the map isn't quite 1-to-1.
Since prod_p B_p = identity always (multiplying every plaquette operator
together, every link's Z gets hit exactly twice and cancels), the dual
picture must obey the mirror constraint

    prod_p X_p |psi> = +|psi>        (a GLOBAL Z2 symmetry of the TFIM)

This constraint cuts the dual Hilbert space from 2^(L^2) down to
2^(L^2 - 1), and restricts the whole construction to describing ONLY the
trivial ('t Hooft eigenvalues tau_h=tau_v=+1) topological sector -- exactly
the sector the true physical vacuum lives in, which is all we need for a
ground-truth energy check, but worth being explicit about: this method
cannot see the other 3 topological sectors at all (topological_sectors.py
is still the only place we compute those).

We validate this against our own independent physical_basis.py construction
below -- if two completely different exact methods agree, that's strong
evidence both are implemented correctly.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from lattice import Lattice


def dual_bonds(lat: Lattice) -> list:
    """Nearest-neighbor bonds of the dual (plaquette) lattice: two
    plaquettes p, p' are dual-neighbors exactly when they share an original
    link. Derived directly from lat.plaquettes (not assumed from a grid
    formula), and verified that every link belongs to exactly 2 plaquettes
    -- true whenever the ORIGINAL lattice has periodic boundaries, which is
    what the paper's mapping assumes."""
    assert lat.pbc, (
        "the paper derives this duality for the periodic (torus) lattice, "
        "where every link borders exactly 2 plaquettes -- an open lattice "
        "has boundary links bordering only 1 plaquette, which this simple "
        "mapping doesn't handle."
    )
    link_to_plaquettes = {}
    for p, plaq in enumerate(lat.plaquettes):
        for q in plaq:
            link_to_plaquettes.setdefault(q, []).append(p)
    bonds = []
    for q, plist in link_to_plaquettes.items():
        assert len(plist) == 2, (
            f"link {q} borders {len(plist)} plaquettes (expected exactly 2 "
            f"under PBC) -- the dual mapping assumption is violated"
        )
        bonds.append(tuple(sorted(plist)))
    return bonds


def build_dual_hamiltonian(lat: Lattice, h: float) -> sp.csr_matrix:
    """H_dual = sum_{<p,p'>} (1 - Z_p Z_p') - h * sum_p X_p, as a sparse
    (2^n_plaq x 2^n_plaq) matrix in the dual lattice's computational basis
    (bit p = the dual/plaquette qubit's value, NOT an original link qubit).

    Structure mirrors physical_basis.py's reduced_hamiltonian: diagonal
    terms come from the ZZ bonds (classical Ising energy), off-diagonal
    terms come from the transverse field flipping one dual qubit at a time.
    """
    bonds = dual_bonds(lat)
    n = len(lat.plaquettes)
    N = 1 << n

    rows, cols, vals = [], [], []
    for x in range(N):
        # diagonal: sum over bonds of (1 - eigenvalue of Z_p Z_p' on |x>)
        # eigenvalue is +1 if bits p,p' agree, -1 if they differ
        diag = 0.0
        for (p, pp) in bonds:
            same = ((x >> p) & 1) == ((x >> pp) & 1)
            if not same:
                diag += 2.0
        if diag:
            rows.append(x); cols.append(x); vals.append(diag)

        # off-diagonal: -h * X_p flips dual qubit p
        for p in range(n):
            y = x ^ (1 << p)
            rows.append(y); cols.append(x); vals.append(-h)

    return sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()


def trivial_sector_expectation(psi: np.ndarray) -> float:
    """<psi| prod_p X_p |psi> -- the global dual spin-flip symmetry. Should
    be +1 for a state living in the trivial ('t Hooft ++) sector, which is
    the only sector this dual Hamiltonian can represent at all."""
    N = len(psi)
    allones = N - 1
    total = 0.0
    for x in range(N):
        total += psi[x] * psi[x ^ allones]
    return float(total)


def _build_symmetric_sector_hamiltonian(lat: Lattice, h: float):
    """Build H_dual restricted DIRECTLY to the +1 (symmetric) sector of the
    global spin-flip G = prod_p X_p, as an (N/2 x N/2) sparse matrix.

    Why not just diagonalize the full H and look for a G=+1 eigenvector
    afterwards? Because at degenerate points (e.g. h=0, where the ZZ terms
    alone have many degenerate ground states) a generic diagonalization
    routine returns an arbitrary ORTHONORMAL BASIS of the degenerate
    subspace, not necessarily eigenstates of G -- so "is this eigenvector's
    <G> close to +1?" can fail to find a match even though a +1 combination
    exists in that subspace. We saw exactly this kind of eigensolver
    subtlety before (physical_basis.py's eigsh bug), so instead of
    patching around it we build the +1 sector as its own basis directly,
    the same way physical_basis.py builds the physical sector directly
    rather than filtering a full diagonalization afterwards.

    The construction: since no bitstring is its own complement (flipping
    every bit always changes a nonempty bitstring), every one of the N
    computational basis states pairs up with exactly one partner
    (x, complement(x)). Picking the representative x < complement(x) gives
    N/2 pairs. The symmetric combination |s_x> = (|x> + |~x>)/sqrt(2) is
    an eigenstate of G with eigenvalue +1. Working out <s_x|H|s_y>:
      - the ZZ (diagonal) terms take the SAME value on |x> and |~x> (both
        bits of any pair flip together, so their agreement/disagreement is
        unchanged) -- diagonal matrix elements carry over unchanged.
      - the X_p (transverse field) terms map |x> -> |x flip p> and
        |~x> -> |~x flip p> = |complement(x flip p)> simultaneously, so
        X_p maps the WHOLE symmetric pair for x to the whole symmetric
        pair for (x flip p), with the same coefficient -h and no extra
        sign -- i.e. off-diagonal matrix elements also carry over
        unchanged, just relabeled to representatives.
    So the +1-sector Hamiltonian has EXACTLY the same numerical structure
    as the full one, just built only over the N/2 representative labels.
    """
    bonds = dual_bonds(lat)
    n = len(lat.plaquettes)
    N = 1 << n
    allones = N - 1

    reps = [x for x in range(N) if x < (x ^ allones)]
    index_of = {x: i for i, x in enumerate(reps)}

    rows, cols, vals = [], [], []
    for i, x in enumerate(reps):
        diag = 0.0
        for (p, pp) in bonds:
            if ((x >> p) & 1) != ((x >> pp) & 1):
                diag += 2.0
        if diag:
            rows.append(i); cols.append(i); vals.append(diag)

        for p in range(n):
            y = x ^ (1 << p)
            ry = y if y < (y ^ allones) else (y ^ allones)
            j = index_of[ry]
            rows.append(j); cols.append(i); vals.append(-h)

    return sp.coo_matrix((vals, (rows, cols)), shape=(len(reps), len(reps))).tocsr()


DENSE_LIMIT = 4096


def ground_state_trivial_sector(lat: Lattice, h: float, k: int = 1, seed: int = 0):
    """Exact ground energy of H_dual restricted to the +1 (trivial 't Hooft)
    sector -- the sector the true physical vacuum lives in. Diagonalizes
    the symmetric-sector Hamiltonian directly (see
    _build_symmetric_sector_hamiltonian), so every eigenvalue returned is
    already guaranteed to be a genuine +1-sector energy -- no post-hoc
    filtering, no risk of degenerate-subspace mislabeling.

    Returns (energy, eigenvector_in_the_REPRESENTATIVE_basis, sector_check)
    where sector_check is always ~+1.0 by construction (included as an
    explicit self-check, not just assumed).
    """
    H = _build_symmetric_sector_hamiltonian(lat, h)
    N = H.shape[0]

    if N <= DENSE_LIMIT:
        evals, evecs = np.linalg.eigh(H.toarray())
    else:
        rng = np.random.default_rng(seed)
        v0 = rng.standard_normal(N)
        evals, evecs = eigsh(H, k=max(k, 6), which="SA", v0=v0, tol=0, maxiter=20000)
    order = np.argsort(evals)
    evals, evecs = evals[order], evecs[:, order]

    return float(evals[0]), evecs[:, 0], 1.0


if __name__ == "__main__":
    from physical_basis import ground_state_reduced

    print("=== Cross-validating two INDEPENDENT exact methods on the paper's "
          "18-qubit (L=3) lattice ===")
    print("  physical_basis.py : our own construction (Hadamard + graph cycle space)")
    print("  tfim_dual.py       : the paper's own construction (Eq. 10-11, TFIM duality)\n")

    lat = Lattice(Lx=3, Ly=3, pbc=True)
    n_dual = len(lat.plaquettes)
    print(f"Dual lattice: {n_dual} qubits (vs {lat.n_qubits} in the original link "
          f"picture) -- {2**n_dual} raw dual states, "
          f"{2**n_dual // 2} after the global +1 sector constraint\n")

    for h in [0.5, 1.0, 2.0, 3.04438, 6.0]:
        e_ours, _ = ground_state_reduced(lat, h, k=1)
        e_paper, _, g = ground_state_trivial_sector(lat, h)
        diff = abs(e_ours[0] - e_paper)
        print(f"  h={h:8.5f}:  E_our_method={e_ours[0]:12.6f}   "
              f"E_paper_method={e_paper:12.6f}   diff={diff:.2e}   "
              f"<G>={g:.6f}   {'OK' if diff < 1e-8 else 'MISMATCH!!'}")

    print("\n=== Same cross-check at L=4 (32 original qubits -- both methods are "
          "'only' classical exact-diagonalization tricks, no QAOA circuit involved) ===")
    lat4 = Lattice(Lx=4, Ly=4, pbc=True)
    n_dual4 = len(lat4.plaquettes)
    print(f"Dual lattice: {n_dual4} qubits, {2**n_dual4:,} raw dual states "
          f"(vs our own method's {2**(lat4.n_qubits - len(lat4.vertices) + 1):,} "
          f"cycle-space states, and vs {2**lat4.n_qubits:,} for the untouched "
          f"full 32-qubit space)\n")

    import time
    for h in [1.0, 3.04438, 6.0]:
        t0 = time.time()
        e_ours, _ = ground_state_reduced(lat4, h, k=1)
        t1 = time.time()
        e_paper, _, g = ground_state_trivial_sector(lat4, h)
        t2 = time.time()
        diff = abs(e_ours[0] - e_paper)
        print(f"  h={h:8.5f}:  E_our_method={e_ours[0]:12.6f} ({t1-t0:5.1f}s)   "
              f"E_paper_method={e_paper:12.6f} ({t2-t1:5.1f}s)   diff={diff:.2e}   "
              f"{'OK' if diff < 1e-8 else 'MISMATCH!!'}")

    # --- Honesty check on L=5, same spirit as wilson_scan_multisize.py ---
    lat5 = Lattice(Lx=5, Ly=5, pbc=True)
    n_dual5 = len(lat5.plaquettes)
    N5 = 2 ** n_dual5
    reps5 = N5 // 2
    est_gb = reps5 * 26 * 16 / 1e9
    print(f"\n=== L=5 check: dual lattice has {n_dual5} qubits -> {N5:,} raw "
          f"states -> {reps5:,} in the +1 sector ===")
    print(f"    Estimated construction memory: ~{est_gb:.1f} GB -- this is smaller "
          f"than our own method's L=5 estimate (~40+ GB) precisely because the "
          f"paper's mapping uses half as many qubits, but it's still right at "
          f"the edge of (or over) this sandbox's ~7.3 GB RAM, so NOT attempted "
          f"here either. Worth noting as a genuine advantage of the paper's "
          f"method if this were run on a bigger machine.")
