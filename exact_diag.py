"""
exact_diag.py

with scipy's sparse Lanczos solver (eigsh). This gives us:

  * the exact ground energy E0(h) -- to check QAOA's variational energy
    against (QAOA energy must always be >= E0, by the variational principle)
  * the exact ground state |psi0(h)> -- to compute a *fidelity* against the
    QAOA output state, |<psi0|psi_QAOA>|^2
  * a check that the true ground state lives in the physical (gauge-
    invariant) sector: <psi0| A_v |psi0> = +1 for every vertex v.

Note on scale: this brute-force approach is only feasible because our test
lattice is tiny. The whole point of the paper is that QAOA sidesteps the
exponential cost of ED for larger lattices -- ED here is purely a debugging
/ verification tool for the small system, not part of the "real" pipeline.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from lattice import Lattice
from hamiltonian import full_hamiltonian, star_operators


def ground_state(lat: Lattice, h: float, k: int = 1):
    """Return (energies, eigenvectors) for the k lowest eigenstates of H(h).

    Uses scipy.sparse.linalg.eigsh (Lanczos), appropriate since H is
    Hermitian and sparse (each Pauli string is a sparse matrix).
    """
    H = full_hamiltonian(lat, h).to_matrix(sparse=True)
    H = sp.csr_matrix(H)
    # eigsh needs k < N-1; 'SA' = smallest algebraic eigenvalues
    evals, evecs = eigsh(H, k=k, which="SA")
    order = np.argsort(evals)
    return evals[order], evecs[:, order]


def check_gauge_sector(lat: Lattice, psi: np.ndarray) -> dict:
    """Compute <psi| A_v |psi> for every vertex v.

    For the true (matter-free) ground state, every value should be +1.0
    (up to numerical precision) -- this confirms the ED ground state lives
    in the physical, zero-gauge-charge sector, matching what the QAOA
    circuit will also target.
    """
    stars = star_operators(lat)
    result = {}
    for v, Av in stars.items():
        Av_m = Av.to_matrix(sparse=True)
        val = np.real(psi.conj() @ (Av_m @ psi))
        result[v] = val
    return result


def scan_h(lat: Lattice, h_values, k: int = 1):
    """Scan the ground energy over a list of h values. Returns a list of
    (h, E0) tuples."""
    out = []
    for h in h_values:
        evals, evecs = ground_state(lat, h, k=k)
        out.append((h, evals[0], evecs[:, 0]))
    return out


if __name__ == "__main__":
    lat = Lattice(Lx=2, Ly=2)
    print(lat.summary())

    for h in [0.0, 1.0, 3.04438, 6.0]:
        evals, evecs = ground_state(lat, h, k=1)
        psi0 = evecs[:, 0]
        print(f"\nh = {h:.5f}:  E0 = {evals[0]:.6f}")

        sector = check_gauge_sector(lat, psi0)
        vals = np.array(list(sector.values()))
        print(f"  <A_v> range: [{vals.min():.6f}, {vals.max():.6f}] "
              f"(should all be ~+1.0 for the physical vacuum sector)")
