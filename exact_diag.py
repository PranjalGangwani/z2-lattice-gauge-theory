"""
exact_diag.py

"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from lattice import Lattice
from hamiltonian import full_hamiltonian, star_operators


def ground_state(lat: Lattice, h: float, k: int = 1):
    """Return (energies, eigenvectors) for the k lowest eigenstates of H(h).

    """
    H = full_hamiltonian(lat, h).to_matrix(sparse=True)
    H = sp.csr_matrix(H)
    # eigsh needs k < N-1; 'SA' = smallest algebraic eigenvalues
    evals, evecs = eigsh(H, k=k, which="SA")
    order = np.argsort(evals)
    return evals[order], evecs[:, order]


def check_gauge_sector(lat: Lattice, psi: np.ndarray) -> dict:
    """Compute <psi| A_v |psi> for every vertex v.

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
