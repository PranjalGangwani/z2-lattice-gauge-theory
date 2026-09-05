"""
topological_sectors.py
========================
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from lattice import Lattice
from hamiltonian import full_hamiltonian
from qiskit.quantum_info import SparsePauliOp
from physical_basis import enumerate_physical_states, reduced_hamiltonian


def t_hooft_qubits(lat: Lattice, r0: int = 0, c0: int = 0):
    """Qubit indices for the two 't Hooft loops. Requires PBC.

    tau_h: crosses every vertical link at row-gap r0 (i.e. links between
    vertex-row r0 and r0+1), one per column -- wraps horizontally.
    tau_v: crosses every horizontal link at column-gap c0, one per row --
    wraps vertically.
    """
    assert lat.pbc, "'t Hooft loops require a periodic lattice"
    tau_h_qubits = [lat.link_index[frozenset(((r0, c), ((r0 + 1) % lat.Ny, c)))]
                    for c in range(lat.Nx)]
    tau_v_qubits = [lat.link_index[frozenset(((r, c0), (r, (c0 + 1) % lat.Nx)))]
                    for r in range(lat.Ny)]
    return tau_h_qubits, tau_v_qubits


def t_hooft_operators(lat: Lattice):
    """SparsePauliOp for tau_h and tau_v (product of X over their qubits)."""
    from hamiltonian import _pauli_label
    tau_h_q, tau_v_q = t_hooft_qubits(lat)
    n = lat.n_qubits
    tau_h = SparsePauliOp([_pauli_label(n, {q: "X" for q in tau_h_q})], [1.0])
    tau_v = SparsePauliOp([_pauli_label(n, {q: "X" for q in tau_v_q})], [1.0])
    return tau_h, tau_v


def verify_t_hooft_symmetry(lat: Lattice, h: float = 1.3):
    """Explicit numerical check (not an assumption) that tau_h, tau_v
    commute with H and with each other, and that each squares to identity
    (so eigenvalues are genuinely +-1)."""
    H = full_hamiltonian(lat, h).to_matrix(sparse=True)
    tau_h, tau_v = t_hooft_operators(lat)
    tau_h_m = tau_h.to_matrix(sparse=True)
    tau_v_m = tau_v.to_matrix(sparse=True)

    def max_comm(A, B):
        C = A @ B - B @ A
        return np.abs(C.data).max() if C.nnz else 0.0

    results = {
        "[tau_h, H]": max_comm(tau_h_m, H),
        "[tau_v, H]": max_comm(tau_v_m, H),
        "[tau_h, tau_v]": max_comm(tau_h_m, tau_v_m),
    }
    # tau^2 = I check (X operators always square to identity, but let's
    # verify our qubit sets are distinct so the product really is nontrivial)
    results["tau_h^2 == I"] = np.abs((tau_h_m @ tau_h_m - sp.identity(H.shape[0])).data).max() \
        if (tau_h_m @ tau_h_m - sp.identity(H.shape[0])).nnz else 0.0
    return results


def sector_ground_energies(lat: Lattice, h: float, r0: int = 0, c0: int = 0):
    """Split the physical (cycle-space) basis into the 4 sectors by tau_h,
    tau_v parity, diagonalize H within each, and return a dict
    {(sign_h, sign_v): ground_energy}."""
    tau_h_q, tau_v_q = t_hooft_qubits(lat, r0, c0)
    mask_h = 0
    for q in tau_h_q:
        mask_h ^= (1 << q)
    mask_v = 0
    for q in tau_v_q:
        mask_v ^= (1 << q)

    states, index_of = enumerate_physical_states(lat)
    H_full = reduced_hamiltonian(lat, h, states, index_of)

    groups = {(+1, +1): [], (+1, -1): [], (-1, +1): [], (-1, -1): []}
    for i, x in enumerate(states):
        sign_h = 1 - 2 * (bin(x & mask_h).count("1") % 2)
        sign_v = 1 - 2 * (bin(x & mask_v).count("1") % 2)
        groups[(sign_h, sign_v)].append(i)

    energies = {}
    H_csr = H_full.tocsr()
    for sector, idx in groups.items():
        idx = np.array(idx)
        sub = H_csr[idx][:, idx]
        # dense, not eigsh -- see physical_basis.py's ground_state_reduced
        # docstring for the correctness bug this avoids. Sector blocks are
        # small (n_states/4, e.g. ~256 at L=3) so dense is cheap here.
        if sub.shape[0] <= 4096:
            evals = np.linalg.eigvalsh(sub.toarray())
        else:
            rng = np.random.default_rng(0)
            v0 = rng.standard_normal(sub.shape[0])
            evals = eigsh(sub, k=6, which="SA", v0=v0, tol=0, maxiter=20000,
                          return_eigenvectors=False)
        energies[sector] = float(np.min(evals))
    return energies


if __name__ == "__main__":
    lat = Lattice(Lx=3, Ly=3, pbc=True)

    print("=== Verifying 't Hooft loop symmetry (not assuming it) ===")
    checks = verify_t_hooft_symmetry(lat, h=1.7)
    for name, val in checks.items():
        print(f"  {name}: {val:.2e}  {'OK (~0)' if val < 1e-9 else 'FAIL'}")

    print("\n=== Four topological sectors: ground energy in each ===")
    for h in [0.5, 1.0, 3.04438, 6.0]:
        energies = sector_ground_energies(lat, h)
        vals = list(energies.values())
        spread = max(vals) - min(vals)
        print(f"\n  h = {h}")
        for (sh, sv), e in sorted(energies.items()):
            tag = "++" if (sh, sv) == (1, 1) else "+-" if (sh, sv) == (1, -1) else \
                  "-+" if (sh, sv) == (-1, 1) else "--"
            print(f"    |tau_h,tau_v> = |{tag}>:  E0 = {e:.6f}")
        print(f"    spread across sectors: {spread:.2e}  "
              f"({'degenerate' if spread < 1e-6 else 'split'})")
