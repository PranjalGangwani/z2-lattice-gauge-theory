"""
observables.py
================

"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector

from lattice import Lattice
from hamiltonian import _pauli_label
from exact_diag import ground_state
from optimize import run_qaoa


def boundary_loop(lat: Lattice) -> list[int]:
    """Qubit indices of the links forming the outer boundary of the whole
    LxL lattice, i.e. the largest Wilson loop available on this lattice."""
    top = [lat.link_index[frozenset(((0, c), (0, c + 1)))] for c in range(lat.Lx)]
    bottom = [lat.link_index[frozenset(((lat.Ly, c), (lat.Ly, c + 1)))] for c in range(lat.Lx)]
    left = [lat.link_index[frozenset(((r, 0), (r + 1, 0)))] for r in range(lat.Ly)]
    right = [lat.link_index[frozenset(((r, lat.Lx), (r + 1, lat.Lx)))] for r in range(lat.Ly)]
    return top + bottom + left + right


def wilson_loop_op(lat: Lattice, loop_qubits: list[int]) -> SparsePauliOp:
    label = _pauli_label(lat.n_qubits, {q: "Z" for q in loop_qubits})
    return SparsePauliOp([label], [1.0])


def expectation(op: SparsePauliOp, psi) -> float:
    """psi may be a qiskit Statevector or a raw numpy array."""
    if not isinstance(psi, Statevector):
        psi = Statevector(psi)
    return float(np.real(psi.expectation_value(op)))


def scan(lat: Lattice, h_values, p: int, n_restarts: int = 5, seed: int = 0):
    """For each h: exact ground energy/state, QAOA energy/state, and both
    Wilson loops (single plaquette, full boundary) evaluated on each."""
    plaq_qubits = list(lat.plaquettes[0])       # smallest loop: area=1
    bnd_qubits = boundary_loop(lat)             # largest loop: area=Lx*Ly

    W_plaq_op = wilson_loop_op(lat, plaq_qubits)
    W_bnd_op = wilson_loop_op(lat, bnd_qubits)

    rows = []
    for h in h_values:
        evals, evecs = ground_state(lat, h, k=1)
        psi_exact = evecs[:, 0]
        E_exact = evals[0]

        res = run_qaoa(lat, h, p, n_restarts=n_restarts, seed=seed, verbose=False)
        psi_qaoa = res["statevector"]
        E_qaoa = res["energy"]

        fidelity = np.abs(np.vdot(psi_exact, psi_qaoa.data)) ** 2

        row = {
            "h": h,
            "E_exact": E_exact,
            "E_qaoa": E_qaoa,
            "fidelity": fidelity,
            "W_plaq_exact": expectation(W_plaq_op, psi_exact),
            "W_plaq_qaoa": expectation(W_plaq_op, psi_qaoa),
            "W_bnd_exact": expectation(W_bnd_op, psi_exact),
            "W_bnd_qaoa": expectation(W_bnd_op, psi_qaoa),
        }
        rows.append(row)
        print(f"h={h:5.2f}  E_exact={E_exact:9.4f}  E_qaoa={E_qaoa:9.4f}  "
              f"fid={fidelity:.5f}  W_plaq={row['W_plaq_exact']:.4f}  "
              f"W_bnd={row['W_bnd_exact']:.4f}")
    return rows


if __name__ == "__main__":
    lat = Lattice(Lx=2, Ly=2)
    h_values = [0.2, 0.6, 1.0, 1.5, 2.0, 3.04438, 4.0, 6.0, 9.0]
    print(f"Scanning h across confinement (h<h_c~3.044) / deconfinement "
          f"(h>h_c) crossover on the {lat.n_qubits}-qubit 2x2 lattice, "
          f"QAOA p=3:\n")
    rows = scan(lat, h_values, p=3, n_restarts=5, seed=0)

    import json
    with open("scan_results.json", "w") as fo:
        json.dump(rows, fo, indent=2)
    print("\nSaved raw results to scan_results.json")
