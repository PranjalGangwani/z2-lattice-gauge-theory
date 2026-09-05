"""
hamiltonian.py

"""

from qiskit.quantum_info import SparsePauliOp
import numpy as np

from lattice import Lattice

def _pauli_label(n_qubits: int, ops: dict) -> str:
    chars = ["I"] * n_qubits
    for q, p in ops.items():
        chars[q] = p
    # reverse because qiskit labels are little-endian (qubit 0 on the right)
    return "".join(reversed(chars))


def electric_term(lat: Lattice) -> SparsePauliOp:
    """H_E = sum_l (1 - X_l), a sum of n_qubits Pauli strings (+ identity)."""
    n = lat.n_qubits
    paulis, coeffs = [], []
    # constant identity piece: + n (from the "1" in each (1 - X_l))
    paulis.append(_pauli_label(n, {}))
    coeffs.append(float(n))
    # -X_l for every link/qubit
    for q in range(n):
        paulis.append(_pauli_label(n, {q: "X"}))
        coeffs.append(-1.0)
    return SparsePauliOp(paulis, coeffs).simplify()


def magnetic_term(lat: Lattice) -> SparsePauliOp:
    """H_B = -sum_p Z_{p1} Z_{p2} Z_{p3} Z_{p4}, one 4-body term per plaquette."""
    n = lat.n_qubits
    paulis, coeffs = [], []
    for plaq in lat.plaquettes:
        ops = {q: "Z" for q in plaq}
        paulis.append(_pauli_label(n, ops))
        coeffs.append(-1.0)
    return SparsePauliOp(paulis, coeffs).simplify()


def full_hamiltonian(lat: Lattice, h: float) -> SparsePauliOp:
    """H(h) = H_E + h * H_B"""
    He = electric_term(lat)
    Hb = magnetic_term(lat)
    return (He + h * Hb).simplify()


def star_operators(lat: Lattice) -> dict:
    n = lat.n_qubits
    ops = {}
    for v, qs in lat.stars.items():
        label = _pauli_label(n, {q: "X" for q in qs})
        ops[v] = SparsePauliOp([label], [1.0])
    return ops


def check_commutation(lat: Lattice, h: float = 1.0) -> None:
    H = full_hamiltonian(lat, h).to_matrix(sparse=True)
    stars = star_operators(lat)

    print("Checking [A_v, H] = 0 for all vertices v ...")
    all_ok = True
    for v, Av in stars.items():
        Av_m = Av.to_matrix(sparse=True)
        comm = Av_m @ H - H @ Av_m
        max_val = np.abs(comm.data).max() if comm.nnz else 0.0
        ok = max_val < 1e-9
        all_ok &= ok
        print(f"  vertex {v}: max|[A_v,H]| = {max_val:.2e}  {'OK' if ok else 'FAIL'}")
    print("All commutators vanish:" , all_ok)
