"""
test_z2_lgt.py
================

"""

import numpy as np

from lattice import Lattice
from hamiltonian import full_hamiltonian, electric_term, magnetic_term, star_operators
from exact_diag import ground_state, check_gauge_sector
from qaoa_circuit import build_qaoa_circuit
from optimize import run_qaoa


def test_lattice_geometry():
    lat = Lattice(Lx=2, Ly=2)
    assert lat.n_qubits == 12                 # 2*L*(L+1) = 2*2*3
    assert len(lat.plaquettes) == 4            # Lx * Ly
    assert len(lat.vertices) == 9              # (Lx+1)*(Ly+1)
    degrees = [len(lat.stars[v]) for v in lat.vertices]
    assert sorted(degrees) == [2, 2, 2, 2, 3, 3, 3, 3, 4]  # 4 corners, 4 edges, 1 center


def test_hamiltonian_hermitian():
    lat = Lattice(Lx=2, Ly=2)
    H = full_hamiltonian(lat, h=1.7)
    assert np.allclose(H.coeffs.imag, 0), "H must be real/Hermitian (Pauli coeffs)"


def test_hamiltonian_term_counts():
    lat = Lattice(Lx=2, Ly=2)
    He = electric_term(lat)
    Hb = magnetic_term(lat)
    assert len(He.paulis) == lat.n_qubits + 1   # n single-qubit X terms + identity
    assert len(Hb.paulis) == len(lat.plaquettes)


def test_gauge_invariance_commutator():
    """Every star operator A_v must commute exactly with H (Gauss law is
    conserved -> a correct LGT Hamiltonian, not an accidental one)."""
    lat = Lattice(Lx=2, Ly=2)
    H = full_hamiltonian(lat, h=1.3).to_matrix(sparse=True)
    stars = star_operators(lat)
    for v, Av in stars.items():
        Av_m = Av.to_matrix(sparse=True)
        comm = Av_m @ H - H @ Av_m
        max_val = np.abs(comm.toarray()).max() if comm.nnz else 0.0
        assert max_val < 1e-9, f"[A_{v}, H] != 0"


def test_h0_ground_state_is_trivial():
    """At h=0, H = H_E only, whose exact ground state is |+>^n with E=0."""
    lat = Lattice(Lx=2, Ly=2)
    evals, evecs = ground_state(lat, h=0.0, k=1)
    assert abs(evals[0]) < 1e-8
    plus = np.ones(2 ** lat.n_qubits) / np.sqrt(2 ** lat.n_qubits)
    fidelity = np.abs(np.vdot(plus, evecs[:, 0])) ** 2
    assert fidelity > 1 - 1e-6


def test_exact_ground_state_is_physical():
    """The true ground state (any h) must satisfy <A_v> = +1 for all v --
    i.e. live in the zero-gauge-charge physical sector."""
    lat = Lattice(Lx=2, Ly=2)
    for h in [0.5, 1.0, 3.0, 6.0]:
        evals, evecs = ground_state(lat, h, k=1)
        sector = check_gauge_sector(lat, evecs[:, 0])
        for v, val in sector.items():
            assert abs(val - 1.0) < 1e-6, f"h={h}, vertex {v}: <A_v>={val}"


def test_variational_principle():
    """QAOA energy must never go below the true ground energy."""
    lat = Lattice(Lx=2, Ly=2)
    h = 1.0
    evals, _ = ground_state(lat, h, k=1)
    result = run_qaoa(lat, h, p=2, n_restarts=3, seed=1, maxiter=150, verbose=False)
    assert result["energy"] >= evals[0] - 1e-6


def test_pbc_lattice_matches_paper_18_qubits():
    """The paper's headline lattice: L=3, periodic -> 18 qubits, 9
    plaquettes, 9 vertices, every vertex degree 4 (no boundary)."""
    lat = Lattice(Lx=3, Ly=3, pbc=True)
    assert lat.n_qubits == 18
    assert len(lat.plaquettes) == 9
    assert len(lat.vertices) == 9
    degrees = {len(lat.stars[v]) for v in lat.vertices}
    assert degrees == {4}, "periodic lattice must have NO boundary vertices"


def test_pbc_rejects_too_small_lattices():
    """L=1 -> self-loop, L=2 -> duplicate edges our frozenset index can't
    disambiguate. Both must be explicitly rejected, not silently wrong."""
    for L in [1, 2]:
        try:
            Lattice(Lx=L, Ly=L, pbc=True)
            raise AssertionError(f"pbc L={L} should have raised ValueError")
        except ValueError:
            pass


def test_pbc_gauge_invariance():
    """Same Gauss-law check as the open lattice, but on the torus -- and
    using the sparse .data max (NOT .toarray(), which OOMs at 18 qubits)."""
    lat = Lattice(Lx=3, Ly=3, pbc=True)
    H = full_hamiltonian(lat, h=1.3).to_matrix(sparse=True)
    stars = star_operators(lat)
    for v, Av in stars.items():
        Av_m = Av.to_matrix(sparse=True)
        comm = Av_m @ H - H @ Av_m
        max_val = np.abs(comm.data).max() if comm.nnz else 0.0
        assert max_val < 1e-9, f"[A_{v}, H] != 0 on periodic lattice"


def test_qaoa_circuit_shapes():
    lat = Lattice(Lx=2, Ly=2)
    p = 2
    qc, betas, gammas = build_qaoa_circuit(lat, p)
    assert qc.num_qubits == lat.n_qubits
    assert len(betas) == p
    assert len(gammas) == p


if __name__ == "__main__":
    tests = [
        test_lattice_geometry,
        test_hamiltonian_hermitian,
        test_hamiltonian_term_counts,
        test_gauge_invariance_commutator,
        test_h0_ground_state_is_trivial,
        test_exact_ground_state_is_physical,
        test_pbc_lattice_matches_paper_18_qubits,
        test_pbc_rejects_too_small_lattices,
        test_pbc_gauge_invariance,
        test_qaoa_circuit_shapes,
        test_variational_principle,  # slowest, run last
    ]
    for t in tests:
        print(f"Running {t.__name__} ...", end=" ")
        t()
        print("PASSED")
    print("\nAll tests passed.")
