"""
topo_entropy.py
=================

Step 13: topological entanglement entropy via the Kitaev-Preskill
tripartition, S_topo = S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC.

Getting an ambient state to trace over
-----------------------------------------
Entanglement entropy needs an actual qubit-by-qubit statevector (so we can
trace out the qubits outside our region of interest) -- but physical_basis.py
only gives us a compact 2^k-dimensional vector in the Hadamard-DUAL frame
(k = cyclomatic number, e.g. 10 for our 18-qubit lattice, vs the ambient
2^18). We need to invert that transform.

Recall H'(h) = U H(h) U with U = H^{\\otimes n} (global Hadamard, self-
inverse). So the AMBIENT exact ground state is:

    |psi_ambient> = U |psi_reduced_embedded>

where |psi_reduced_embedded> is the reduced-frame ground state placed into
the full 2^n space (zero everywhere except at the "physical" cycle-space
basis labels). Applying U = H^{\\otimes n} to a vector is exactly a
(normalized) Fast Walsh-Hadamard Transform (FWHT) -- an O(n * 2^n)
algorithm, not the naive O(4^n) of multiplying by a dense matrix.

We validate this reconstruction against our OWN trusted brute-force ED
(exact_diag.py) on the small 2x2 lattice before trusting it at 18 qubits.
"""

import numpy as np
from qiskit.quantum_info import Statevector, partial_trace, entropy

from lattice import Lattice
from physical_basis import enumerate_physical_states, ground_state_reduced


def fwht(a: np.ndarray) -> np.ndarray:
    """In-place-style (returns a copy) Fast Walsh-Hadamard Transform,
    UNnormalized: out[x] = sum_y (-1)^(popcount(x & y) mod 2) * a[y]."""
    a = a.astype(complex).copy()
    n = len(a)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            x = a[i:i + h].copy()
            y = a[i + h:i + 2 * h].copy()
            a[i:i + h] = x + y
            a[i + h:i + 2 * h] = x - y
        h *= 2
    return a


def reconstruct_ambient_state(lat: Lattice, h: float) -> np.ndarray:
    """Exact ground state as a full 2^n_qubits ambient statevector, built
    from the compact reduced (cycle-space) ground state via FWHT."""
    states, index_of = enumerate_physical_states(lat)
    evals, evecs = ground_state_reduced(lat, h, k=1, states=states, index_of=index_of)
    psi_reduced = evecs[:, 0]

    N = 2 ** lat.n_qubits
    embedded = np.zeros(N, dtype=complex)
    for x, amp in zip(states, psi_reduced):
        embedded[x] = amp

    ambient = fwht(embedded) / np.sqrt(N)
    # fix global phase/sign convention (physically irrelevant, just tidy)
    if ambient[np.argmax(np.abs(ambient))].real < 0:
        ambient = -ambient
    norm = np.linalg.norm(ambient)
    assert abs(norm - 1.0) < 1e-8, f"reconstructed state not normalized: {norm}"
    return ambient


def kitaev_preskill_entropy(psi: np.ndarray, n_qubits: int, region_a, region_b, region_c):
    """S_topo = S_A+S_B+S_C-S_AB-S_BC-S_AC+S_ABC, all von Neumann
    entropies in natural-log units (nats)."""
    sv = Statevector(psi, dims=(2,) * n_qubits)
    all_qubits = set(range(n_qubits))

    def S(region):
        region = list(region)
        trace_out = list(all_qubits - set(region))
        rho = partial_trace(sv, trace_out)
        return entropy(rho, base=np.e)

    A, B, C = list(region_a), list(region_b), list(region_c)
    AB, BC, AC = A + B, B + C, A + C
    ABC = A + B + C

    S_A, S_B, S_C = S(A), S(B), S(C)
    S_AB, S_BC, S_AC = S(AB), S(BC), S(AC)
    S_ABC = S(ABC)

    S_topo = S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC
    return {
        "S_A": S_A, "S_B": S_B, "S_C": S_C,
        "S_AB": S_AB, "S_BC": S_BC, "S_AC": S_AC,
        "S_ABC": S_ABC, "S_topo": S_topo,
    }


def choose_kp_regions(lat: Lattice):
    """A concrete Kitaev-Preskill tripartition on our L=3 lattice: 3
    regions of 2 qubits each (6 qubits total), matching the paper's
    reported region sizes.

    HONESTY NOTE: the paper reports "6 qubits, 2 per region, cutting 5
    vertices" but we do not have its exact figure -- this is OUR choice
    of a small, mutually-adjacent, roughly-symmetric tripartition
    satisfying those same counts, not a verified reproduction of their
    specific geometry. The qualitative physics (S_topo far from 0 deep in
    the deconfined phase, S_topo ~ 0 deep in the confined phase) is a
    property of the topological phase itself and shouldn't depend on the
    exact region shape, as long as it's a genuine closed tripartition
    around a shared neighborhood -- but the precise numeric curve may
    differ from the paper's own choice of region.
    """
    # take the plaquette at cell (0,0) and its 3 neighbors sharing vertex (1,1)
    # (the "central" vertex where 4 plaquettes meet); use 2 links from each
    # of 3 of those plaquettes, chosen to be mutually adjacent and non-
    # overlapping.
    p00 = lat.plaquettes[0 * lat.Lx + 0]  # bottom,right,top,left qubit ids
    p01 = lat.plaquettes[0 * lat.Lx + 1]
    p10 = lat.plaquettes[1 * lat.Lx + 0]

    # each plaquette tuple is (bottom, right, top, left); pick 2 qubits from
    # each of 3 different plaquettes, avoiding double-using any qubit
    region_a = [p00[0], p00[3]]   # bottom, left of plaquette (0,0)
    region_b = [p01[3], p01[2]]   # left, top of plaquette (0,1) (left link is shared w/ p00 -> pick others)
    region_c = [p10[0], p10[1]]   # bottom, right of plaquette (1,0)

    all_q = region_a + region_b + region_c
    assert len(set(all_q)) == 6, f"region qubits not all distinct: {all_q}"
    return region_a, region_b, region_c


if __name__ == "__main__":
    print("=== Validating FWHT reconstruction against trusted brute-force ED (2x2) ===")
    from exact_diag import ground_state as full_ground_state
    lat_small = Lattice(Lx=2, Ly=2)
    for h in [1.0, 3.04438, 6.0]:
        evals_full, evecs_full = full_ground_state(lat_small, h, k=1)
        psi_reconstructed = reconstruct_ambient_state(lat_small, h)
        psi_full = evecs_full[:, 0]
        if psi_full[np.argmax(np.abs(psi_full))].real < 0:
            psi_full = -psi_full
        fidelity = abs(np.vdot(psi_full, psi_reconstructed)) ** 2
        print(f"  h={h:8.5f}: |<psi_full|psi_reconstructed>|^2 = {fidelity:.10f}")

    print("\n=== Topological entanglement entropy on the 18-qubit lattice ===")
    lat = Lattice(Lx=3, Ly=3, pbc=True)
    region_a, region_b, region_c = choose_kp_regions(lat)
    print(f"Regions: A={region_a}  B={region_b}  C={region_c}")

    results = []
    for h in [0.5, 1.0, 1.5, 2.0, 3.04438, 4.0, 6.0, 9.0]:
        psi = reconstruct_ambient_state(lat, h)
        r = kitaev_preskill_entropy(psi, lat.n_qubits, region_a, region_b, region_c)
        results.append({"h": h, **r})
        print(f"  h={h:6.3f}:  S_topo = {r['S_topo']:+.4f} nats   "
              f"(S_A={r['S_A']:.3f}, S_ABC={r['S_ABC']:.3f})")

    import json
    with open("topo_entropy_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved topo_entropy_results.json")
