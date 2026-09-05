"""
wilson_scan_multisize.py
==========================
"""
import numpy as np

from lattice import Lattice
from physical_basis import enumerate_physical_states, ground_state_reduced


def rectangular_loop_mask(lat: Lattice, r0: int, c0: int, a: int, b: int):
    assert a < lat.Ly and b < lat.Lx, (
        f"loop {a}x{b} would wrap around an L={lat.Lx} torus -- "
        f"that's a 't Hooft loop, not a contractible Wilson loop"
    )
    mask = 0
    for r in range(r0, r0 + a):
        for c in range(c0, c0 + b):
            cell = (r % lat.Ly) * lat.Lx + (c % lat.Lx)
            for q in lat.plaquettes[cell]:
                mask ^= (1 << q)
    return mask, a * b, 2 * (a + b)


def wilson_expectation_reduced(mask: int, states: list, index_of: dict, psi: np.ndarray) -> float:
    total = 0.0
    for j, x in enumerate(states):
        i = index_of[x ^ mask]
        total += psi[i] * psi[j]
    return float(total)


def available_loop_shapes(L: int):
    shapes = []
    for a in range(1, L):
        for b in range(a, L):
            shapes.append((a, b))
    return shapes


def scan_lattice_size(L: int, h_values, verbose=True):
    lat = Lattice(Lx=L, Ly=L, pbc=True)
    states, index_of = enumerate_physical_states(lat)
    shapes = available_loop_shapes(L)

    rows = []
    for h in h_values:
        evals, evecs = ground_state_reduced(lat, h, k=1, states=states, index_of=index_of)
        psi = evecs[:, 0]
        for (a, b) in shapes:
            mask, area, perim = rectangular_loop_mask(lat, 0, 0, a, b)
            W = wilson_expectation_reduced(mask, states, index_of, psi)
            rows.append({"L": L, "h": h, "shape": (a, b), "area": area,
                         "perimeter": perim, "W": W})
        if verbose:
            shown = ", ".join(f"{a}x{b}:W={r['W']:.4f}" for (a, b), r in
                               zip(shapes, rows[-len(shapes):]))
            print(f"  L={L} h={h:6.3f}: {shown}")
    return rows


def fit_area_perimeter_law(rows_for_one_h):
    A = np.array([r["area"] for r in rows_for_one_h], dtype=float)
    P = np.array([r["perimeter"] for r in rows_for_one_h], dtype=float)
    W = np.array([r["W"] for r in rows_for_one_h], dtype=float)
    y = np.log(np.abs(W) + 1e-15)
    M = np.stack([-A, -P], axis=1)
    (chi, delta), *_ = np.linalg.lstsq(M, y, rcond=None)
    return chi, delta


if __name__ == "__main__":
    h_values = [0.5, 1.0, 1.5, 2.0, 3.04438, 4.0, 6.0, 9.0]

    print("=== L=3 (18 qubits): available contractible loop shapes ===")
    print(available_loop_shapes(3), "\n")
    rows_L3 = scan_lattice_size(3, h_values)

    print("\n=== L=4 (32 qubits): reduced ED only feasible route here "
          "(full QAOA statevector would need 68.7 GB) ===")
    print(available_loop_shapes(4), "\n")
    rows_L4 = scan_lattice_size(4, h_values)

    print("\n=== Area/perimeter law fit: chi(h), delta(h), pooling L=3 and L=4 loops ===")
    all_rows = rows_L3 + rows_L4
    for h in h_values:
        rows_h = [r for r in all_rows if r["h"] == h]
        chi, delta = fit_area_perimeter_law(rows_h)
        print(f"  h={h:6.3f}: chi={chi:8.4f}  delta={delta:8.4f}  "
              f"({'confined: area-law dominates' if chi > 0.05 else 'deconfined-ish: perimeter-law'})")

    import json
    with open("wilson_multisize_results.json", "w") as f:
        json.dump({"L3": rows_L3, "L4": rows_L4}, f, indent=2)
    print("\nSaved wilson_multisize_results.json")

    # --- Honesty check on L=5 before even attempting it ---
    lat5 = Lattice(Lx=5, Ly=5, pbc=True)
    k5 = lat5.n_qubits - len(lat5.vertices) + 1
    print(f"\n=== L=5 check: reduced subspace would be 2^{k5} = {2**k5:,} dimensional ===")
    print(f"    Estimated sparse Hamiltonian memory (~26 nonzeros/row): "
          f"{2**k5 * 26 * 24 / 1e9:.1f} GB -- not attempted in this sandbox (~7 GB RAM).")
