"""
qaoa_18qubit_demo.py
=====================
"""

import time
import numpy as np

from lattice import Lattice
from optimize import run_qaoa
from physical_basis import ground_state_reduced
from hamiltonian import star_operators


def qaoa_gauge_invariance(lat: Lattice, psi) -> np.ndarray:
    stars = star_operators(lat)
    vals = []
    for v, Av in stars.items():
        Av_m = Av.to_matrix(sparse=True)
        vals.append(np.real(np.vdot(psi.data, Av_m @ psi.data)))
    return np.array(vals)


if __name__ == "__main__":
    lat = Lattice(Lx=3, Ly=3, pbc=True)
    h = 1.0
    p = 4

    print(f"QAOA (p={p}) on the paper's {lat.n_qubits}-qubit, 3x3 periodic "
          f"lattice, h={h}\n")

    t0 = time.time()
    result = run_qaoa(lat, h, p, n_restarts=6, seed=7, maxiter=200, verbose=True)
    dt = time.time() - t0

    evals, _ = ground_state_reduced(lat, h, k=1)
    gauge_vals = qaoa_gauge_invariance(lat, result["statevector"])

    print(f"\nWall time: {dt:.1f} s")
    print(f"E_qaoa  = {result['energy']:.6f}")
    print(f"E_exact = {evals[0]:.6f}  (from the cycle-space reduced ED)")
    print(f"Gap (>=0 by the variational principle): {result['energy'] - evals[0]:.6f}")
    print(f"Relative gap: {(result['energy'] - evals[0]) / abs(evals[0]) * 100:.3f}%")
    print(f"QAOA <A_v> range: [{gauge_vals.min():.10f}, {gauge_vals.max():.10f}] "
          f"(should be ~+1.0 -- confirms the circuit stays in the physical "
          f"sector even though nothing forces it to)")
