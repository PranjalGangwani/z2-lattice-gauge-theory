"""
optimize.py
"""

import numpy as np
from scipy.optimize import minimize
from qiskit.quantum_info import Statevector

from lattice import Lattice
from hamiltonian import full_hamiltonian
from qaoa_circuit import build_qaoa_circuit


def energy_function(lat: Lattice, h: float, p: int, qc, betas, gammas):
    """Return a function f(x) -> <H> for x = concat(beta_1..p, gamma_1..p),
    plus the SparsePauliOp H(h) it uses (handy for reuse).

    IMPORTANT PERFORMANCE NOTE: qiskit's PauliEvolutionGate, left as an
    opaque instruction with a *symbolic* Parameter for its time, falls back
    to dense matrix exponentiation (scipy sparse expm) of the FULL 2^n x 2^n
    operator every time Statevector.from_instruction() is called on it --
    even though the underlying unitary only touches 4 qubits per plaquette.
    That's ~15 seconds per energy evaluation on 12 qubits, which makes
    optimization intractable.

    The fix: decompose the circuit ONCE (with symbolic parameters still in
    place) into native 1-/2-qubit gates -- exactly the CNOT-staircase shown
    in qaoa_circuit.py's __main__ block. Binding parameters and simulating
    THAT circuit only applies local few-qubit unitaries, which is ~1000x
    faster (single-digit milliseconds here) and is also what real hardware
    would actually execute.
    """
    H = full_hamiltonian(lat, h)
    qc_native = qc.decompose(reps=3)  # symbolic params survive decomposition

    def f(x):
        beta_vals = x[:p]
        gamma_vals = x[p:]
        bound = qc_native.assign_parameters(
            {**{betas[i]: beta_vals[i] for i in range(p)},
             **{gammas[i]: gamma_vals[i] for i in range(p)}}
        )
        psi = Statevector.from_instruction(bound)
        energy = np.real(psi.expectation_value(H))
        return energy

    return f, H


def run_qaoa(lat: Lattice, h: float, p: int, n_restarts: int = 8, seed: int = 0,
             maxiter: int = 300, verbose: bool = True):
    """Optimize QAOA parameters for H(h) with p layers."""
    rng = np.random.default_rng(seed)
    qc, betas, gammas = build_qaoa_circuit(lat, p)
    f, H = energy_function(lat, h, p, qc, betas, gammas)

    best = None
    for trial in range(n_restarts):
        x0 = rng.uniform(0, np.pi, size=2 * p)
        res = minimize(f, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.5})
        if best is None or res.fun < best.fun:
            best = res
        if verbose:
            print(f"  [h={h:.3f}] restart {trial+1}/{n_restarts}: "
                  f"E = {res.fun:.6f} (best so far: {best.fun:.6f})")

    beta_opt = best.x[:p]
    gamma_opt = best.x[p:]
    qc_native = qc.decompose(reps=3)
    bound = qc_native.assign_parameters(
        {**{betas[i]: beta_opt[i] for i in range(p)},
         **{gammas[i]: gamma_opt[i] for i in range(p)}}
    )
    psi_opt = Statevector.from_instruction(bound)

    return {
        "h": h,
        "p": p,
        "energy": best.fun,
        "beta": beta_opt,
        "gamma": gamma_opt,
        "statevector": psi_opt,
        "hamiltonian": H,
    }


if __name__ == "__main__":
    lat = Lattice(Lx=2, Ly=2)
    h = 1.0
    p = 3

    print(f"Optimizing QAOA (p={p}) for h={h} on the {lat.n_qubits}-qubit "
          f"2x2 lattice...\n")
    result = run_qaoa(lat, h, p, n_restarts=6, seed=42, maxiter=200)

    print(f"\nBest QAOA energy: {result['energy']:.6f}")
    print(f"Best beta:  {result['beta']}")
    print(f"Best gamma: {result['gamma']}")

    # quick sanity: variational principle says QAOA energy >= true ground energy
    from exact_diag import ground_state
    evals, evecs = ground_state(lat, h, k=1)
    print(f"\nExact ground energy E0 = {evals[0]:.6f}")
    print(f"QAOA energy - E0       = {result['energy'] - evals[0]:.6f}  "
          f"(should be >= 0, variational principle)")

    fidelity = np.abs(np.vdot(evecs[:, 0], result["statevector"].data)) ** 2
    print(f"Fidelity |<psi_exact|psi_QAOA>|^2 = {fidelity:.6f}")
