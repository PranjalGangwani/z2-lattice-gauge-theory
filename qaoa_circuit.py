"""
qaoa_circuit.py

"""

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp

from lattice import Lattice
from hamiltonian import electric_term, magnetic_term, _pauli_label


def build_qaoa_circuit(lat: Lattice, p: int) -> tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    """Build a p-layer QAOA circuit for the Z2 LGT Hamiltonian on `lat`.
    """
    n = lat.n_qubits
    betas = ParameterVector("beta", p)
    gammas = ParameterVector("gamma", p)

    qc = QuantumCircuit(n, name=f"QAOA_p{p}")

    # --- initial state: ground state of H_E = |+>^n ---
    qc.h(range(n))

    for layer in range(p):
        for plaq in lat.plaquettes:
            label = _pauli_label(n, {q: "Z" for q in plaq})
            zzzz = SparsePauliOp([label], [-1.0])  # matches H_B's single-plaquette term
            # PauliEvolutionGate(op, time) implements exp(-i * time * op)
            gate = PauliEvolutionGate(zzzz, time=gammas[layer])
            qc.append(gate, range(n))

        # --- mixer unitary exp(-i * beta * H_E) ---
        # H_E = sum_l (1 - X_l); the constant "1" part is a global phase we
        # can drop (irrelevant for expectation values), leaving RX on every
        # qubit: exp(i*beta*X_l) = RX_l(-2*beta).
        for q in range(n):
            qc.rx(-2 * betas[layer], q)

        qc.barrier()

    return qc, betas, gammas


if __name__ == "__main__":
    lat = Lattice(Lx=2, Ly=2)
    p = 2
    qc, betas, gammas = build_qaoa_circuit(lat, p)

    print(f"QAOA circuit: {lat.n_qubits} qubits, p={p} layers")
    print(f"  {len(betas)} mixer params (beta), {len(gammas)} problem params (gamma)")
    print(f"  circuit depth (with PauliEvolutionGate as an opaque instruction): {qc.depth()}")
    print(f"  total instructions: {len(qc.data)}")

    # Decompose once to see the actual native 1- and 2-qubit gates
    decomposed = qc.decompose(reps=3)
    print(f"\nAfter decomposing PauliEvolutionGate into native gates:")
    print(f"  depth: {decomposed.depth()}")
    counts = decomposed.count_ops()
    print(f"  gate counts: {dict(counts)}")

    print("\nFirst few lines of the (partially) decomposed circuit for one "
          "plaquette's ZZZZ evolution -- this is the CNOT staircase:")
    single_plaq_op = SparsePauliOp([_pauli_label(lat.n_qubits, {q: "Z" for q in lat.plaquettes[0]})], [1.0])
    demo = QuantumCircuit(lat.n_qubits)
    demo.append(PauliEvolutionGate(single_plaq_op, time=Parameter("g")), range(lat.n_qubits))
    print(demo.decompose(reps=3))
