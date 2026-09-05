## How to run

python3 lattice.py               # lattice geometry, open + periodic (torus)

python3 hamiltonian.py           # H_E, H_B, gauge constraint, verify [A_v,H]=0

python3 exact_diag.py            # brute-force ED on the small 12-qubit lattice

python3 qaoa_circuit.py          # QAOA ansatz circuit, inspect native gates

python3 optimize.py              # optimize QAOA at h=1 on the 12-qubit lattice

python3 observables.py           # small-lattice Wilson loop scan

python3 make_plots.py            # -> z2_lgt_qaoa_verification.png

python3 physical_basis.py        # the cycle-space trick: ED at 18 qubits

python3 qaoa_18qubit_demo.py     # QAOA on the paper's actual 18-qubit lattice (~4 min)

python3 wilson_scan_multisize.py # confinement fit, L=3 and L=4 -> wilson_multisize_results.json

python3 make_plots_multisize.py  # -> wilson_multisize_verification.png

python3 topological_sectors.py   # the 4 |tau_h,tau_v> ground states

python3 topo_entropy.py          # -> topo_entropy_results.json

python3 make_plots_topo_entropy.py  # -> topo_entropy_verification.png

python3 test_z2_lgt.py           

# What is reproduced - 

-Paper component	Status	Where

Hamiltonian H=H_E+h*H_B, Gauss-law gauge constraint

-QAOA circuit (mixer=H_E, problem=H_B)

-Classical optimizer

-12 Qubit lattice

-18 qubit lattice

-confinement and deconfinement

-4 states

-Topological entanglement entropy (Kitaev-Preskill)


Very tough to run for L = 4 and L = 5 ( 32 qubits and 50 qubits - 2^32 / 2^50 )

Used Duality trick to work around the same
