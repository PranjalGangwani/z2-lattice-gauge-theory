"""
physical_basis.py
==================

"""

from collections import deque

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from lattice import Lattice


def _bfs_spanning_tree(lat: Lattice):
    """Return (parent, parent_edge_qubit) dicts from a BFS spanning tree
    rooted at lat.vertices[0], plus the set of qubit indices used by the
    tree (the rest are "chords" that each define one fundamental cycle)."""
    adj = {v: [] for v in lat.vertices}
    for q, link in enumerate(lat.links):
        va, vb = tuple(link)
        adj[va].append((vb, q))
        adj[vb].append((va, q))

    root = lat.vertices[0]
    parent = {root: None}
    parent_edge_qubit = {root: None}
    visited = {root}
    tree_qubits = set()
    queue = deque([root])
    while queue:
        v = queue.popleft()
        for nbr, q in adj[v]:
            if nbr not in visited:
                visited.add(nbr)
                parent[nbr] = v
                parent_edge_qubit[nbr] = q
                tree_qubits.add(q)
                queue.append(nbr)

    assert len(visited) == len(lat.vertices), "lattice graph must be connected"
    return parent, parent_edge_qubit, tree_qubits


def _path_to_root_mask(v, parent, parent_edge_qubit):
    mask = 0
    while parent[v] is not None:
        mask ^= (1 << parent_edge_qubit[v])
        v = parent[v]
    return mask


def fundamental_cycle_basis(lat: Lattice) -> list:
    """Return a list of k = n_links - n_vertices + 1 integer bitmasks
    (bit q set <=> link q is in the cycle), one per non-tree ("chord")
    edge, spanning the cycle space of the lattice graph."""
    parent, parent_edge_qubit, tree_qubits = _bfs_spanning_tree(lat)
    chords = [q for q in range(lat.n_qubits) if q not in tree_qubits]

    cycles = []
    for q in chords:
        va, vb = tuple(lat.links[q])
        mask = (1 << q)
        mask ^= _path_to_root_mask(va, parent, parent_edge_qubit)
        mask ^= _path_to_root_mask(vb, parent, parent_edge_qubit)
        cycles.append(mask)
    return cycles


def enumerate_physical_states(lat: Lattice, cycle_basis=None):
    """All 2^k bitmasks reachable as XOR-combinations of the fundamental
    cycle basis -- i.e. every element of the cycle space, i.e. every
    physical (gauge-invariant) computational basis state IN THE
    HADAMARD-TRANSFORMED FRAME. Returns (states, index_of_state)."""
    if cycle_basis is None:
        cycle_basis = fundamental_cycle_basis(lat)
    k = len(cycle_basis)
    n_states = 1 << k
    states = [0] * n_states
    for i in range(n_states):
        s = 0
        for j in range(k):
            if (i >> j) & 1:
                s ^= cycle_basis[j]
        states[i] = s
    index_of = {s: i for i, s in enumerate(states)}
    assert len(index_of) == n_states, "cycle basis vectors were not independent!"
    return states, index_of


def reduced_hamiltonian(lat: Lattice, h: float, states=None, index_of=None) -> sp.csr_matrix:
    """Build H'(h) restricted to the physical subspace, as a sparse
    (2^k x 2^k) matrix, in the Hadamard-transformed frame described above.

        diagonal:      2 * popcount(x')             (from H_E')
        off-diagonal:  -h  at (index[x' ^ mask_p], index[x'])   for each
                        plaquette p, mask_p = XOR of its 4 qubit bits
                        (from h * H_B')
    """
    if states is None or index_of is None:
        states, index_of = enumerate_physical_states(lat)
    n_states = len(states)

    plaquette_masks = []
    for plaq in lat.plaquettes:
        m = 0
        for q in plaq:
            m ^= (1 << q)
        plaquette_masks.append(m)

    rows, cols, vals = [], [], []
    for i, x in enumerate(states):
        # H_E' diagonal term
        rows.append(i); cols.append(i)
        vals.append(2.0 * bin(x).count("1"))

        # h * H_B' off-diagonal terms
        for mask_p in plaquette_masks:
            y = x ^ mask_p
            j = index_of[y]  # guaranteed present: plaquette boundaries are themselves cycles
            rows.append(j); cols.append(i)
            vals.append(-h)

    H = sp.coo_matrix((vals, (rows, cols)), shape=(n_states, n_states)).tocsr()
    return H


def ground_state_reduced(lat: Lattice, h: float, k: int = 1, states=None, index_of=None, seed: int = 0):
    
    if states is None or index_of is None:
        states, index_of = enumerate_physical_states(lat)
    H = reduced_hamiltonian(lat, h, states, index_of)
    n_states = H.shape[0]

    DENSE_LIMIT = 4096
    if n_states <= DENSE_LIMIT:
        evals, evecs = np.linalg.eigh(H.toarray())
        order = np.argsort(evals)
        return evals[order][:k], evecs[:, order][:, :k]

    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(n_states)
    evals, evecs = eigsh(H, k=max(k, 6), which="SA", v0=v0, tol=0, maxiter=20000)
    order = np.argsort(evals)
    return evals[order][:k], evecs[:, order][:, :k]


if __name__ == "__main__":
    print("=== Validation: reduced-subspace ED must match full brute-force ED ===")
    from exact_diag import ground_state as full_ground_state

    lat = Lattice(Lx=2, Ly=2)  # pbc=False: our original 12-qubit lattice
    cycles = fundamental_cycle_basis(lat)
    k = len(cycles)
    print(f"2x2 open lattice: n_qubits={lat.n_qubits}, n_vertices={len(lat.vertices)}, "
          f"cyclomatic number k={k}  (physical subspace dim = 2^{k} = {2**k}, "
          f"vs full space 2^{lat.n_qubits} = {2**lat.n_qubits})")

    for h in [0.0, 1.0, 3.04438, 6.0]:
        e_full, _ = full_ground_state(lat, h, k=1)
        e_red, _ = ground_state_reduced(lat, h, k=1)
        diff = abs(e_full[0] - e_red[0])
        print(f"  h={h:8.5f}:  E_full={e_full[0]:12.6f}   E_reduced={e_red[0]:12.6f}   "
              f"diff={diff:.2e}  {'OK' if diff < 1e-8 else 'MISMATCH!!'}")

    print("\n=== Now the actual target: 3x3 periodic (18 qubits) ===")
    lat_pbc = Lattice(Lx=3, Ly=3, pbc=True)
    cycles_pbc = fundamental_cycle_basis(lat_pbc)
    k_pbc = len(cycles_pbc)
    print(f"3x3 periodic lattice: n_qubits={lat_pbc.n_qubits}, "
          f"n_vertices={len(lat_pbc.vertices)}, cyclomatic number k={k_pbc}  "
          f"(physical subspace dim = 2^{k_pbc} = {2**k_pbc}, "
          f"vs full space 2^{lat_pbc.n_qubits} = {2**lat_pbc.n_qubits:,})")

    import time
    for h in [1.0, 2.0, 3.04438, 4.0, 6.0]:
        t0 = time.time()
        evals, evecs = ground_state_reduced(lat_pbc, h, k=1)
        dt = time.time() - t0
        print(f"  h={h:8.5f}:  E0 = {evals[0]:12.6f}   (computed in {dt*1000:.1f} ms)")
