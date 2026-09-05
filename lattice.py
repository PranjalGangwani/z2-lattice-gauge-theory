
from dataclasses import dataclass, field
from itertools import product

Vertex = tuple  # (row, col)

@dataclass
class Lattice:
    Lx: int
    Ly: int
    pbc: bool = False
    n_qubits: int = field(init=False)
    link_index: dict = field(init=False)
    links: list = field(init=False)
    plaquettes: list = field(init=False)
    stars: dict = field(init=False)

    def __post_init__(self):
        if self.pbc and (self.Lx < 3 or self.Ly < 3):
            raise ValueError(
            )
        # Ny, Nx
        # Open: (Ly+1) x (Lx+1)
        # Periodic: Ly x Lx
        self.Ny = self.Ly if self.pbc else self.Ly + 1
        self.Nx = self.Lx if self.pbc else self.Lx + 1

        self.vertices = [(r, c) for r, c in product(range(self.Ny), range(self.Nx))]

        def V(r, c):
            if self.pbc:
                return (r % self.Ny, c % self.Nx)
            return (r, c)

        # --- build links (edges) and assign each a qubit index ---
        self.link_index = {}
        self.links = []

        def add_link(va, vb):
            key = frozenset((va, vb))
            if key not in self.link_index:
                self.link_index[key] = len(self.links)
                self.links.append(key)
            return self.link_index[key]

        # horizontal links: (r, c) -- (r, c+1), wrapping c+1 under pbc
        c_range = range(self.Nx) if self.pbc else range(self.Nx - 1)
        for r in range(self.Ny):
            for c in c_range:
                add_link(V(r, c), V(r, c + 1))
        # vertical links: (r, c) -- (r+1, c), wrapping r+1 under pbc
        r_range = range(self.Ny) if self.pbc else range(self.Ny - 1)
        for r in r_range:
            for c in range(self.Nx):
                add_link(V(r, c), V(r + 1, c))

        self.n_qubits = len(self.links)

        # --- build plaquettes: one elementary square per (r, c) cell ---
        # (r, c) ranges over Ly x Lx cells regardless of pbc; V() handles
        # wrapping the plaquette's own corners back onto the torus.
        self.plaquettes = []
        for r in range(self.Ly):
            for c in range(self.Lx):
                bl, br = V(r, c), V(r, c + 1)
                tl, tr = V(r + 1, c), V(r + 1, c + 1)
                bottom = self.link_index[frozenset((bl, br))]
                right = self.link_index[frozenset((br, tr))]
                top = self.link_index[frozenset((tl, tr))]
                left = self.link_index[frozenset((bl, tl))]
                self.plaquettes.append((bottom, right, top, left))

        # --- build stars: all links touching each vertex ---
        self.stars = {v: [] for v in self.vertices}
        for q, link in enumerate(self.links):
            for v in link:
                self.stars[v].append(q)

    # convenience

    def summary(self) -> str:
        n_plaq = len(self.plaquettes)
        n_vert = len(self.vertices)
        deg = {v: len(self.stars[v]) for v in self.vertices}
        bc = "periodic (torus)" if self.pbc else "open"
        return (
            f"Lattice {self.Lx}x{self.Ly} plaquettes ({bc})\n"
            f"  vertices   : {n_vert}\n"
            f"  qubits/links: {self.n_qubits}\n"
            f"  plaquettes : {n_plaq}\n"
            f"  vertex degrees (min/max): {min(deg.values())}/{max(deg.values())}"
        )


if __name__ == "__main__":
    print("=== Open boundary, 2x2 (previous demo, unchanged) ===")
    lat = Lattice(Lx=2, Ly=2)
    print(lat.summary())

    print("\n=== Periodic boundary, 3x3 (paper's main 18-qubit lattice) ===")
    lat_pbc = Lattice(Lx=3, Ly=3, pbc=True)
    print(lat_pbc.summary())
    print(f"  expected: 18 qubits, 9 plaquettes, 9 vertices, all degree 4 "
          f"(periodic -> no boundary, every vertex looks identical)")
    degs = sorted(len(lat_pbc.stars[v]) for v in lat_pbc.vertices)
    print(f"  actual vertex degrees: {set(degs)}")
