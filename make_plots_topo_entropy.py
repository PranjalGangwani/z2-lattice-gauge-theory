"""
make_plots_topo_entropy.py
============================

Plot S_topo(h) from topo_entropy_results.json -- the paper's headline
topological-order signature: S_topo ~ 0 in the confined phase, S_topo ->
-ln(2) deep in the deconfined/topologically-ordered phase.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("topo_entropy_results.json") as f:
    results = json.load(f)

h_values = [r["h"] for r in results]
S_topo = [r["S_topo"] for r in results]

h_c = 3.04438
minus_ln2 = -np.log(2)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(h_values, S_topo, "o-", color="darkorange", label=r"$S_{topo}(h)$ (our KP regions)")
ax.axhline(minus_ln2, color="steelblue", linestyle="--",
           label=r"universal $\mathbb{Z}_2$ value $-\ln 2 \approx -0.693$")
ax.axhline(0, color="gray", linewidth=0.8)
ax.axvline(h_c, color="crimson", linestyle=":", label=r"$h_c \approx 3.044$")
ax.set_xlabel("h")
ax.set_ylabel(r"$S_{topo}$ (nats)")
ax.set_title("Topological entanglement entropy: confined (~0) vs\ndeconfined (approaching $-\\ln 2$)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("topo_entropy_verification.png", dpi=150)
print("Saved topo_entropy_verification.png")
