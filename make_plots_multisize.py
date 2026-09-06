"""
make_plots_multisize.py
=========================

"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wilson_scan_multisize import fit_area_perimeter_law

with open("wilson_multisize_results.json") as f:
    data = json.load(f)

all_rows = data["L3"] + data["L4"]
h_values = sorted({r["h"] for r in all_rows})

chis, deltas = [], []
for h in h_values:
    rows_h = [r for r in all_rows if r["h"] == h]
    chi, delta = fit_area_perimeter_law(rows_h)
    chis.append(chi)
    deltas.append(delta)

h_c = 3.04438

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
ax.plot(h_values, chis, "o-", color="steelblue", label=r"$\chi(h)$ (area-law coefficient)")
ax.axhline(0, color="gray", linewidth=0.8)
ax.axvline(h_c, color="crimson", linestyle=":", label=r"$h_c \approx 3.044$")
ax.set_xlabel("h")
ax.set_ylabel(r"$\chi(h)$")
ax.set_title("Confinement order parameter\n(exact ED, pooled L=3,4 loop sizes)")
ax.legend()

ax = axes[1]
for L, rows in [(3, data["L3"]), (4, data["L4"])]:
    for shape in sorted({tuple(r["shape"]) for r in rows}):
        ys = [r["W"] for r in rows if tuple(r["shape"]) == shape]
        ax.plot(h_values, ys, "o-", label=f"L={L} loop {shape[0]}x{shape[1]}", alpha=0.8)
ax.axvline(h_c, color="gray", linestyle=":")
ax.set_xlabel("h")
ax.set_ylabel(r"$\langle W_\Gamma \rangle$")
ax.set_title("Wilson loops of every available size, L=3 and L=4")
ax.legend(fontsize=7, ncol=2)

fig.tight_layout()
fig.savefig("wilson_multisize_verification.png", dpi=150)
print("Saved wilson_multisize_verification.png")
