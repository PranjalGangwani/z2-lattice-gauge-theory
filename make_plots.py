"""
make_plots.py
=============

"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("scan_results.json") as f:
    rows = json.load(f)

h = [r["h"] for r in rows]
E_exact = [r["E_exact"] for r in rows]
E_qaoa = [r["E_qaoa"] for r in rows]
fid = [r["fidelity"] for r in rows]
W_plaq_exact = [r["W_plaq_exact"] for r in rows]
W_plaq_qaoa = [r["W_plaq_qaoa"] for r in rows]
W_bnd_exact = [r["W_bnd_exact"] for r in rows]
W_bnd_qaoa = [r["W_bnd_qaoa"] for r in rows]

h_c = 3.04438

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

ax = axes[0]
ax.plot(h, E_exact, "o-", label="Exact diagonalization", color="black")
ax.plot(h, E_qaoa, "s--", label="QAOA (p=3)", color="crimson")
ax.axvline(h_c, color="gray", linestyle=":", label=r"$h_c$ (thermodynamic limit)")
ax.set_xlabel("h")
ax.set_ylabel("Ground state energy")
ax.set_title("QAOA vs exact ground energy")
ax.legend()

ax = axes[1]
ax.plot(h, fid, "o-", color="darkorange")
ax.axvline(h_c, color="gray", linestyle=":")
ax.set_xlabel("h")
ax.set_ylabel(r"$|\langle\psi_{exact}|\psi_{QAOA}\rangle|^2$")
ax.set_title("State fidelity")
ax.set_ylim(0.99, 1.001)

ax = axes[2]
ax.plot(h, W_plaq_exact, "o-", label=r"$W$ area=1 (exact)", color="steelblue")
ax.plot(h, W_plaq_qaoa, "s--", label=r"$W$ area=1 (QAOA)", color="lightsteelblue")
ax.plot(h, W_bnd_exact, "o-", label=r"$W$ area=4 (exact)", color="firebrick")
ax.plot(h, W_bnd_qaoa, "s--", label=r"$W$ area=4 (QAOA)", color="lightcoral")
ax.axvline(h_c, color="gray", linestyle=":")
ax.set_xlabel("h")
ax.set_ylabel(r"$\langle W_\Gamma \rangle$")
ax.set_title("Wilson loops: confinement (small h) -> deconfinement (large h)")
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig("z2_lgt_qaoa_verification.png", dpi=150)
print("Saved z2_lgt_qaoa_verification.png")
