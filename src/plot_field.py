"""CFD vs model on one VALIDATION case: |U|/U_inf and Cp maps + error, surface Cp, boundary-layer profile at x/c = 0.5,
skin friction along the chord (official wall shear: airfrans for the truth, the evaluator's derive_case for predictions).
Second figure profiles-<case>.png: the 4 scored boundary-layer profiles (official extraction, FluidsBench extract.py).

python src/plot_field.py mlp_naca-full [--case NAME] [--compare mlp-full]   -> runs/<run>/field-<case>.png
Default case: median surface-pressure error in the run's val score (src/score.py output). Test cases are refused.
"""
import argparse
import csv
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402
import pyvista as pv  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

import airfrans as af  # noqa: E402

from data import DATA, DATASET, RUNS, load, splits  # noqa: E402

FB = DATA.parent / "fluidsbench-submission"
sys.path.insert(0, str(FB))
from scripts.derive_airfrans_predicted_fields import derive_case  # noqa: E402

PROFILES = json.loads((FB / "benchmark-specs/airfrans/velocity-profiles-v1.json").read_text())
_spec = importlib.util.spec_from_file_location("fb_extract", FB / "examples/airfrans-profile-extraction/extract.py")
extract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(extract)

SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#104281", "#3987e5", "#f0efec", "#e34948", "#9e2a2a"])
INK, MUTED, ORANGE, AQUA, SOLID = "#0b0b0b", "#52514e", "#eb6834", "#1baf7a", "#d9d8d4"
WIN = (-0.3, 1.4, -0.45, 0.45)  # zoom window (chord = 1 m, leading edge at 0)


def dimless(u, p, uinf):
    return np.hypot(u[:, 0], u[:, 1]) / uinf, p / (0.5 * uinf**2)


def wall_shear(name, pred=None):
    """Wall shear (M, 2) on the airfoil curve, .vtp order: truth, or derived from a prediction exactly as the evaluator does."""
    sim = af.Simulation(root=str(DATASET), name=name)  # fresh: derive_case injects the prediction into sim
    if pred is None:
        return sim.wallshearstress(over_airfoil=True, reference=True)
    with tempfile.TemporaryDirectory() as tmp:
        derive_case(sim, name, {"velocity": pred["velocity"], "pressure": pred["pressure"], "internal_points": sim.position,
                                "airfoil_points": sim.airfoil_position}, Path(tmp))
        c = np.load(Path(tmp) / f"{name}-curve-predictions.npz")
        return np.c_[c["wall_shear_stress_x_pred"], c["wall_shear_stress_y_pred"]]


def profiles(name, pred=None):
    """Scored profiles {(station, quantity): (distance m, values)}: truth, or from a prediction (official extraction)."""
    sim = af.Simulation(root=str(DATASET), name=name)
    series, _ = extract.extract_profiles(sim, PROFILES, reference=pred is None,
                                         velocity=None if pred is None else pred["velocity"])
    return {(r["station_id"], r["quantity_id"]): (np.array(r["coordinate"]), np.array(r["prediction"])) for r in series}


def plot_profiles(case, preds, colors, main, out):
    prof = {"CFD": profiles(case)} | {n: profiles(case, p) for n, p in preds.items()}
    fig, axs = plt.subplots(2, 4, figsize=(16, 9), sharey=True, layout="constrained")
    for j, st in enumerate(PROFILES["stations"]):
        for i, q in enumerate(PROFILES["quantities"]):
            ax = axs[i, j]
            for n, pr in prof.items():
                dist, v = pr[(st["id"], q["id"])]
                ax.plot(v[1:], dist[1:], color=INK if n == "CFD" else colors[n], lw=2.5 if n == "CFD" else 1.5,
                        ls="-" if n in ("CFD", main) else "--", label=n)
            ax.set_yscale("log")
            ax.grid(True, color="#e6e5e1", lw=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_title(f"x/c = {st['x_over_c']}, {q['definition']}", loc="left", color=INK, fontsize=11)
            ax.set_xlabel(q["definition"], color=MUTED)
            if j == 0:
                ax.set_ylabel("distance à la paroi le long de la normale (m)", color=MUTED)
    axs[0, 0].legend(frameon=False)
    fig.suptitle(f"{case} : profils de couche limite notés (extrados, ligne normale de 0,1 m, 1001 points)", x=0.01, ha="left")
    fig.savefig(out, dpi=100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="e.g. mlp_naca-full")
    ap.add_argument("--case")
    ap.add_argument("--compare", help="second run shown on the curves, e.g. mlp-full")
    a = ap.parse_args()
    run, task = RUNS / a.run, a.run.rsplit("-", 1)[1]
    rows = {r["case_id"]: r for r in csv.DictReader(open(run / "raw" / f"score-{task}-val-cases.csv"))}
    if a.case is None:
        srt = sorted(rows, key=lambda n: float(rows[n]["surface_pressure_rel_l2"]))
        a.case = srt[len(srt) // 2]
    assert a.case in splits(task)["val"], "validation cases only: test targets are never looked at"

    x, y, surf = load(a.case)
    pts, d, normal_y = x[:, :2], x[:, 4], x[:, 6]
    uinf = float(np.hypot(*x[0, 2:4]))
    preds = {a.run: np.load(run / "raw" / f"{a.case}.npz")}
    if a.compare:
        preds[a.compare] = np.load(RUNS / a.compare / "raw" / f"{a.case}.npz")
    U_t, Cp_t = dimless(y[:, :2], y[:, 2], uinf)
    U_p, Cp_p = dimless(preds[a.run]["velocity"], preds[a.run]["pressure"], uinf)

    quads = pv.read(DATASET / a.case / f"{a.case}_internal.vtu").cells_dict[pv.CellType.QUAD]
    near = (pts[:, 0] > WIN[0] - 0.2) & (pts[:, 0] < WIN[1] + 0.2) & (np.abs(pts[:, 1]) < WIN[3] + 0.2)
    q = quads[near[quads].all(1)]
    tri = mtri.Triangulation(pts[:, 0], pts[:, 1], np.r_[q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
    win = near & (pts[:, 0] > WIN[0]) & (pts[:, 0] < WIN[1]) & (np.abs(pts[:, 1]) < WIN[3])

    fig = plt.figure(figsize=(16, 16), layout="constrained")
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1.15, 1.15])
    # colour range clipped at the 0.5 % extremes of the window (leading-edge peak would flatten everything else)
    lo, hi = np.percentile(Cp_t[win], [0.5, 99.5])
    maps = [("|U| / U∞", U_t, U_p, SEQ, plt.Normalize(0, np.percentile(U_t[win], 99.5))),
            ("Cp", Cp_t, Cp_p, DIV, TwoSlopeNorm(0, min(lo, -1e-3), max(hi, 1e-3)))]
    for r, (lab, t, p, cmap, norm) in enumerate(maps):
        err = p - t
        lim = np.percentile(np.abs(err[win]), 99.5)
        for c, (val, cm, nm, title) in enumerate([(t, cmap, norm, f"CFD : {lab}"), (p, cmap, norm, f"{a.run} : {lab}"),
                                                  (err, DIV, plt.Normalize(-lim, lim), f"erreur {a.run} − CFD")]):
            ax = fig.add_subplot(gs[r, c])
            im = ax.tripcolor(tri, val, cmap=cm, norm=nm, shading="gouraud", rasterized=True)
            ax.set(xlim=WIN[:2], ylim=WIN[2:], aspect="equal", facecolor=SOLID, xticks=[], yticks=[])
            ax.set_title(title, loc="left", color=INK, fontsize=11)
            if c:
                fig.colorbar(im, ax=ax, shrink=0.8, extend="both")

    names = list(preds)[::-1]  # comparison drawn first, main run on top
    colors = {a.run: ORANGE, a.compare: AQUA}
    ax = fig.add_subplot(gs[2, :2])
    for side, sel in (("extrados", surf & (normal_y < 0)), ("intrados", surf & (normal_y >= 0))):
        o = np.flatnonzero(sel)[np.argsort(pts[sel, 0])]  # normals point into the airfoil: extrados has n_y < 0
        ax.plot(pts[o, 0], Cp_t[o], color=INK, lw=2, label="CFD" if side == "extrados" else None)
        for n in names:
            ax.plot(pts[o, 0], dimless(preds[n]["velocity"], preds[n]["pressure"], uinf)[1][o], color=colors[n], lw=1.5,
                    ls="-" if n == a.run else "--", label=n if side == "extrados" else None)
    ax.invert_yaxis()
    ax.set(xlabel="x / c", ylabel="Cp")
    ax.set_title("Pression à la paroi (axe Cp inversé)", loc="left", color=INK)

    bx = fig.add_subplot(gs[2, 2])
    yw = pts[surf & (np.abs(pts[:, 0] - 0.5) < 0.01), 1]
    s = (np.abs(pts[:, 0] - 0.5) < 0.004) & (d > 0) & (d < 0.05) & (pts[:, 1] > yw.mean())  # extrados, mid-chord slab
    o = np.flatnonzero(s)[np.argsort(d[s])]
    bx.plot(d[o], U_t[o], color=INK, lw=2, label="CFD")  # line, so predictions (markers) stay visible on top
    for n in names:
        bx.scatter(d[s], dimless(preds[n]["velocity"], preds[n]["pressure"], uinf)[0][s], s=14, color=colors[n], label=n)
    bx.set(xscale="log", xlabel="distance à la paroi d (m)", ylabel="|U| / U∞")
    bx.set_title("Couche limite, extrados, x/c ≈ 0,5", loc="left", color=INK)
    foil = pv.read(DATASET / a.case / f"{a.case}_aerofoil.vtp")
    fx, fny = np.asarray(foil.points[:, 0]), np.asarray(foil.point_data["Normals"][:, 1])
    cf = {"CFD": wall_shear(a.case)[:, 0] / (0.5 * uinf**2)} | {n: wall_shear(a.case, preds[n])[:, 0] / (0.5 * uinf**2) for n in names}
    sub = gs[3, :].subgridspec(1, 2)
    fax = []
    for k, (side, sel) in enumerate((("extrados", fny < 0), ("intrados", fny >= 0))):
        z = fig.add_subplot(sub[0, k])
        o = np.flatnonzero(sel)[np.argsort(fx[sel])]
        m = (fx[o] > 0.02) & (fx[o] < 0.98)  # y range from CFD + main run, away from the leading/trailing-edge spikes
        ref = np.r_[cf["CFD"][o][m], cf[a.run][o][m]]
        pad = 0.1 * (ref.max() - ref.min())
        lo, hi = ref.min() - pad, ref.max() + pad
        for n, v in cf.items():
            off = np.mean((v[o][m] < lo) | (v[o][m] > hi)) > 0.5
            z.plot(fx[o], v[o], color=INK if n == "CFD" else colors[n], lw=2 if n == "CFD" else 1.5,
                   ls="--" if n == a.compare else "-", label=n + (" (hors échelle)" if off else ""))
        z.set(ylim=(lo, hi), xlabel="x / c", ylabel="Cf_x = τ_x / (½ U∞²)")
        z.set_title(f"Frottement pariétal, {side}", loc="left", color=INK)
        fax.append(z)
    for z in (ax, bx, *fax):
        z.grid(True, color="#e6e5e1", lw=0.8)
        z.spines[["top", "right"]].set_visible(False)
        z.legend(frameon=False)

    m = rows[a.case]
    fig.suptitle(f"{a.case}   (U∞ = {uinf:.1f} m/s, α = {a.case.split('_')[3]}°)\nerreurs {a.run} : "
                 f"vitesse {float(m['flow_domain_velocity_rel_l2']):.2f} %, pression {float(m['flow_domain_pressure_rel_l2']):.2f} %, "
                 f"p paroi {float(m['surface_pressure_rel_l2']):.2f} %, frottement {float(m['surface_wall_shear_rel_l2']):.1f} %, "
                 f"Cd {float(m['cd_pred']):.4f} / {float(m['cd']):.4f}, Cl {float(m['cl_pred']):.3f} / {float(m['cl']):.3f}",
                 x=0.01, ha="left", fontsize=11)
    out = run / f"field-{a.case}.png"
    fig.savefig(out, dpi=100)
    plot_profiles(a.case, preds, colors, a.run, run / f"profiles-{a.case}.png")
    print(out)


if __name__ == "__main__":
    main()
