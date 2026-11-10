"""AirfRANS: native-order reader, per-case cache, train/val/test splits.

Inputs per node  x = [x, y, Uinf_x, Uinf_y, sdf, nx, ny]   (same features/order as airfrans.dataset.load)
Targets per node y = [Ux, Uy, p, nut]                       (p kinematic, m2/s2)
surf = wall nodes, from geometry only (sdf == 0). airfrans uses U_x == 0, i.e. the targets -> would leak on test cases.

Run `python src/data.py` to build the cache and self-check against the airfrans library.
"""
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[2] / "data"  # same layout locally and on clusters: <root>/{airfrans-physics-ml,data}
DATASET, CACHE = DATA / "Dataset", DATA / "cache"
RUNS = DATA.parent / "runs"  # training runs, predictions, entries (outside the repo)
# ponytail: assumes 2 HW threads/core (local machine: 52 threads = 26 cores); pass --workers explicitly in Slurm jobs
WORKERS = max(1, len(os.sched_getaffinity(0)) // 2)


def uinf_aoa(name):
    """Freestream speed (m/s) and angle of attack (deg) encoded in the case name."""
    return float(name.split("_")[2]), float(name.split("_")[3])


def read_case(name, root=DATASET):
    import pyvista as pv  # only needed to build the cache: training envs (clusters) need just numpy + torch

    vol = pv.read(root / name / f"{name}_internal.vtu")
    foil = pv.read(root / name / f"{name}_aerofoil.vtp")
    uinf, aoa = uinf_aoa(name)
    pos = np.asarray(vol.points[:, :2], np.float64)
    sdf = -np.asarray(vol.point_data["implicit_distance"], np.float64)
    surf = sdf == 0
    assert surf.sum() == foil.n_points, name
    # wall normals live on the .vtp; map them onto the volume wall nodes by exact position
    idx = {tuple(p): i for i, p in enumerate(np.asarray(foil.points[:, :2], np.float64))}
    normals = np.zeros_like(pos)
    normals[surf] = foil.point_data["Normals"][[idx[tuple(p)] for p in pos[surf]], :2]
    a = np.deg2rad(aoa)
    x = np.column_stack([pos, np.broadcast_to([uinf * np.cos(a), uinf * np.sin(a)], pos.shape), sdf, normals])
    y = np.column_stack([vol.point_data["U"][:, :2], vol.point_data["p"], vol.point_data["nut"]])
    return x.astype(np.float32), y.astype(np.float32), surf


def _cache_one(name):
    out = CACHE / f"{name}.npz"
    if not out.exists():
        x, y, surf = read_case(name)
        tmp = out.with_suffix(".tmp.npz")
        np.savez(tmp, x=x, y=y, surf=surf)
        tmp.rename(out)  # atomic: an interrupted build never leaves a truncated file
    return name


def load(name):
    d = np.load(CACHE / f"{name}.npz")
    return d["x"], d["y"], d["surf"]


def inputs(name):
    """Inputs only (npz is lazy: the targets are never read). Use this for anything that touches test cases."""
    return np.load(CACHE / f"{name}.npz")["x"]


def splits(task):
    """task in {'full', 'scarce', 'aoa'}. Validation is carved out of the official train set only."""
    m = json.loads((DATASET / "manifest.json").read_text())
    train = sorted(m[f"{task}_train"])
    test = m["full_test" if task == "scarce" else f"{task}_test"]  # official order == FluidsBench case_ids
    if task == "aoa":  # mimic extrapolation: validate on the most extreme AoA of the training range
        a = np.array([uinf_aoa(n)[1] for n in train])
        lo, hi = np.quantile(a, [0.05, 0.95])
        val = [n for n, ai in zip(train, a) if ai <= lo or ai >= hi]
    else:
        val = np.random.default_rng(0).choice(train, size=len(train) // 8, replace=False).tolist()
    train = [n for n in train if n not in set(val)]
    assert not set(train + val) & set(test), "test case leaked into train/val"
    return {"train": train, "val": sorted(val), "test": test}


if __name__ == "__main__":
    import airfrans as af

    names = sorted(p.name for p in DATASET.iterdir() if p.is_dir())
    # self-check: identical to the official library loader (which uses targets for the wall mask)
    s = af.Simulation(root=str(DATASET), name=names[0])
    x, y, surf = read_case(names[0])
    ref = np.concatenate([s.position, s.input_velocity, s.sdf, s.normals], 1)
    assert np.allclose(x, ref, atol=1e-5) and (surf == s.surface).all()
    assert np.allclose(y, np.concatenate([s.velocity, s.pressure, s.nu_t], 1), rtol=1e-6, atol=1e-6)
    for t in ("full", "scarce", "aoa"):
        sp = splits(t)
        print(t, {k: len(v) for k, v in sp.items()})

    CACHE.mkdir(exist_ok=True)
    with Pool(WORKERS) as pool:
        for i, _ in enumerate(pool.imap_unordered(_cache_one, names), 1):
            if i % 100 == 0:
                print(f"cached {i}/{len(names)}", flush=True)
    print("cache:", CACHE)
