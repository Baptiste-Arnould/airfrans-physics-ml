"""Write official prediction files: <entry>/predictions/<case>.npz with exactly
velocity (N,2) m/s, pressure (N,) m2/s2, internal_points (N,3), airfoil_points (M,3) -- native node order.

Native coordinates come from the meshes (never from the float32 cache) so the official order check is exact.
Baseline: python src/export.py uniform --task full   -> runs/uniform-full/{entry.json,predictions/}
Trained:  python src/export.py mlp --task full       -> reads runs/mlp-full/raw/ (src/train.py configs/mlp.toml)
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyvista as pv

from data import DATASET, RUNS, WORKERS, inputs, splits

SPLIT_ID = {"full": "full", "scarce": "scarce", "aoa": "aoa_extrapolation"}  # airfrans task -> FluidsBench split


def write(name, fields, out_dir):
    """fields: (N, 3) = Ux, Uy, p in physical units, native node order."""
    case = DATASET / name
    pts = np.asarray(pv.read(case / f"{name}_internal.vtu").points, np.float64)
    foil = np.asarray(pv.read(case / f"{name}_aerofoil.vtp").points, np.float64)
    fields = np.asarray(fields, np.float64)
    assert fields.shape == (len(pts), 3) and np.isfinite(fields).all(), name
    np.savez(Path(out_dir) / f"{name}.npz", velocity=fields[:, :2], pressure=fields[:, 2],
             internal_points=pts, airfoil_points=foil)


def _uniform(args):
    """Trivial baseline: freestream velocity everywhere, p = 0. Uses inputs only."""
    name, out_dir = args
    x = inputs(name)
    write(name, np.c_[x[:, 2:4], np.zeros(len(x))], out_dir)


def _from_run(args):
    """Trained model: raw predictions written by src/train.py in <entry>/raw/."""
    name, out_dir = args
    d = np.load(out_dir.parent / "raw" / f"{name}.npz")
    write(name, np.c_[d["velocity"], d["pressure"]], out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="uniform, or a config name trained by src/train.py (e.g. mlp)")
    ap.add_argument("--task", choices=list(SPLIT_ID), default="full")
    a = ap.parse_args()
    entry = RUNS / f"{a.model}-{a.task}"
    (entry / "predictions").mkdir(parents=True, exist_ok=True)
    (entry / "entry.json").write_text(json.dumps({
        "schema": "fluidsbench-airfrans-entry-v1",
        "submission_id": f"{a.model}-{SPLIT_ID[a.task]}-v1".replace("_", "-"),
        "split_id": SPLIT_ID[a.task],
        "force_prediction_source": "field_integrated",
    }, indent=2))
    names = splits(a.task)["test"]
    with Pool(WORKERS) as pool:
        pool.map(_uniform if a.model == "uniform" else _from_run, [(n, entry / "predictions") for n in names])
    print(f"{len(names)} predictions -> {entry / 'predictions'}")
