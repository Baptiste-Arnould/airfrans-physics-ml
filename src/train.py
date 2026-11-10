"""Pointwise MLP baseline (AirfRANS paper style): every mesh node is an independent sample.

Features (9): x, y, cos a, sin a, U_inf (== Re: c = 1 m and nu fixed), d, log10(d + 1e-7), nx, ny
  + if naca = true (7): NACA parameters from the case name: XX (thickness), is_5digit, M, P (4-digit), L, P, Q (5-digit)
Targets  (3): Ux/U_inf, Uy/U_inf, Cp = p / (U_inf^2 / 2)      -- both z-scored with train-split stats
No-slip is a hard constraint: U = 0 on wall nodes (d == 0, geometry only), in training and in prediction.
If delta is set (m): the network outputs N and U/U_inf = f(d) * N with f = 1 - exp(-d/delta) (0 at the wall, ~d/delta near it,
1 far away), and the loss is on N: the wall slope, hence the friction, becomes a direct O(1) output.

python src/train.py configs/mlp.toml [--smoke]
-> runs/<config>-<task>/ log.csv, last.pt (resume), best.pt (lowest val loss), train.json, raw/<case>.npz
raw/ holds val + test predictions (velocity, pressure; physical units, native order):
score val with src/score.py, export test with src/export.py.
"""
import argparse
import hashlib
import json
import os
import time
import tomllib
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data import RUNS, WORKERS, inputs, load, splits

LOG_COLS = ["step", "lr", "train", "val", "val_ux", "val_uy", "val_cp", "seconds"]


def naca(name):
    """[XX, is_5digit, M, P, L, P, Q]: name ends with M_P_XX (4-digit) or L_P_Q_XX (5-digit); other family's slots = 0."""
    p = [float(v) for v in name.split("_")[4:]]
    return [p[-1], 0, *p[:2], 0, 0, 0] if len(p) == 3 else [p[-1], 1, 0, 0, *p[:3]]


def features(x, name, use_naca):
    """Cache inputs [x, y, Uinf_x, Uinf_y, sdf, nx, ny] -> features (N, 9 or 16), U_inf (N,)."""
    uinf = np.hypot(x[:, 2], x[:, 3])
    d = x[:, 4]
    f = np.column_stack([x[:, 0], x[:, 1], x[:, 2] / uinf, x[:, 3] / uinf, uinf, d, np.log10(d + 1e-7), x[:, 5], x[:, 6]])
    if use_naca:
        f = np.column_stack([f, np.broadcast_to(naca(name), (len(f), 7))])
    return f.astype(np.float32), uinf


def damping(d, delta):
    """Velocity output factor f(d) = 1 - exp(-d/delta); delta None -> 1 (network outputs U/U_inf directly)."""
    return np.ones_like(d) if delta is None else -np.expm1(-d / delta)


def case(name, use_naca, delta=None):
    """Features + dimensionless targets [U/(U_inf f), V/(U_inf f), Cp] of one case (train/val only: reads the targets)."""
    x, y, _ = load(name)
    f, uinf = features(x, name, use_naca)
    fw = damping(x[:, 4], delta)[:, None]
    n = np.divide(y[:, :2] / uinf[:, None], fw, out=np.zeros((len(y), 2), np.float32), where=fw > 0)  # wall: 0, masked
    return f, np.column_stack([n, y[:, 2] / (0.5 * uinf**2)]).astype(np.float32)


class MLP(nn.Module):
    def __init__(self, width, depth, xm, xs, ym, ys):
        super().__init__()
        dims = [len(xm)] + [width] * depth
        layers = []
        for a, b in zip(dims, dims[1:]):
            layers += [nn.Linear(a, b), nn.GELU()]
        self.net = nn.Sequential(*layers, nn.Linear(width, 3))
        for k, v in dict(xm=xm, xs=xs, ym=ym, ys=ys).items():  # normalisation stats travel with the weights
            self.register_buffer(k, torch.as_tensor(v, dtype=torch.float32))

    def forward(self, f):
        """Raw features -> z-scored targets; U forced to 0 on wall nodes (so their U error, and gradient, is 0)."""
        out = self.net((f - self.xm) / self.xs)
        u = torch.where((f[:, 5] == 0)[:, None], -self.ym[:2] / self.ys[:2], out[:, :2])
        return torch.cat([u, out[:, 2:]], 1)

    def z(self, t):
        return (t - self.ym) / self.ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--smoke", action="store_true", help="3 cases per split, 200 steps: local pipeline check")
    a = ap.parse_args()
    cfg = tomllib.loads(Path(a.config).read_text())
    sp = splits(cfg["task"])
    if a.smoke:
        sp = {k: v[:3] for k, v in sp.items()}
        cfg |= dict(steps=200, val_every=50)
    use_naca, delta = cfg.get("naca", False), cfg.get("delta")
    run = RUNS / f"{Path(a.config).stem}-{cfg['task']}{'-smoke' * a.smoke}"
    (run / "raw").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with Pool(WORKERS) as pool:  # before any CUDA call (fork)
        load_case = partial(case, use_naca=use_naca, delta=delta)
        tr, va = pool.map(load_case, sp["train"]), pool.map(load_case, sp["val"])
    F, T = (np.concatenate(z) for z in zip(*tr))
    stats = [s(A, 0, dtype=np.float64) for A in (F, T) for s in (np.mean, np.std)]  # xm xs ym ys, train split only
    stats[1][stats[1] == 0] = 1  # constant feature (e.g. one NACA family only): no information, avoid 0/0
    print(f"data: {len(sp['train'])} train cases, {len(F):,} points, loaded in {time.time() - t0:.0f} s", flush=True)

    dev = "cuda"
    torch.manual_seed(cfg["seed"])
    torch.set_float32_matmul_precision("high")  # TF32 matmuls on A100/H100
    F, T = torch.from_numpy(F).to(dev), torch.from_numpy(T).to(dev)
    Fv, Tv = (torch.from_numpy(np.concatenate(z)).to(dev) for z in zip(*va))
    del tr, va
    model = MLP(cfg["width"], cfg["depth"], *stats).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["steps"])
    gen = torch.Generator(dev).manual_seed(cfg["seed"])
    step, best, best_step, elapsed = 0, float("inf"), 0, 0.0
    if (run / "last.pt").exists():  # resume after a walltime kill
        ck = torch.load(run / "last.pt")
        model.load_state_dict(ck["model"]), opt.load_state_dict(ck["opt"]), sched.load_state_dict(ck["sched"])
        gen.set_state(ck["gen"].cpu())
        step, best, best_step, elapsed = ck["step"], ck["best"], ck["best_step"], ck["elapsed"]
        print(f"resumed at step {step}", flush=True)
    else:
        (run / "log.csv").write_text(",".join(LOG_COLS) + "\n")

    @torch.no_grad()
    def val_loss():
        se = sum(((model(f) - model.z(t)) ** 2).sum(0) for f, t in zip(Fv.split(2**20), Tv.split(2**20)))
        return (se / len(Fv)).tolist()  # MSE per target (z-scored), Ux Uy Cp

    run_loss, t1 = torch.zeros((), device=dev), time.time() - elapsed
    while step < cfg["steps"]:
        i = torch.randint(len(F), (cfg["batch"],), device=dev, generator=gen)
        loss = ((model(F[i]) - model.z(T[i])) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        step += 1
        run_loss += loss.detach()
        if step % cfg["val_every"] == 0:
            v = val_loss()
            elapsed = time.time() - t1
            row = [step, sched.get_last_lr()[0], run_loss.item() / cfg["val_every"], sum(v) / 3, *v, elapsed]
            with open(run / "log.csv", "a") as fh:
                fh.write(",".join(f"{r:.6g}" for r in row) + "\n")
            print(" ".join(f"{c}={r:.4g}" for c, r in zip(LOG_COLS, row)), flush=True)
            run_loss.zero_()
            if sum(v) / 3 < best:
                best, best_step = sum(v) / 3, step
                torch.save(model.state_dict(), run / "best.pt")
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), gen=gen.get_state(),
                            step=step, best=best, best_step=best_step, elapsed=elapsed), run / "last.pt")

    # predict val + test with the best checkpoint; inputs only, so no test target is ever read
    model.load_state_dict(torch.load(run / "best.pt"))

    @torch.no_grad()
    def predict(names):
        """Cached inputs -> raw/<case>.npz. Returns wall seconds (read inputs + features + GPU + write)."""
        t = time.time()
        for name in names:
            f, uinf = features(inputs(name), name, use_naca)
            y = torch.cat([model(c) for c in torch.from_numpy(f).to(dev).split(2**20)])
            y = (y * model.ys + model.ym).cpu().numpy().astype(np.float64)
            y[:, :2] *= damping(f[:, 5].astype(np.float64), delta)[:, None]  # N -> U/U_inf
            y[f[:, 5] == 0, :2] = 0  # exact no-slip in physical units
            np.savez(run / "raw" / f"{name}.npz", velocity=y[:, :2] * uinf[:, None], pressure=y[:, 2] * 0.5 * uinf**2)
        return time.time() - t

    predict(sp["val"])
    test_seconds = predict(sp["test"])  # the full official test split: reported in the package
    info = dict(config=cfg, smoke=a.smoke, train_cases=len(sp["train"]), train_points=len(F),
                params=sum(p.numel() for p in model.parameters()), best_val=best, best_step=best_step,
                train_seconds=round(elapsed), test_cases=len(sp["test"]), predict_test_seconds=round(test_seconds, 2),
                gpu=torch.cuda.get_device_name(), torch=torch.__version__, slurm_job=os.environ.get("SLURM_JOB_ID"),
                checkpoint_sha256=hashlib.sha256((run / "best.pt").read_bytes()).hexdigest())
    (run / "train.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
