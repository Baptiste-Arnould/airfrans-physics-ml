"""Learning curves of a run: runs/<run>/log.csv -> runs/<run>/loss.png.

python src/plot_log.py mlp-full                     # run already local
python src/plot_log.py mlp-full --watch trillium    # live: re-fetch log.csv every 30 s until the job writes train.json
Open loss.png in VS Code: the preview reloads each time the file changes.
"""
import argparse
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from data import RUNS  # noqa: E402

HOSTS = {"trillium": "trillium-gpu", "narval": "narval"}  # socket ~/.ssh/cm-<h>, host <h>.alliancecan.ca
REMOTE_RUNS = "/scratch/baarn45/FLUID_BENCH/runs"
BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e6e5e1"


def fetch(run, cluster):
    h = HOSTS[cluster]
    src = f"baarn45@{h}.alliancecan.ca:{REMOTE_RUNS}/{run.name}/"
    r = subprocess.run(["rsync", "-a", "--ignore-missing-args", "-e", f"ssh -S {Path.home()}/.ssh/cm-{h} -o BatchMode=yes",
                        src + "log.csv", src + "train.json", f"{run}/"], capture_output=True, text=True, timeout=120)
    if r.returncode not in (0, 23):  # 23: run dir not created yet (job still pending) -> try again later
        raise SystemExit(f"rsync failed ({r.returncode}), is the SSH master open?\n{r.stderr}")


def plot(run):
    d = np.atleast_1d(np.genfromtxt(run / "log.csv", delimiter=",", names=True))
    if d.size == 0:  # header only: first validation not reached yet
        return
    for c in ("train", "val", "val_ux", "val_uy", "val_cp"):  # log.csv stores MSE; plot RMSE (in std units, readable)
        d[c] = np.sqrt(d[c])
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    a.plot(d["step"], d["train"], color=BLUE, lw=2, label="train (moyenne des derniers pas)")
    a.plot(d["step"], d["val"], color=ORANGE, lw=2, label="validation")
    for c, col, lab in (("val_ux", BLUE, "Ux / U∞"), ("val_uy", ORANGE, "Uy / U∞"), ("val_cp", AQUA, "Cp")):
        b.plot(d["step"], d[c], color=col, lw=2, label=lab)
    i = d["val"].argmin()
    for ax in (a, b):
        ax.set_yscale("log")
        ax.axvline(d["step"][i], color=MUTED, lw=1, ls=":")
        ax.grid(True, color=GRID, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("pas d'optimisation", color=MUTED)
        ax.legend(frameon=False, labelcolor=INK)
    right = d["step"][i] > d["step"][-1] / 2  # label on the inner side of the marker line, never off the axes
    a.annotate(f"best.pt (pas {int(d['step'][i])})", (d["step"][i], d["val"][i]), xytext=(-8 if right else 8, -14),
               textcoords="offset points", color=INK, ha="right" if right else "left")
    a.set_ylabel("RMSE (en écarts-types de la variable)", color=MUTED)
    a.set_title("Loss train et validation", loc="left", color=INK)
    b.set_title("Validation, par variable", loc="left", color=INK)
    state = "terminé" if (run / "train.json").exists() else "en cours"
    fig.suptitle(f"{run.name} : {int(d['step'][-1]):,} pas, {d['seconds'][-1] / 60:.1f} min ({state})", x=0.01, ha="left")
    tmp = run / "loss.tmp.png"
    fig.savefig(tmp, dpi=110, bbox_inches="tight")
    plt.close(fig)
    tmp.replace(run / "loss.png")  # atomic: the VS Code preview never sees a half-written file


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="directory name in runs/, e.g. mlp-full")
    ap.add_argument("--watch", choices=list(HOSTS), help="cluster to poll every 30 s")
    a = ap.parse_args()
    run = RUNS / a.run
    run.mkdir(parents=True, exist_ok=True)
    while True:  # ponytail: stops on train.json only; a crashed job needs Ctrl-C
        if a.watch:
            fetch(run, a.watch)
        if (run / "log.csv").exists():
            plot(run)
        if not a.watch or (run / "train.json").exists():
            break
        time.sleep(30)
    print(run / "loss.png")
