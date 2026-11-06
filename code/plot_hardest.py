"""
plot_hardest.py  ->  figure for Step 7 (Reviewer 2, comment 4)

Reads results_hardest.jsonl (from hardest_cells.py) and draws a grouped bar
chart of median witness-recovery time per hardest cell, grouped by solver
(VF3, nauty, Bliss). Thin caps show the 90th percentile so the tail is visible.

Usage:
    python3 plot_hardest.py                      # reads results_hardest.jsonl
    python3 plot_hardest.py results_hardest.jsonl --out plot_hardest_cells.png
"""

import argparse
import json
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SOLVER_ORDER = ["vf2pp", "vf3", "nauty", "bliss"]
SOLVER_COLOR = {"vf2pp": "#8c8c8c", "vf3": "#1f77b4",
                "nauty": "#ff7f0e", "bliss": "#2ca02c"}
SOLVER_LABEL = {"vf2pp": "VF2++", "vf3": "VF3", "nauty": "nauty", "bliss": "Bliss"}


def load(path):
    cells = defaultdict(lambda: defaultdict(list))   # cellkey -> solver -> [ms]
    order = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("timeout_hit") or not r.get("success", False):
                continue
            key = (r["family"], r["n"], r["d"], r.get("k", 0))
            if key not in order:
                order.append(key)
            t = r.get("solver_time_ms")           # v4: prefer in-solver time
            cells[key][r["attack"]].append(float(t if t is not None else r["time_ms"]))
    return cells, order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default="results_hardest.jsonl")
    ap.add_argument("--out", default="plot_hardest_cells.png")
    ap.add_argument("--logy", action="store_true", help="log-scale y axis")
    args = ap.parse_args()

    cells, order = load(args.results)
    solvers = [s for s in SOLVER_ORDER
               if any(s in cells[k] for k in order)]
    labels = [f"{fam}\n{n}/{d}\nk={k}" for (fam, n, d, k) in order]

    x = np.arange(len(order))
    width = 0.8 / max(1, len(solvers))

    plt.figure(figsize=(max(7, 1.4 * len(order)), 4.5))
    for i, s in enumerate(solvers):
        meds, errs = [], []
        for k in order:
            v = cells[k].get(s, [])
            if v:
                m = float(np.median(v))
                p90 = float(np.percentile(v, 90))
                meds.append(m)
                errs.append(max(0.0, p90 - m))
            else:
                meds.append(np.nan)
                errs.append(0.0)
        plt.bar(x + i * width, meds, width, yerr=[[0] * len(meds), errs],
                capsize=3, label=SOLVER_LABEL[s], color=SOLVER_COLOR[s],
                error_kw={"elinewidth": 0.8, "alpha": 0.6})

    plt.xticks(x + width * (len(solvers) - 1) / 2, labels, fontsize=8)
    plt.ylabel("Median witness-recovery time (ms)")
    if args.logy:
        plt.yscale("log")
    plt.title("Witness recovery on the hardest cells (caps = 90th percentile)")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
