"""
plot_extend.py  ->  figure for Step 9 (Reviewer 2 comment 5, Reviewer 3)

Reads one or more extended-grid JSONL files (from extend_grid.py) and draws
median recovery time versus n on a log-y axis, one line per (family, solver).
Timed-out cells are marked so the VF3 tail at large n is visible. Accepts
multiple files because Step 9 runs Bliss and VF3 into separate files.

Usage:
    python3 plot_extend.py results_extended_bliss.jsonl results_extended.jsonl
    python3 plot_extend.py results_extended*.jsonl --out plot_extend_grid.png
"""

import argparse
import glob
import json
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SOLVER_STYLE = {"vf3": ("-", "o"), "bliss": ("-", "s"),
                "nauty": ("-", "^"), "vf2pp": ("-", "d")}
SOLVER_LABEL = {"vf3": "VF3", "bliss": "Bliss", "nauty": "nauty", "vf2pp": "VF2++"}
FAM_COLOR = {"F1": "#1f77b4", "F2": "#d62728", "F3": "#2ca02c", "F4": "#9467bd"}


def load(paths):
    # (family, solver) -> n -> {"t":[ms], "to":count, "N":count}
    series = defaultdict(lambda: defaultdict(lambda: {"t": [], "to": 0, "N": 0}))
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                key = (r["family"], r["attack"])
                cell = series[key][r["n"]]
                cell["N"] += 1
                if r.get("timeout_hit"):
                    cell["to"] += 1
                elif r.get("success", False):
                    t = r.get("solver_time_ms")   # v4: prefer in-solver time
                    cell["t"].append(float(t if t is not None else r["time_ms"]))
    return series


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+",
                    help="one or more extend_grid JSONL files (globs allowed)")
    ap.add_argument("--out", default="plot_extend_grid.png")
    args = ap.parse_args()

    paths = []
    for p in args.results:
        paths.extend(glob.glob(p))
    series = load(paths)

    plt.figure(figsize=(7.5, 5))
    for (fam, solver), by_n in sorted(series.items()):
        ns = sorted(by_n)
        meds = [np.median(by_n[n]["t"]) if by_n[n]["t"] else np.nan for n in ns]
        ls, mk = SOLVER_STYLE.get(solver, ("-", "o"))
        color = FAM_COLOR.get(fam, "#333333")
        plt.plot(ns, meds, ls, marker=mk, color=color,
                 label=f"{fam} / {SOLVER_LABEL.get(solver, solver)}")
        # mark cells with any timeout
        for n in ns:
            if by_n[n]["to"] > 0 and by_n[n]["t"]:
                plt.annotate(f"{by_n[n]['to']}/{by_n[n]['N']} TO",
                             (n, np.median(by_n[n]["t"])), fontsize=7,
                             xytext=(0, 8), textcoords="offset points",
                             ha="center", color=color)

    ax = plt.gca()
    ax.axvline(256, color="gray", ls=":", lw=1)
    ax.text(256, ax.get_ylim()[1], "published range ends (n=256)",
            rotation=90, va="top", ha="right", fontsize=7, color="gray")
    ax.set_xlim(left=min(240, ax.get_xlim()[0]))
    plt.yscale("log")
    plt.xlabel("graph size n")
    plt.ylabel("median recovery time (ms, log scale)")
    plt.title("Scaling past n=256: Bliss stays flat, VF3 develops a tail")
    plt.legend(fontsize=8)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
