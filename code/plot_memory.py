"""
plot_memory.py  ->  figure for Step 8 (Reviewer 1, Section 4.2)

Reads results_memory.jsonl (from memory_profile.py) and draws a two-panel
figure: (left) solver working memory (MB) per cell grouped by solver, and
(right) median solve time (ms, log scale) for the same cells. Together this
shows R1's point that the VF2++/VF3 difference is in time, not memory.

Usage:
    python3 plot_memory.py
    python3 plot_memory.py results_memory.jsonl --out plot_memory.png --metric delta
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
    mem = defaultdict(lambda: defaultdict(list))
    tim = defaultdict(lambda: defaultdict(list))
    order = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("timeout_hit") or r.get("delta_kb") is None:
                continue   # timed-out / failed solves carry no memory numbers
            key = (r["family"], r["n"], r["d"], r.get("k", 0))
            if key not in order:
                order.append(key)
            mem[key][r["attack"]].append(r)          # keep record for chosen metric
            tim[key][r["attack"]].append(float(r["time_ms"]))
    return mem, tim, order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default="results_memory.jsonl")
    ap.add_argument("--out", default="plot_memory.png")
    ap.add_argument("--metric", choices=["delta", "peak"], default="delta",
                    help="delta = solver working set; peak = absolute RSS")
    args = ap.parse_args()

    mem, tim, order = load(args.results)
    field = "delta_kb" if args.metric == "delta" else "peak_kb"
    solvers = [s for s in SOLVER_ORDER if any(s in mem[k] for k in order)]
    labels = [f"{fam}\n{n}/{d}\nk={k}" for (fam, n, d, k) in order]

    x = np.arange(len(order))
    width = 0.8 / max(1, len(solvers))

    fig, (axm, axt) = plt.subplots(1, 2, figsize=(max(10, 1.6 * len(order)), 4.5))

    for i, s in enumerate(solvers):
        mvals = [float(np.median([rec[field] for rec in mem[k].get(s, [{field: np.nan}])])) / 1024.0
                 if mem[k].get(s) else np.nan for k in order]
        tvals = [float(np.median(tim[k][s])) if tim[k].get(s) else np.nan for k in order]
        axm.bar(x + i * width, mvals, width, label=SOLVER_LABEL[s], color=SOLVER_COLOR[s])
        axt.bar(x + i * width, tvals, width, label=SOLVER_LABEL[s], color=SOLVER_COLOR[s])

    ylab = "Solver working memory (MB)" if args.metric == "delta" else "Peak RSS (MB)"
    axm.set_ylabel(ylab)
    axm.set_title("Memory")
    axt.set_ylabel("Median solve time (ms)")
    axt.set_yscale("log")
    axt.set_title("Time (log scale)")
    for ax in (axm, axt):
        ax.set_xticks(x + width * (len(solvers) - 1) / 2)
        ax.set_xticklabels(labels, fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Solver memory vs. time: the VF2++/VF3 gap is time, not memory")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
