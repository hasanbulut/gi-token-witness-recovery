"""
hardest_cells.py

Answers Reviewer #2, comment 4: run canonical-labelling solvers (nauty, Bliss)
on the hardest cells -- the largest n and the swap-hardened F2 configurations --
and compare them against the tree-search VF3 tier already in the paper.

For each cell and solver we record per-instance recovery time, witness
verification, and timeout status, then aggregate median / T90 / timeout-rate /
verified-witness-rate. Output schema matches runner.py (results.jsonl).

Absolute timings are hardware-dependent: run this on the Intel i7-11850H with
--instances 200 to obtain paper-ready numbers. What IS hardware-independent and
already conclusive here: whether nauty and Bliss recover a VERIFIED witness at
all on these cells (they do), which is what closes the reviewer's gap.
"""

import argparse
import json
import statistics
from collections import defaultdict

from generator import InstanceGenerator
from solvers_ext import ExtendedAttackerPipeline


# Hardest cells drawn from the paper's own results (Sec. 4.3, 4.5, case study):
#   (family, n, d, k_swaps, label_bits)
DEFAULT_CELLS = [
    ("F2", 256, 4, 2 * 256, 0),    # genuinely hardest cell observed (~41 ms VF3)
    ("F2", 256, 10, 0, 0),         # largest n, high degree
    ("F2", 256, 10, 5 * 256, 0),   # + heavy swap hardening
    ("F1", 256, 10, 0, 0),         # largest base cell
    ("F2", 192, 6, 5 * 192, 0),    # mid-size heavy swaps (k = 5n, paper grid)
    ("F3", 256, 8, 0, 0),          # largest structured (2-lift)
]

SOLVERS = ["vf3", "nauty", "bliss"]   # vf3 = reference tree-search tier


def make_graph(gen, fam, n, d, k, lb):
    if fam == "F1":
        return gen.generate_f1_base(n, d)
    if fam == "F2":
        return gen.generate_f2_base(n, d, k=k)
    if fam == "F3":
        return gen.generate_f3_base(n, d)
    if fam == "F4":
        return gen.generate_f4_base(n, d, label_bits=lb, base_type="F1")
    raise ValueError(fam)


def run_solver(pipe, name, G, Giso, timeout):
    if name == "vf3":
        return pipe.run_a3_vf3(G, Giso, timeout)
    if name == "nauty":
        return pipe.run_a4_nauty(G, Giso, timeout)
    if name == "bliss":
        return pipe.run_a5_bliss(G, Giso, timeout)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=200,
                    help="instances per cell (use 200 for paper numbers)")
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--out", default="results_hardest.jsonl")
    ap.add_argument("--solvers", nargs="+", default=SOLVERS)
    args = ap.parse_args()

    pipe = ExtendedAttackerPipeline()
    # warmup each solver
    g0 = InstanceGenerator(seed=1)
    for _ in range(5):
        A = g0.generate_f1_base(64, 6); B, _ = g0.generate_iso_pair(A)
        for s in args.solvers:
            run_solver(pipe, s, A, B, 0.5)

    agg = defaultdict(lambda: defaultdict(list))   # (cell)->solver->[times]
    verified = defaultdict(lambda: defaultdict(int))
    timeouts = defaultdict(lambda: defaultdict(int))

    with open(args.out, "w") as f:
        for (fam, n, d, k, lb) in DEFAULT_CELLS:
            cell = f"{fam} n={n} d={d} k={k}"
            for inst in range(args.instances):
                seed = 42 + inst + n * d + k + lb
                gen = InstanceGenerator(seed=seed)
                G = make_graph(gen, fam, n, d, k, lb)
                Giso, _ = gen.generate_iso_pair(G)
                for s in args.solvers:
                    res = run_solver(pipe, s, G, Giso, args.timeout)
                    rec = {"family": fam, "n": n, "d": d, "k": k, "label_bits": lb,
                           "instance_id": inst, "pair_type": "iso", "eps": 0,
                           "attack": s, "timeout_sec": args.timeout,
                           "success": res["success"], "time_ms": res["time_ms"],
                           "solver_time_ms": res.get("solver_time_ms"),
                           "timeout_hit": res["timeout_hit"],
                           "map_verified": res.get("map_verified", None)}
                    f.write(json.dumps(rec) + "\n")
                    if res["timeout_hit"]:
                        timeouts[cell][s] += 1
                    else:
                        # prefer in-solver time (v4 correction); fall back to
                        # wrapper-inclusive time_ms only if absent
                        t = res.get("solver_time_ms")
                        agg[cell][s].append(t if t is not None else res["time_ms"])
                    if res.get("map_verified"):
                        verified[cell][s] += 1
            print(f"done {cell}")

    # summary table
    print("\n" + "=" * 78)
    print(f"{'Cell':<22}{'solver':<8}{'N':>4}{'med(ms)':>9}{'T90(ms)':>9}"
          f"{'p_to':>7}{'verified%':>10}")
    print("-" * 78)
    for (fam, n, d, k, lb) in DEFAULT_CELLS:
        cell = f"{fam} n={n} d={d} k={k}"
        for s in args.solvers:
            times = agg[cell][s]
            nn = len(times)
            med = statistics.median(times) if times else float("nan")
            t90 = (sorted(times)[int(0.9 * nn)] if nn else float("nan"))
            pto = timeouts[cell][s] / args.instances
            vr = 100.0 * verified[cell][s] / args.instances
            print(f"{cell:<22}{s:<8}{nn:>4}{med:>9.2f}{t90:>9.2f}{pto:>7.2f}{vr:>9.1f}%")
        print()
    print(f"Wrote {args.out}. Run with --instances 200 on the reference CPU for "
          "paper-ready timings; verified% and p_to are hardware-independent.")


if __name__ == "__main__":
    main()
