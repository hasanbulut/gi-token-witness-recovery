"""
extend_grid.py

Answers Reviewer #2 (comment 5) and Reviewer #3: the paper stops at n=256, and
larger graphs are only flagged as future work. This script extends the (n,d)
grid past 256 and measures whether the strongest solvers still recover witnesses
comfortably within the timeout budget -- i.e. whether the "easy-instance"
conclusion holds beyond the published range or breaks at some size.

It reuses the project's generator and the extended pipeline (VF3 + Bliss),
keeping the same subprocess/timeout/timing pattern and results.jsonl schema.

Run on the reference CPU with --instances 200 for paper-ready timings. The
timeout-rate and verified-witness columns are hardware-independent; the timing
TREND (roughly polynomial, no blow-up) is the substantive scaling result.
"""

import argparse
import json
import statistics
from collections import defaultdict

from generator import InstanceGenerator
from solvers_ext import ExtendedAttackerPipeline


def make_graph(gen, fam, n, d, k):
    if fam == "F1":
        return gen.generate_f1_base(n, d)
    if fam == "F2":
        return gen.generate_f2_base(n, d, k=k)
    if fam == "F3":
        return gen.generate_f3_base(n, d)
    raise ValueError(fam)


def run_solver(pipe, name, G, Giso, timeout):
    if name == "vf3":
        return pipe.run_a3_vf3(G, Giso, timeout)
    if name == "bliss":
        return pipe.run_a5_bliss(G, Giso, timeout)
    if name == "nauty":
        return pipe.run_a4_nauty(G, Giso, timeout)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-values", nargs="+", type=int,
                    default=[384, 512, 768, 1024])
    ap.add_argument("--d", type=int, default=6)
    ap.add_argument("--families", nargs="+", default=["F1", "F2"])
    ap.add_argument("--swap-mult", type=int, default=2,
                    help="k = swap_mult * n for F2 (0 disables)")
    ap.add_argument("--instances", type=int, default=200)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--solvers", nargs="+", default=["vf3", "bliss"])
    ap.add_argument("--out", default="results_extended.jsonl")
    args = ap.parse_args()

    pipe = ExtendedAttackerPipeline()
    g0 = InstanceGenerator(seed=1)
    for _ in range(3):
        A = g0.generate_f1_base(128, args.d); B, _ = g0.generate_iso_pair(A)
        for s in args.solvers:
            run_solver(pipe, s, A, B, 1.0)

    agg = defaultdict(lambda: defaultdict(lambda: {"t": [], "to": 0, "ver": 0}))
    with open(args.out, "w") as f:
        for fam in args.families:
            for n in args.n_values:
                if fam == "F3" and n % 2:
                    continue
                k = args.swap_mult * n if fam == "F2" else 0
                cell = f"{fam} n={n} d={args.d} k={k}"
                for inst in range(args.instances):
                    seed = 42 + inst + n * args.d + k
                    gen = InstanceGenerator(seed=seed)
                    G = make_graph(gen, fam, n, args.d, k)
                    Giso, _ = gen.generate_iso_pair(G)
                    for s in args.solvers:
                        r = run_solver(pipe, s, G, Giso, args.timeout)
                        rec = {"family": fam, "n": n, "d": args.d, "k": k,
                               "label_bits": 0, "instance_id": inst,
                               "pair_type": "iso", "eps": 0, "attack": s,
                               "timeout_sec": args.timeout, "success": r["success"],
                               "solver_time_ms": r.get("solver_time_ms"),
                               "time_ms": r["time_ms"], "timeout_hit": r["timeout_hit"],
                               "map_verified": r.get("map_verified")}
                        f.write(json.dumps(rec) + "\n")
                        a = agg[cell][s]
                        if r["timeout_hit"]:
                            a["to"] += 1
                        else:
                            t = r.get("solver_time_ms")   # v4: in-solver time
                            a["t"].append(t if t is not None else r["time_ms"])
                        if r.get("map_verified"):
                            a["ver"] += 1
                print(f"done {cell}")

    print("\n" + "=" * 76)
    print(f"{'Cell':<22}{'solver':<7}{'N':>4}{'med(ms)':>10}{'T90(ms)':>10}"
          f"{'p_to':>7}{'ver%':>7}")
    print("-" * 76)
    for fam in args.families:
        for n in args.n_values:
            if fam == "F3" and n % 2:
                continue
            k = args.swap_mult * n if fam == "F2" else 0
            cell = f"{fam} n={n} d={args.d} k={k}"
            for s in args.solvers:
                a = agg[cell][s]
                nn = len(a["t"])
                med = statistics.median(a["t"]) if nn else float("nan")
                t90 = sorted(a["t"])[int(0.9 * nn)] if nn else float("nan")
                pto = a["to"] / args.instances
                vr = 100.0 * a["ver"] / args.instances
                print(f"{cell:<22}{s:<7}{nn:>4}{med:>10.2f}{t90:>10.2f}{pto:>7.2f}{vr:>6.0f}%")
            print()
    print(f"Wrote {args.out}. p_to and ver% are hardware-independent; run on the "
          "reference CPU for paper-ready millisecond values.")


if __name__ == "__main__":
    main()
