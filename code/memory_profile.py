"""
memory_profile.py

Answers Reviewer #1, comment on Sec. 4.2: complement the VF2++-vs-VF3 runtime
comparison with solver memory consumption.

Method: each (solver, instance) runs in its own forked subprocess. Inside that
process we snapshot peak RSS (resource.ru_maxrss, a high-water mark) right after
the two graph objects are built (baseline) and again after the solve (peak). The
delta isolates the memory the SOLVER's search/canonical structures required, on
top of the graphs themselves; we also report absolute peak RSS. A psutil sampler
catches any sub-interval spike the coarse HWM might otherwise smooth over.

Memory is more portable across machines than wall-clock time, but absolute MB
still depend on library builds and the allocator, so regenerate on the reference
CPU for the paper. The ORDERING across solvers (the point of R1's comment) is the
robust finding.
"""

import argparse
import json
import resource
import statistics
import threading
import time
from collections import defaultdict
from multiprocessing import Process, Queue

import networkx as nx

# Pre-import heavy solvers in the PARENT so forked children inherit them and do
# not each pay a fresh C-extension import. Baseline RSS then already includes
# these libraries, which is fine because we report the per-solve DELTA.
try:
    import vf3py  # noqa: F401
    import pynauty  # noqa: F401
    import igraph  # noqa: F401
except Exception:
    pass


def _peak_rss_kb():
    # Linux: ru_maxrss is in kilobytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def _solve(name, G1, G2):
    """Direct (un-wrapped) solve call per solver; returns (found, verified)."""
    if name == "vf2pp":
        m = nx.vf2pp_isomorphism(G1, G2)
        return (m is not None), _verify(G1, G2, m)
    if name == "vf3":
        import vf3py
        isos = list(vf3py.get_exact_isomorphisms(G1, G2))
        m = isos[0] if isos else None
        return bool(isos), _verify(G1, G2, m)
    if name == "nauty":
        import pynauty
        from attackers import networkx_to_pynauty
        g1, map1 = networkx_to_pynauty(G1)
        g2, map2 = networkx_to_pynauty(G2)
        if not pynauty.isomorphic(g1, g2):
            return False, False
        c1 = pynauty.canon_label(g1); c2 = pynauty.canon_label(g2)
        r1 = {v: k for k, v in map1.items()}; r2 = {v: k for k, v in map2.items()}
        m = {r1[c1[i]]: r2[c2[i]] for i in range(len(c1))}
        return True, _verify(G1, G2, m)
    if name == "bliss":
        from solvers_ext import networkx_to_igraph
        ig1, _, r1 = networkx_to_igraph(G1)
        ig2, _, r2 = networkx_to_igraph(G2)
        if not ig1.isomorphic_bliss(ig2):
            return False, False
        p1 = ig1.canonical_permutation(); p2 = ig2.canonical_permutation()
        m = {r1[p1[k]]: r2[p2[k]] for k in range(len(p1))}
        return True, _verify(G1, G2, m)
    raise ValueError(name)


def _verify(G1, G2, m):
    if not m or len(m) != G1.number_of_nodes():
        return False
    for u, v in G1.edges():
        if not G2.has_edge(m.get(u), m.get(v)):
            return False
    return True


def _worker(name, fam, n, d, k, lb, seed, q):
    from generator import InstanceGenerator
    gen = InstanceGenerator(seed=seed)
    if fam == "F1":
        G = gen.generate_f1_base(n, d)
    elif fam == "F2":
        G = gen.generate_f2_base(n, d, k=k)
    elif fam == "F3":
        G = gen.generate_f3_base(n, d)
    elif fam == "F4":
        G = gen.generate_f4_base(n, d, label_bits=lb, base_type="F1")
    Giso, _ = gen.generate_iso_pair(G)

    baseline = _peak_rss_kb()
    # fine-grained sampler
    stop = threading.Event()
    peak_sampled = [baseline]
    try:
        import psutil
        proc = psutil.Process()

        def sample():
            while not stop.is_set():
                peak_sampled[0] = max(peak_sampled[0], proc.memory_info().rss // 1024)
                time.sleep(0.001)
        t = threading.Thread(target=sample, daemon=True); t.start()
    except Exception:
        t = None

    t0 = time.perf_counter()
    found, verified = _solve(name, G, Giso)
    dt = (time.perf_counter() - t0) * 1000
    stop.set()
    if t:
        t.join(timeout=0.1)

    peak_hwm = _peak_rss_kb()
    peak = max(peak_hwm, peak_sampled[0])
    q.put({"found": found, "verified": verified, "time_ms": dt,
           "baseline_kb": baseline, "peak_kb": peak,
           "delta_kb": max(0, peak - baseline)})


def profile(name, fam, n, d, k, lb, seed, timeout_s=120.0):
    """Run one (solver, instance) in a subprocess. If it exceeds timeout_s,
    kill it and return a timeout record (memory fields = None), so one
    pathological instance (VF2++ heavy tail) cannot stall the whole run."""
    q = Queue()
    p = Process(target=_worker, args=(name, fam, n, d, k, lb, seed, q))
    p.start()
    p.join(timeout_s)
    if p.is_alive():
        p.terminate()
        p.join()
        return {"found": False, "verified": False, "time_ms": timeout_s * 1000.0,
                "baseline_kb": None, "peak_kb": None, "delta_kb": None,
                "timeout_hit": True}
    try:
        res = q.get(timeout=5)
    except Exception:
        return {"found": False, "verified": False, "time_ms": None,
                "baseline_kb": None, "peak_kb": None, "delta_kb": None,
                "timeout_hit": False, "error": True}
    res["timeout_hit"] = False
    return res


DEFAULT_CELLS = [
    ("F1", 192, 6, 0, 0),
    ("F2", 192, 6, 5 * 192, 0),
    ("F2", 256, 10, 0, 0),
    ("F2", 256, 4, 2 * 256, 0),   # hardest observed
]
SOLVERS = ["vf2pp", "vf3", "nauty", "bliss"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=100)
    ap.add_argument("--out", default="results_memory.jsonl")
    ap.add_argument("--solvers", nargs="+", default=SOLVERS)
    ap.add_argument("--per-solve-timeout", type=float, default=120.0,
                    help="kill any single (solver, instance) run after this "
                         "many seconds; it is counted in the p_to column")
    args = ap.parse_args()

    agg = defaultdict(lambda: defaultdict(lambda: {"delta": [], "peak": [],
                                                   "time": [], "ver": 0, "to": 0}))
    with open(args.out, "w") as f:
        for (fam, n, d, k, lb) in DEFAULT_CELLS:
            cell = f"{fam} n={n} d={d} k={k}"
            for inst in range(args.instances):
                seed = 42 + inst + n * d + k + lb
                for s in args.solvers:
                    r = profile(s, fam, n, d, k, lb, seed,
                                timeout_s=args.per_solve_timeout)
                    rec = {"attack": s, "family": fam, "n": n, "d": d, "k": k,
                           "label_bits": lb, "instance_id": inst,
                           "time_ms": r["time_ms"], "peak_kb": r["peak_kb"],
                           "delta_kb": r["delta_kb"], "verified": r["verified"],
                           "timeout_hit": r.get("timeout_hit", False)}
                    f.write(json.dumps(rec) + "\n")
                    a = agg[cell][s]
                    if r.get("timeout_hit") or r.get("delta_kb") is None:
                        a["to"] += 1
                        continue
                    a["delta"].append(r["delta_kb"]); a["peak"].append(r["peak_kb"])
                    a["time"].append(r["time_ms"]); a["ver"] += int(r["verified"])
            print(f"done {cell}")

    print("\n" + "=" * 81)
    print(f"{'Cell':<22}{'solver':<8}{'solveMem(MB)':>13}{'peakRSS(MB)':>13}"
          f"{'time(ms)':>10}{'p_to':>7}{'ver%':>7}")
    print("-" * 81)
    for (fam, n, d, k, lb) in DEFAULT_CELLS:
        cell = f"{fam} n={n} d={d} k={k}"
        for s in args.solvers:
            a = agg[cell][s]
            dmb = statistics.median(a["delta"]) / 1024.0 if a["delta"] else float("nan")
            pmb = statistics.median(a["peak"]) / 1024.0 if a["peak"] else float("nan")
            tm = statistics.median(a["time"]) if a["time"] else float("nan")
            pto = a["to"] / args.instances
            vr = 100.0 * a["ver"] / args.instances
            print(f"{cell:<22}{s:<8}{dmb:>13.2f}{pmb:>13.1f}{tm:>10.2f}"
                  f"{pto:>7.2f}{vr:>6.0f}%")
        print()
    print("solveMem = median additional RSS attributable to the solver over the "
          "graph objects; peakRSS = absolute. Ordering is the portable finding.")


if __name__ == "__main__":
    main()
