"""
cost_per_byte.py  (corrected, 2026-07)

Answers Reviewer #1 (Sec. 4.1): attacker-cost-per-byte efficiency metrics.

Changes vs. the previous version:

1. GROUPING KEY FIXED. The old key (family, n, d) pooled all F2 swap levels
   and all F4 label widths into one bucket, so the "base hardening cost"
   mixed hardened and unhardened instances. The key is now
   (family, n, d, k, label_bits), and by default rows are FILTERED to base
   hardening (k == 0 and label_bits == 0). --include-hardening keeps every
   hardening level as its own row instead.

2. BYTE FORMULA FIXED FOR n > 256. B = n*d assumed one byte per edge
   endpoint, valid only while vertex ids fit in 8 bits. The footprint is now

     B_graph = n*d * ceil(log2(n)/8)  +  ceil(n*label_bits/8)

   (edge-list endpoints at the required id width, plus packed label storage
   for F4), before any metadata/framing overhead. For n <= 256 with no
   labels this reduces exactly to the paper's original n*d, so published
   small-grid numbers are unchanged; for n = 384..1024 endpoints cost
   2 bytes and the old formula under-counted by 2x.

3. SOLVER TIME PREFERRED. With the corrected runner, records carry
   solver_time_ms (in-child, matches the manuscript's stated methodology);
   it is used when present, falling back to time_ms for old files. Records
   are also required to have map_verified (when the field exists), so
   timings of unverified witnesses never enter the medians.

4. REPORTING LEADS WITH THE SCALING EXPONENT. The log-log exponent
   T ~ B^p is the defensible headline (RUNBOOK note 7); the linear
   "bytes per extra ms" fit is reported as a marginal-cost illustration,
   and the long-range extrapolation to 1 s / 10 s budgets is printed ONLY
   when the linear fit is adequate (R^2 >= 0.8) -- a low-R^2 linear fit
   must not be used for long-range security projections. When suppressed,
   the observed ms/byte range across the measured grid is reported instead,
   which is what the reviewer literally asked for.

5. --paper DEPRECATED. The embedded submitted-manuscript medians predate
   the corrected pipeline (wrapper-inclusive timing, unverified nauty).
   They still print for comparison but with a loud warning; the definitive
   tables must come from --results on the corrected results.jsonl.
"""

import argparse
import json
import math
import statistics
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Submitted-manuscript medians (Sec. 4.1). DEPRECATED: produced by the
# pre-correction pipeline (timing included process spawn/join). Kept only
# for before/after comparison; never mix with corrected-run numbers.
PAPER_TABLE = [
    ("F1", 64, 4, 0, 0, 10.6),
    ("F1", 96, 6, 0, 0, 11.8),
    ("F1", 128, 6, 0, 0, 14.8),
    ("F1", 192, 6, 0, 0, 19.7),
    ("F2", 192, 6, 0, 0, 22.9),
    ("F3", 192, 8, 0, 0, 18.1),
    ("F2", 256, 10, 0, 0, 20.2),
]


def byte_footprint(n, d, label_bits=0, pair=False):
    """Edge-list token bytes with id-width awareness + packed labels.

    endpoints: n*d endpoint slots (n*d/2 edges x 2), each ceil(log2(n)/8)
    bytes wide; labels: ceil(n*label_bits/8) bytes. Reduces to n*d for
    n <= 256, label_bits == 0 (the paper's original small-grid formula).
    """
    id_width = max(1, math.ceil(math.log2(n) / 8))
    b = n * d * id_width + math.ceil(n * label_bits / 8)
    return 2 * b if pair else b


def record_time(r):
    """Prefer in-child solver time (corrected schema); fall back to legacy."""
    t = r.get("solver_time_ms")
    return float(t) if t is not None else float(r["time_ms"])


def from_results(path, attack="vf3", timeout_sec=5.0,
                 include_hardening=False):
    cells = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("attack") != attack or r.get("pair_type", "iso") != "iso":
                continue
            if r.get("timeout_hit") or not r.get("success", False):
                continue
            # corrected schema: only verified witnesses count
            if "map_verified" in r and not r["map_verified"]:
                continue
            if timeout_sec is not None and \
                    r.get("timeout_sec") not in (timeout_sec, None):
                continue
            k = r.get("k", 0)
            lb = r.get("label_bits", 0)
            if not include_hardening and (k != 0 or lb != 0):
                continue
            key = (r["family"], r["n"], r["d"], k, lb)
            cells[key].append(record_time(r))
    return [(fam, n, d, k, lb, statistics.median(ts))
            for (fam, n, d, k, lb), ts in sorted(cells.items())]


def analyze(rows, pair=False, target_ms=(1000.0, 10000.0),
            fig="cost_per_byte_eff.png", r2_gate=0.8):
    label = ("B_pair" if pair else "B_graph") + \
        " = nd*ceil(log2(n)/8) + ceil(n*lb/8)" + (" x2" if pair else "")
    print(f"Byte footprint: {label}\n")

    hdr = (f"{'Config':<22}{'B(bytes)':>9}{'T_med(ms)':>10}"
           f"{'ms/byte':>10}{'bytes/ms':>10}")
    print(hdr)
    print("-" * len(hdr))
    Bs, Ts = [], []
    for fam, n, d, k, lb, t in rows:
        B = byte_footprint(n, d, lb, pair=pair)
        Bs.append(B)
        Ts.append(t)
        cfg = f"{fam} {n}/{d}" + (f" k={k}" if k else "") + \
              (f" lb={lb}" if lb else "")
        print(f"{cfg:<22}{B:>9}{t:>10.2f}{t / B:>10.5f}{B / t:>10.2f}")

    Bs = np.array(Bs, float)
    Ts = np.array(Ts, float)

    # ---- ITEM 3: deduplicate identical baseline configurations ----------
    # F1, F2(k=0) and F4(0-bit) are the SAME graph at the SAME byte count
    # (shared base graph, no hardening); their medians differ only by timing
    # jitter, so keying on the measured time would fail to collapse them.
    #
    # F3 MUST NOT be collapsed with them: it is a genuinely different 2-lift
    # construction that merely also has k == 0 and lb == 0. Keying purely on
    # (k == 0 and lb == 0) swallowed F3 and left a single fitting point,
    # invalidating the scaling exponent.
    from collections import defaultdict as _dd
    groups = _dd(list)
    order_seen = []
    for i, (fam, n, d, k, lb, t) in enumerate(rows):
        if fam in ("F1", "F2", "F4") and k == 0 and lb == 0:
            struct = ("F1_BASE", n, d)      # the shared random-regular root
        else:
            struct = (fam, n, d, k, lb)     # F3, and every hardened cell
        if struct not in groups:
            order_seen.append(struct)
        groups[struct].append(i)
    Bs2, Ts2 = [], []
    collapsed = 0
    for struct in order_seen:
        idxs = groups[struct]
        if len(idxs) > 1:
            collapsed += len(idxs) - 1
        Bs2.append(float(np.mean([Bs[j] for j in idxs])))
        Ts2.append(float(np.median([Ts[j] for j in idxs])))
    if collapsed:
        print(f"[dedup] collapsed {collapsed} duplicate baseline row(s) "
              "(F1 / F2 k=0 / F4 0-bit share one graph) to a single fitting "
              f"point; {len(order_seen)} distinct points remain (F3 kept "
              "separate).\n")
    Bs = np.array(Bs2, float)
    Ts = np.array(Ts2, float)

    # ---- headline: log-log scaling exponent -----------------------------
    p, _ = np.polyfit(np.log(Bs), np.log(Ts), 1)
    print(f"\nHEADLINE  log-log scaling: T_med ~ B^{p:.3f} "
          f"({'sub-linear' if p < 1 else 'supra-linear'} in token bytes)")
    if p < 1:
        print("  => each additional token byte buys the defender LESS delay "
              "than the last; byte spending has diminishing security returns.")

    # ---- secondary: linear marginal cost, extrapolation gated on R^2 ----
    a, b = np.polyfit(Bs, Ts, 1)
    pred = a * Bs + b
    ss_tot = np.sum((Ts - Ts.mean()) ** 2)
    r2 = 1 - np.sum((Ts - pred) ** 2) / ss_tot if ss_tot > 0 else float("nan")
    print(f"\nLinear marginal fit  T_med = {a:.5f}*B + {b:.2f}   (R^2={r2:.3f})")
    if a > 0:
        print(f"  within the measured range: {a:.5f} ms per extra byte "
              f"=> ~{1.0 / a:,.0f} bytes per extra ms of attacker time")

    if not math.isnan(r2) and r2 >= r2_gate and a > 0:
        print("\nExtrapolated bytes for target attacker budgets "
              f"(linear fit, R^2={r2:.2f} >= {r2_gate}):")
        B_max = Bs.max()
        for tgt in target_ms:
            B_needed = (tgt - b) / a
            print(f"  T_med = {tgt / 1000:.0f} s : ~{B_needed:,.0f} bytes "
                  f"(~{B_needed / 1024:,.0f} KB), "
                  f"~{(B_needed - B_max) / 1024:,.0f} KB beyond the largest "
                  f"tested token")
    else:
        print(f"\nExtrapolation SUPPRESSED (R^2={r2:.2f} < {r2_gate}): a weak "
              "linear fit must not back long-range security projections.")
        print("Report the OBSERVED efficiency instead (what R1 asked for):")
        eff = Ts / Bs
        print(f"  attacker cost per token byte across the measured grid: "
              f"{eff.min():.5f} - {eff.max():.5f} ms/byte "
              f"(median {np.median(eff):.5f})")
        print(f"  equivalently {1 / eff.max():,.0f} - {1 / eff.min():,.0f} "
              "bytes per ms of attacker delay")
        # Report ONLY what the grid measured -- no projection to a 1 s budget,
        # which the measured range never reaches.
        print(f"  => across ALL measured token sizes, the highest observed "
              f"median attacker cost was {Ts.max():.2f} ms "
              f"(token {int(Bs[np.argmax(Ts)])} B) -- far below one second. "
              "This is stated directly from the measurements; no long-range "
              "extrapolation is used.")

    # ---- figure ----------------------------------------------------------
    order = np.argsort(Bs)
    plt.figure(figsize=(6, 4))
    plt.plot(Bs[order], (Bs / Ts)[order], "o-")
    plt.xlabel(f"token bytes ({'pair' if pair else 'one graph'}, "
               "id-width aware)")
    plt.ylabel("defender bytes per ms of attacker delay")
    plt.title("Cost of security: bytes per ms of attacker resistance")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig, dpi=150)
    plt.close()
    print(f"\nWrote {fig}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None,
                    help="corrected runner.py results.jsonl (the definitive "
                         "source; required for paper tables)")
    ap.add_argument("--attack", default="vf3")
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--pair", action="store_true",
                    help="pair-level footprint (2 graphs)")
    ap.add_argument("--include-hardening", action="store_true",
                    help="keep every (k, label_bits) level as its own row "
                         "instead of filtering to base hardening")
    ap.add_argument("--paper", action="store_true",
                    help="DEPRECATED: submitted-manuscript medians, "
                         "pre-correction pipeline; comparison only")
    args = ap.parse_args()

    if args.results and not args.paper:
        rows = from_results(args.results, args.attack, args.timeout,
                            include_hardening=args.include_hardening)
        if not rows:
            print("No matching records in results file.")
            return
        scope = ("all hardening levels" if args.include_hardening
                 else "base hardening only (k=0, label_bits=0)")
        print(f"Using {len(rows)} cells from {args.results} "
              f"(attack={args.attack}, {scope}).\n")
    else:
        print("*" * 72)
        print("WARNING: using DEPRECATED submitted-manuscript medians. These "
              "were produced\nby the pre-correction pipeline (wrapper-"
              "inclusive timing) and MUST NOT be\nmixed with corrected-run "
              "numbers or used in the revised tables.")
        print("*" * 72 + "\n")
        rows = PAPER_TABLE

    analyze(rows, pair=args.pair)


if __name__ == "__main__":
    main()
