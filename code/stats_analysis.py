"""
stats_analysis.py  (corrected, 2026-07)

Distributional and significance analysis of witness-recovery timings.

Changes vs. the previous version:

1. PAIRED SWAP TEST (the reviewer's required redesign). The corrected
   runner generates ONE base graph per (family, n, d, instance_id) and
   applies every k level (and every label width) to copies of it with the
   same target permutation. Records carry instance_id/base_graph_id, so
   k = 0 vs k > 0 is now tested with the WILCOXON SIGNED-RANK test on
   per-instance timing differences -- isolating the effect of the swaps
   themselves. The same paired test is run for F4 label widths (0 bits vs
   the widest), which is the correctly-designed version of the label
   experiment now that solvers are label-aware. Matched-pairs rank-biserial
   correlation is reported as the effect size.

2. UNPAIRED FALLBACK. On a legacy results.jsonl (no shared base graphs,
   k baked into the seed) instances cannot be paired; the script detects
   this and falls back to Mann-Whitney U with an explicit warning that the
   design does not isolate the swap effect.

3. SOLVER TIME PREFERRED. solver_time_ms (in-child) is used when present,
   falling back to time_ms for old files; records with map_verified ==
   False are excluded wherever the field exists.

4. F1 vs F2(k=0) CAVEAT. F1 and F2 with k = 0 are the same graph
   construction and must not be treated as independent families in the
   cross-family test; F2 now enters that test only at its smallest k > 0,
   and the output says so.

Outputs:
  stats_summary.csv        per-cell N, median, T90, mean, std, 95% CIs
  stats_significance.txt   Kruskal-Wallis + paired Wilcoxon (swaps, labels)
  box_families.png         per-family distributions at the reference cell
  box_swaps.png            F2 swap-level distributions at the reference cell
"""

import argparse
import csv
import json
import math
import re
from collections import defaultdict

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def record_time(r):
    t = r.get("solver_time_ms")
    return float(t) if t is not None else float(r["time_ms"])


def load(path, attack="vf3", pair_type="iso", timeout_sec=5.0):
    """cells[key] -> dict {instance_id: (time_ms, base_graph_id)}; key =
    (family, n, d, k, label_bits). Instance ids enable pairing.

    Also returns census[key] = {total, verified, timeout, error} so summaries
    can SHOW the timeout/error rate rather than silently assuming it is zero.
    Timing distributions remain conditional on verified success (that is what
    the medians describe); the census makes the conditioning explicit."""
    cells = defaultdict(dict)
    census = defaultdict(lambda: {"total": 0, "verified": 0,
                                  "timeout": 0, "error": 0})
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("attack") != attack:
            continue
        if r.get("pair_type", "iso") != pair_type:
            continue
        if timeout_sec is not None and \
                r.get("timeout_sec") not in (timeout_sec, None):
            continue
        key = (r["family"], r["n"], r["d"],
               r.get("k", 0), r.get("label_bits", 0))
        c = census[key]
        c["total"] += 1
        if r.get("status") == "solver_error" or r.get("error"):
            c["error"] += 1
            continue
        if r.get("timeout_hit"):
            c["timeout"] += 1
            continue
        if not r.get("success", False):
            continue
        if "map_verified" in r and not r["map_verified"]:
            continue
        c["verified"] += 1
        cells[key][r.get("instance_id", len(cells[key]))] = (
            record_time(r), r.get("base_graph_id"))
    return cells, dict(census)


def times(cells, key):
    return [t for (t, _) in cells[key].values()]


# --------------------------------------------------------------------------
# Summary statistics (unchanged outputs, new input shape)
# --------------------------------------------------------------------------

def bootstrap_ci(x, stat_fn, n_boot=10000, alpha=0.05, seed=42):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boot = stat_fn(x[idx], axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def p90(a, axis=None):
    return np.percentile(a, 90, axis=axis)


def summarize(cells, census=None, out_csv="stats_summary.csv"):
    census = census or {}
    rows = []
    for key in sorted(cells):
        fam, n, d, k, lb = key
        x = np.asarray(times(cells, key), dtype=float)
        med_ci = bootstrap_ci(x, np.median)
        t90_ci = bootstrap_ci(x, p90)
        c = census.get(key, {"total": len(x), "verified": len(x),
                             "timeout": 0, "error": 0})
        rows.append({
            "family": fam, "n": n, "d": d, "k": k, "label_bits": lb,
            "N": len(x),
            "N_total": c["total"],
            "N_timeout": c["timeout"],
            "N_error": c["error"],
            "p_timeout": round(c["timeout"] / c["total"], 4)
            if c["total"] else 0.0,
            "median_ms": round(float(np.median(x)), 3),
            "median_ci_lo": round(med_ci[0], 3),
            "median_ci_hi": round(med_ci[1], 3),
            "T90_ms": round(float(np.percentile(x, 90)), 3),
            "T90_ci_lo": round(t90_ci[0], 3),
            "T90_ci_hi": round(t90_ci[1], 3),
            "mean_ms": round(float(np.mean(x)), 3),
            "std_ms": round(float(np.std(x, ddof=1)), 3),
            "iqr_ms": round(float(np.percentile(x, 75)
                                  - np.percentile(x, 25)), 3),
        })
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


# --------------------------------------------------------------------------
# Effect sizes
# --------------------------------------------------------------------------


def holm_adjust(pvals):
    """Holm-Bonferroni step-down with the mandatory cumulative-maximum step,
    which the previous inline version omitted (making adjusted p-values
    anti-conservative). Returns adjusted p-values in the input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, pvals[idx] * (m - rank)))
        adj[idx] = running
    return adj


def rank_biserial_unpaired(a, b):
    U, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
    return 1.0 - (2.0 * U) / (len(a) * len(b))


def rank_biserial_paired(diffs):
    """Matched-pairs rank-biserial from signed ranks of the differences."""
    d = np.asarray([x for x in diffs if x != 0], dtype=float)
    if len(d) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    r_pos = ranks[d > 0].sum()
    r_neg = ranks[d < 0].sum()
    return float((r_pos - r_neg) / ranks.sum())


def _effect_label(es):
    a = abs(es)
    return ("negligible" if a < 0.1 else "small" if a < 0.3
            else "medium" if a < 0.5 else "large")


# --------------------------------------------------------------------------
# Paired hardening tests (the redesign)
# --------------------------------------------------------------------------

def paired_test(cells, key_a, key_b, name_a, name_b, min_pairs=6):
    """Wilcoxon signed-rank on per-instance differences between two cells
    of the same (family-root, n, d). Falls back to Mann-Whitney with a
    warning when instances cannot be paired (legacy results files)."""
    lines = []
    da, db = cells.get(key_a, {}), cells.get(key_b, {})
    # pair only instances whose records prove a SHARED base graph; legacy
    # files reuse instance ids across k levels but on different base graphs
    common = sorted(i for i in set(da) & set(db)
                    if da[i][1] is not None and da[i][1] == db[i][1])

    if len(common) >= min_pairs:
        a = np.array([da[i][0] for i in common])
        b = np.array([db[i][0] for i in common])
        diffs = b - a
        med_shift = float(np.median(diffs))
        if np.all(diffs == 0):
            lines.append(f"  {name_a} vs {name_b}: all {len(common)} paired "
                         "differences are zero (identical timings).")
            return lines
        W, p = stats.wilcoxon(a, b, alternative="two-sided",
                              zero_method="wilcox")
        es = rank_biserial_paired(diffs)
        lines.append(f"  PAIRED Wilcoxon signed-rank, {len(common)} pairs "
                     f"(same base graph, same target permutation):")
        lines.append(f"    {name_a}: median {np.median(a):.2f} ms | "
                     f"{name_b}: median {np.median(b):.2f} ms | "
                     f"median per-instance shift {med_shift:+.2f} ms")
        lines.append(f"    W = {W:.1f}, p = {p:.4g}  "
                     f"({'significant' if p < 0.05 else 'not significant'} "
                     "at 0.05)")
        lines.append(f"    matched-pairs rank-biserial = {es:+.3f} "
                     f"({_effect_label(es)})")
    elif da and db:
        a = [t for (t, _) in da.values()]
        b = [t for (t, _) in db.values()]
        U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        es = rank_biserial_unpaired(a, b)
        lines.append(f"  WARNING: only {len(common)} instances share a "
                     "verifiable base_graph_id -- cannot pair (legacy "
                     "results file: k was baked into the seed, so levels "
                     "used different base graphs).")
        lines.append("  Falling back to UNPAIRED Mann-Whitney; this design "
                     "does NOT isolate the hardening effect, because the "
                     "compared levels were built on different base graphs.")
        lines.append(f"    {name_a}: median {np.median(a):.2f} ms | "
                     f"{name_b}: median {np.median(b):.2f} ms")
        lines.append(f"    U = {U:.1f}, p = {p:.4g}, "
                     f"rank-biserial = {es:+.3f} ({_effect_label(es)})")
    else:
        lines.append(f"  {name_a} vs {name_b}: insufficient data.")
    return lines



def paired_family_block(cells, base_key, level_keys, base_name, level_names,
                        header):
    """Run paired_test for every hardening level of one family and Holm-adjust
    the p-values ACROSS the levels of that family (the multiplicity that
    actually exists here). Falls back gracefully when a level is unpaired."""
    lines = [header]
    # collect p-values from paired comparisons that were actually paired
    results = []  # (level_name, key, raw_lines, p_or_None)
    for lk, ln in zip(level_keys, level_names):
        sub = paired_test(cells, base_key, lk, base_name, ln)
        p = None
        for row in sub:
            m = re.search(r"[Wp]\s*=\s*[\d.eE+-]+.*?p\s*=\s*([\d.eE+-]+)",
                          row)
            if not m:
                m = re.search(r"\bp\s*=\s*([\d.eE+-]+)", row)
            if m:
                try:
                    p = float(m.group(1))
                except ValueError:
                    p = None
                break
        results.append((ln, sub, p))
    ps = [p for (_, _, p) in results if p is not None]
    adj = holm_adjust(ps) if ps else []
    ai = 0
    for ln, sub, p in results:
        lines.extend(sub)
        if p is not None:
            hp = adj[ai]; ai += 1
            verdict = "significant" if hp < 0.05 else "not significant"
            lines.append(f"    Holm-adjusted p (within family) = {hp:.4g} "
                         f"({verdict} at 0.05)")
    return lines


def significance(cells, ref_cell, attack="vf3",
                 out_txt="stats_significance.txt"):
    n, d = ref_cell
    lines = [f"Significance tests at reference cell n={n}, d={d}", ""]

    # ---- Cross-construction: F1 vs F3 only (independent groups) ---------
    # F2(k>0) and F4(bits>0) are derived from the SAME F1 base graph (shared
    # graph_seed) and are therefore statistically dependent on F1; an
    # independent-groups test (Kruskal-Wallis / Mann-Whitney across families)
    # would violate its independence assumption. They are handled by the
    # PAIRED tests below. F3 now uses a decoupled seed offset in runner.py, so
    # F1 vs F3 is a genuine independent two-sample comparison.
    f1 = times(cells, ("F1", n, d, 0, 0)) if ("F1", n, d, 0, 0) in cells else []
    f3 = times(cells, ("F3", n, d, 0, 0)) if ("F3", n, d, 0, 0) in cells else []
    if len(f1) >= 5 and len(f3) >= 5:
        U, p = stats.mannwhitneyu(f1, f3, alternative="two-sided")
        es = rank_biserial_unpaired(f1, f3)
        lines.append("Cross-construction F1 vs F3 (independent two-sample "
                     "Mann-Whitney; the only genuinely independent family "
                     "comparison -- F2/F4 share F1's base graph and are tested "
                     "paired below):")
        lines.append(f"  F1: median {np.median(f1):.2f} ms (N={len(f1)}) | "
                     f"F3: median {np.median(f3):.2f} ms (N={len(f3)})")
        sig = 'significant' if p < 0.05 else 'not significant'
        lines.append(f"  U = {U:.1f}, p = {p:.4g}  ({sig} at 0.05), "
                     f"rank-biserial = {es:+.3f} ({_effect_label(es)})")
        lines.append("")

    # ---- Paired swap effect (R1 Sec. 4.4, redesigned) -------------------
    f2_ks = sorted(k for (fam, nn, dd, k, lb) in cells
                   if fam == "F2" and nn == n and dd == d and lb == 0)
    if f2_ks and 0 in f2_ks and len(f2_ks) > 1:
        k_his = [k for k in f2_ks if k > 0]
        lines += paired_family_block(
            cells, ("F2", n, d, 0, 0),
            [("F2", n, d, k, 0) for k in k_his],
            "k=0", [f"k={k}" for k in k_his],
            "Degree-preserving swap effect in F2 (paired: every k level "
            "applied to the SAME base graph with the SAME target "
            "permutation; p-values Holm-adjusted across k levels):")
        lines.append("  -> states whether the swap delay is a real effect "
                     "or noise, with the base graph held fixed as R1 "
                     "Sec. 4.4 requires. NOTE: at large N a sub-millisecond "
                     "shift can be 'significant'; report the EFFECT SIZE / "
                     "median shift, not the p-value, as the security claim.")
        lines.append("")

    # ---- Paired label effect (F4, label-aware solvers) -------------------
    f4_lbs = sorted(lb for (fam, nn, dd, k, lb) in cells
                    if fam == "F4" and nn == n and dd == d and k == 0)
    if f4_lbs and 0 in f4_lbs and len(f4_lbs) > 1:
        lb_his = [lb for lb in f4_lbs if lb > 0]
        lines += paired_family_block(
            cells, ("F4", n, d, 0, 0),
            [("F4", n, d, 0, lb) for lb in lb_his],
            "0-bit", [f"{lb}-bit" for lb in lb_his],
            "Exposed-label effect in F4 (paired, label-aware matching "
            "enforced by all solvers; nested label masks isolate width; "
            "p-values Holm-adjusted across label widths):")
        lines.append("  -> valid only on results produced by the corrected "
                     "label-aware pipeline; on the old pipeline labels were "
                     "ignored and any difference was noise.")
        if attack == "vf3":
            lines.append("  CAVEAT (vf3): vf3py enforces labels via a Python "
                         "node_match callback, so its labelled timings "
                         "include FFI overhead. Interpret the label effect "
                         "via the native-colour solvers (bliss, nauty) or "
                         "vf2pp; the attacker takes the minimum over "
                         "solvers, and for those labels ACCELERATE recovery.")
        lines.append("")

    text = "\n".join(lines)
    with open(out_txt, "w") as f:
        f.write(text + "\n")
    return text


# --------------------------------------------------------------------------
# Box plots (matplotlib <=3.7 compatible)
# --------------------------------------------------------------------------

def boxplots(cells, ref_cell, prefix=""):
    n, d = ref_cell
    # Only F1 and F3 are distinct CONSTRUCTIONS at base hardening: F2(k=0) and
    # F4(0-bit) are the same graph as F1 (shared base seed), so plotting all
    # four as "families" would show one distribution three times.
    fams, data = [], []
    for fam in ("F1", "F3"):
        key = (fam, n, d, 0, 0)
        if key in cells:
            fams.append(fam)
            data.append(times(cells, key))
    if data:
        plt.figure(figsize=(6, 4))
        plt.boxplot(data, showmeans=True)
        plt.xticks(range(1, len(fams) + 1), fams)
        plt.ylabel("mapping-recovery solver time (ms)")
        plt.title(f"Distinct constructions at base hardening "
                  f"(n={n}, d={d}); F2/F4 appear in the paired plots")
        plt.tight_layout()
        plt.savefig(f"{prefix}box_families.png", dpi=150)
        plt.close()

    f2_ks = sorted(k for (fam, nn, dd, k, lb) in cells
                   if fam == "F2" and nn == n and dd == d and lb == 0)
    if len(f2_ks) > 1:
        data = [times(cells, ("F2", n, d, k, 0)) for k in f2_ks]
        plt.figure(figsize=(6, 4))
        plt.boxplot(data, showmeans=True)
        plt.xticks(range(1, len(f2_ks) + 1), [f"k={k}" for k in f2_ks])
        plt.ylabel("mapping-recovery solver time (ms)")
        plt.title(f"F2 swap hardening (n={n}, d={d}, paired instances)")
        plt.tight_layout()
        plt.savefig(f"{prefix}box_swaps.png", dpi=150)
        plt.close()


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="path to results.jsonl")
    ap.add_argument("--attack", default="vf3")
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--ref-n", type=int, default=192)
    ap.add_argument("--ref-d", type=int, default=6)
    ap.add_argument("--out-prefix", default=None,
                    help="prefix for output files; defaults to the solver "
                         "name so per-solver runs do not overwrite each other")
    args = ap.parse_args()
    prefix = (args.out_prefix if args.out_prefix is not None
              else f"{args.attack}_")
    if prefix and not prefix.endswith(("_", "-", ".")):
        prefix += "_"

    cells, census = load(args.results, attack=args.attack,
                         timeout_sec=args.timeout)
    if not cells:
        print("No matching records. Check --attack/--timeout and file schema.")
        return
    rows = summarize(cells, census,
                     out_csv=f"{prefix}stats_summary.csv")
    total_to = sum(c["timeout"] for c in census.values())
    total_err = sum(c["error"] for c in census.values())
    print(f"Wrote {prefix}stats_summary.csv ({len(rows)} cells). "
          f"Timeouts across all cells: {total_to}; solver errors: {total_err} "
          "(timing summaries are conditional on verified success).")
    for r in rows:
        print(f"  {r['family']} n={r['n']} d={r['d']} k={r['k']} "
              f"lb={r['label_bits']} | N={r['N']:>3} "
              f"med={r['median_ms']:.2f} "
              f"[{r['median_ci_lo']:.2f},{r['median_ci_hi']:.2f}] "
              f"T90={r['T90_ms']:.2f} std={r['std_ms']:.2f}")
    print()
    print(significance(cells, (args.ref_n, args.ref_d),
                       attack=args.attack,
                       out_txt=f"{prefix}stats_significance.txt"))
    boxplots(cells, (args.ref_n, args.ref_d), prefix=prefix)
    print(f"\nWrote {prefix}box_families.png, {prefix}box_swaps.png, "
          f"{prefix}stats_significance.txt.")


if __name__ == "__main__":
    main()
