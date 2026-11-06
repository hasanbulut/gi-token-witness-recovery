"""
runner.py  (corrected, 2026-07)

Changes vs. the submitted version:

1. map_verified IS NOW LOGGED. The submitted runner computed map_verified in
   run_a3/run_a4 and then silently dropped it, while logging "success" =
   map_found. Since the paper's security metric is witness RECOVERY, every
   solver record now carries map_found AND map_verified, and "success" is
   redefined as (map_found AND map_verified). WL-1 keeps its own semantics
   (indistinguishability), recorded as before.

2. TIMING SPLIT LOGGED. Every solver record carries solver_time_ms
   (in-child, around the solver call -- matches the manuscript's stated
   methodology) and end_to_end_time_ms (spawn/join inclusive). time_ms is
   kept equal to end_to_end_time_ms so existing plot scripts still run;
   the paper's tables should use solver_time_ms.

3. PAIRED HARDENING DESIGN. The base-graph seed no longer contains k or
   label_bits. For each (family, n, d, instance_id) ONE base graph is
   generated; all F2 swap levels are applied to copies of that same graph
   (k=0 is literally the base), all F4 label widths label copies of the
   same graph, and the SAME target permutation (perm_seed) is used across
   levels. Records carry base_graph_id so stats_analysis can run paired
   (Wilcoxon signed-rank) tests. Cell seeds are collision-free via the
   grid index (the old n*d term collided, e.g. 64*6 == 96*4).

4. BLISS TIER. A5 (igraph/BLISS) runs in the main grid with the same
   timeout ladder as VF3/nauty, so the strongest observed attacker is
   measured everywhere, not only on hardest cells.

5. REPRODUCIBILITY METADATA. results_meta.json records platform, Python,
   and library versions next to results.jsonl. runner refuses to append to
   an existing results.jsonl (the old behaviour silently duplicated
   records across re-runs).
"""

import json
import os
import platform
import sys

from config_parser import ExperimentConfig
from generator import InstanceGenerator
from attackers import AttackerPipeline

RESULTS_FILE = "results.jsonl"
META_FILE = "results_meta.json"


def append_to_jsonl(filepath, record):
    with open(filepath, 'a') as f:
        f.write(json.dumps(record) + '\n')


def write_meta():
    def ver(mod):
        try:
            m = __import__(mod)
            return getattr(m, '__version__', 'unknown')
        except Exception:
            return 'not-installed'
    meta = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "libraries": {m: ver(m) for m in
                      ("networkx", "numpy", "scipy", "pynauty",
                       "vf3py", "igraph", "yaml", "psutil")},
    }
    with open(META_FILE, 'w') as f:
        json.dump(meta, f, indent=2)


def run_warmup(pipeline, gen):
    for _ in range(10):
        G_base = gen.generate_f1_base(32, 4)
        G_iso, _ = gen.generate_iso_pair(G_base)
        pipeline.run_a2_vf2pp(G_base, G_iso, timeout_sec=0.1)
        pipeline.run_a3_vf3(G_base, G_iso, timeout_sec=0.1)
        pipeline.run_a4_nauty(G_base, G_iso, timeout_sec=0.1)
        pipeline.run_a5_bliss(G_base, G_iso, timeout_sec=0.1)


def classify(res, pair_type, timed_out=None):
    """Distinguish a real solver failure from a native crash, so an errored run
    is never silently counted as 'no witness found'."""
    if res.get("error"):
        return "solver_error"
    if timed_out if timed_out is not None else res["timeout_hit"]:
        return "timeout"
    if pair_type == "noniso":
        return "correct_noniso_rejection" if not res["map_found"] \
            else "invalid_witness"
    if res["map_found"] and res["map_verified"]:
        return "verified_witness"
    if res["map_found"] and not res["map_verified"]:
        return "invalid_witness"
    return "no_witness"


def solver_record(base_rec, attack, timeout_sec, res,
                  pair_type="iso", eps=0):
    rec = base_rec.copy()
    rec.update({
        "pair_type": pair_type, "eps": eps,
        "attack": attack, "timeout_sec": timeout_sec,
        "map_found": res["map_found"],
        "map_verified": res["map_verified"],
        "success": bool(res["map_found"] and res["map_verified"]),
        "solver_time_ms": res["solver_time_ms"],
        "end_to_end_time_ms": res["end_to_end_time_ms"],
        "time_ms": res["time_ms"],           # legacy = end-to-end
        "timeout_hit": res["timeout_hit"],
        "labels_enforced": res.get("labels_enforced", False),
        "status": classify(res, pair_type),
        "error": res.get("error"),
        "process_exitcode": res.get("process_exitcode"),
    })
    return rec


def emit_ladder(base_rec, attack, ladder, res, pair_type="iso", eps=0):
    """ONE measured run at max(ladder) -> one record per threshold, derived.

    Why: the old design re-executed the same solver on the same pair once per
    threshold (VF2++ x3, VF3/nauty/Bliss x4 = 15 forks per pair, 648k forks on
    the full grid). Running once at the ladder's own maximum and deriving the
    lower thresholds emits the SAME table with 3-4x fewer forks, and loses no
    information: every threshold at or below the cap is decidable from the
    measured time.

    It is also MORE correct. The old ladder enforced tau on p.join(), i.e. on
    process spawn + IPC + solve. At n=256 VF2++ solves in ~92 ms but the
    wrapper added ~52 ms (Manager 47 + fork), pushing it over tau = 0.1 s --
    so the SUBMITTED p_to(0.1) column reports timeouts that are wrapper
    artifacts, not solver behaviour. Deriving from solver_time_ms removes that
    contamination. Expect the tau = 0.1 column to move; that is the fix.

    Deliberately NOT a uniform 30 s cap (which an external review proposed):
    for VF2++'s pathological cells a single 30 s run costs MORE than the old
    ladder's 0.1+1+5 = 6.1 s truncation. Capping each solver at its own ladder
    max is cheaper in both regimes.
    """
    cap = max(ladder)
    st = res["solver_time_ms"]
    recs = []
    for tau in ladder:
        # exceeded if the run hit the cap (st is None) or solved slower than tau
        exceeded = (st is None) or (st > tau * 1000.0)
        rec = solver_record(base_rec, attack, tau, res, pair_type, eps)
        rec["timeout_hit"] = bool(exceeded)
        rec["success"] = bool(res["map_found"] and res["map_verified"]
                              and not exceeded)
        rec["status"] = classify(res, pair_type, timed_out=exceeded)
        rec["derived_from_single_run"] = True
        rec["measured_cap_sec"] = cap
        recs.append(rec)
    return recs


def run_full_experiment():
    config = ExperimentConfig("config.yaml")
    pipeline = AttackerPipeline()

    if os.path.exists(RESULTS_FILE):
        raise SystemExit(
            f"{RESULTS_FILE} already exists -- move it aside first. "
            "(The old runner silently appended, duplicating records.)")
    open(RESULTS_FILE, 'w').close()
    write_meta()

    base_seed = config.settings['random_seed']
    init_gen = InstanceGenerator(seed=base_seed)
    run_warmup(pipeline, init_gen)

    instances_per_cell = config.settings.get('instances_per_cell', 200)
    grid = config.generate_base_grid()
    families = config.families
    targeted_cells = [(128, 6), (192, 6), (256, 8)]

    timeouts_vf2 = [0.1, 1.0, 5.0]
    timeouts_strong = [0.1, 1.0, 5.0, 30.0]

    swap_multipliers = config.f_specific['F2']['swap_multipliers']
    label_bits_list = config.f_specific['F4']['label_bits']

    def run_all_solvers(base_rec, G_base, G_iso, is_targeted,
                        H_noniso, noisy_graphs):
        # A1 WL-1 runs in-process (no fork); unchanged semantics.
        res_wl = pipeline.run_a1_wl1(G_base, G_iso)
        rec_wl = base_rec.copy()
        rec_wl.update({"pair_type": "iso", "eps": 0, "attack": "wl1",
                       "timeout_sec": 0,
                       "success": not res_wl["wl_distinguished"],
                       "time_ms": 0})
        append_to_jsonl(RESULTS_FILE, rec_wl)

        # A2-A5: ONE measured run each, at that solver's own ladder max.
        for attack, fn, ladder in (
                ("vf2pp", pipeline.run_a2_vf2pp, timeouts_vf2),
                ("vf3", pipeline.run_a3_vf3, timeouts_strong),
                ("nauty", pipeline.run_a4_nauty, timeouts_strong),
                ("bliss", pipeline.run_a5_bliss, timeouts_strong)):
            res = fn(G_base, G_iso, timeout_sec=max(ladder))
            for rec in emit_ladder(base_rec, attack, ladder, res):
                append_to_jsonl(RESULTS_FILE, rec)

        if is_targeted:
            t_target = 5.0
            for attack, fn in (("nauty", pipeline.run_a4_nauty),
                               ("vf2pp", pipeline.run_a2_vf2pp),
                               ("vf3", pipeline.run_a3_vf3),
                               ("bliss", pipeline.run_a5_bliss)):
                res = fn(G_base, H_noniso, timeout_sec=t_target)
                append_to_jsonl(RESULTS_FILE,
                                solver_record(base_rec, attack, t_target,
                                              res, pair_type="noniso"))
            for eps, G_noisy in noisy_graphs.items():
                res = pipeline.run_a3_vf3(G_base, G_noisy,
                                          timeout_sec=t_target)
                append_to_jsonl(RESULTS_FILE,
                                solver_record(base_rec, "vf3", t_target,
                                              res, pair_type="noisy",
                                              eps=eps))

    for family in families:
        for cell_idx, (n, d) in enumerate(grid):
            if family == "F3" and n % 2 != 0:
                continue

            k_multipliers = swap_multipliers if family == "F2" else [0]
            l_bits_list = label_bits_list if family == "F4" else [0]
            is_targeted = (n, d) in targeted_cells

            for instance_id in range(instances_per_cell):
                # Base-graph seed: NO k, NO label_bits, collision-free cell
                # offset (old scheme used n*d, which collides: 64*6 == 96*4).
                graph_seed = (base_seed
                              + 1_000_003 * cell_idx
                              + instance_id)
                perm_seed = graph_seed + 500_000_011
                base_graph_id = f"{family}-{n}-{d}-{instance_id}"

                if family == "F3":
                    # F3 gets a DECOUPLED seed so its 2-lift root does not
                    # share F1's underlying random-regular graph. This makes
                    # the F1-vs-F3 cross-construction comparison a genuine
                    # independent two-sample test. (F1/F2/F4 deliberately
                    # share a root for the paired hardening tests.)
                    gen = InstanceGenerator(seed=graph_seed + 250_000_013)
                    G_root = gen.generate_f3_base(n, d)
                else:                       # F1, F2, F4 share an F1 root
                    gen = InstanceGenerator(seed=graph_seed)
                    G_root = gen.generate_f1_base(n, d)

                for k_mult in k_multipliers:
                    k_swaps = k_mult * n
                    for l_bits in l_bits_list:
                        # hardening applied to COPIES of the same root
                        G_base = G_root
                        if family == "F2":
                            G_base = InstanceGenerator.apply_swaps(
                                G_root, k_swaps,
                                swap_seed=graph_seed + 7 + k_mult)
                        elif family == "F4":
                            G_base = InstanceGenerator.assign_labels(
                                G_root, l_bits,
                                label_seed=graph_seed + 13 + l_bits)

                        # SAME permutation across hardening levels
                        G_iso, _ = InstanceGenerator.iso_pair_with_rng(
                            G_base, perm_seed)

                        H_noniso, noisy_graphs = None, {}
                        if is_targeted:
                            pair_gen = InstanceGenerator(
                                seed=graph_seed + 900_000_007)
                            H_noniso = pair_gen.generate_hard_noniso_pair(
                                G_base, num_swaps=5)
                            for eps in config.noise['epsilons']:
                                if eps > 0:
                                    noisy_graphs[eps] = (
                                        pair_gen.generate_noisy_iso(
                                            G_iso, eps))

                        base_rec = {
                            "family": family, "n": n, "d": d,
                            "k": k_swaps, "label_bits": l_bits,
                            "seed": graph_seed,
                            "base_graph_id": base_graph_id,
                            "instance_id": instance_id,
                        }
                        run_all_solvers(base_rec, G_base, G_iso,
                                        is_targeted, H_noniso, noisy_graphs)


if __name__ == "__main__":
    run_full_experiment()
    print("Tum deneyler tamamlandi ve sonuclar 'results.jsonl' "
          "dosyasina kaydedildi!")
