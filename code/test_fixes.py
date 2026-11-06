import networkx as nx, random, time
from generator import InstanceGenerator
from attackers import AttackerPipeline, graphs_are_labelled

pipe = AttackerPipeline()
FAIL = []

def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'}  {name}")
    if not cond: FAIL.append(name)

# ---- 1. Witness correctness, all solvers, unlabelled -------------------
gen = InstanceGenerator(seed=101)
G = gen.generate_f1_base(96, 6)
Giso, true_map = gen.generate_iso_pair(G)
for name, fn in (("vf2pp", pipe.run_a2_vf2pp), ("vf3", pipe.run_a3_vf3),
                 ("nauty", pipe.run_a4_nauty), ("bliss", pipe.run_a5_bliss)):
    r = fn(G, Giso, timeout_sec=30)
    check(f"{name} unlabelled: found+verified, solver_time recorded",
          r["map_found"] and r["map_verified"] and r["solver_time_ms"] is not None)

# ---- 2. nauty over 20 pairs (the empirical 0/20 -> 20/20 claim) --------
ok = 0
for t in range(20):
    g2 = InstanceGenerator(seed=200+t)
    Gt = g2.generate_f1_base(64, 6)
    Gi, _ = g2.generate_iso_pair(Gt)
    r = pipe.run_a4_nauty(Gt, Gi, timeout_sec=30)
    ok += r["map_verified"]
check(f"nauty corrected: {ok}/20 strict-verified", ok == 20)

# ---- 3. Label-aware matching (F4) ---------------------------------------
g3 = InstanceGenerator(seed=303)
GL = g3.generate_f4_base(96, 6, label_bits=4)
GLiso, _ = g3.generate_iso_pair(GL)   # relabel_nodes preserves attrs
check("label detection", graphs_are_labelled(GL, GLiso))
for name, fn in (("vf2pp", pipe.run_a2_vf2pp), ("vf3", pipe.run_a3_vf3),
                 ("nauty", pipe.run_a4_nauty), ("bliss", pipe.run_a5_bliss)):
    r = fn(GL, GLiso, timeout_sec=30)
    lbl_ok = r["map_verified"] and r["labels_enforced"] and all(
        GL.nodes[u]['label'] == GLiso.nodes[v]['label']
        for u, v in r["mapping"].items())
    check(f"{name} labelled: verified witness respects labels", lbl_ok)

# ---- 3b. Labels actually constrain: iso-as-graphs but label-incompatible ----
GA = nx.cycle_graph(6)
GB = nx.cycle_graph(6)
for v in GA.nodes(): GA.nodes[v]['label'] = v % 2       # 3 zeros, 3 ones
for v in GB.nodes(): GB.nodes[v]['label'] = 0 if v < 4 else 1  # 4 zeros, 2 ones
for name, fn in (("vf2pp", pipe.run_a2_vf2pp), ("vf3", pipe.run_a3_vf3),
                 ("nauty", pipe.run_a4_nauty), ("bliss", pipe.run_a5_bliss)):
    r = fn(GA, GB, timeout_sec=10)
    check(f"{name} rejects structure-iso but label-incompatible pair",
          not r["success"] and not r["map_verified"])

# ---- 4. Strict verifier catches the OLD broken nauty witness ------------
import pynauty
from attackers import networkx_to_pynauty
def old_broken_nauty(G1, G2):
    g1, m1 = networkx_to_pynauty(G1); g2, m2 = networkx_to_pynauty(G2)
    c1 = pynauty.canon_label(g1); c2 = pynauty.canon_label(g2)
    inv2 = [0]*len(c2)
    for i, c in enumerate(c2): inv2[c] = i
    wl = [inv2[c1[i]] for i in range(len(c1))]
    r1 = {v: k for k, v in m1.items()}; r2 = {v: k for k, v in m2.items()}
    return {r1[i]: r2[wl[i]] for i in range(len(wl))}
bad = old_broken_nauty(G, Giso)
check("strict verify_mapping rejects old broken nauty witness",
      not pipe.verify_mapping(G, Giso, bad))
check("strict verify_mapping accepts the true permutation",
      pipe.verify_mapping(G, Giso, true_map))

# ---- 5. VF3 first-witness: fast on high-automorphism graph --------------
C = nx.cycle_graph(200)   # |Aut| = 400; old list(...) enumerates all 400
Ciso, _ = InstanceGenerator.iso_pair_with_rng(C, 42)
r = pipe.run_a3_vf3(C, Ciso, timeout_sec=30)
check(f"vf3 first-witness on C_200 (|Aut|=400): verified, "
      f"solver {r['solver_time_ms']:.1f} ms",
      r["map_verified"] and r["solver_time_ms"] < 5000)

# ---- 6. Timing split sanity ----------------------------------------------
r = pipe.run_a3_vf3(G, Giso, timeout_sec=30)
check(f"solver_time ({r['solver_time_ms']:.2f} ms) < end_to_end "
      f"({r['end_to_end_time_ms']:.2f} ms); overhead "
      f"{r['end_to_end_time_ms']-r['solver_time_ms']:.1f} ms",
      0 < r["solver_time_ms"] < r["end_to_end_time_ms"])

# ---- 7. Paired design ----------------------------------------------------
root = InstanceGenerator(seed=777).generate_f1_base(64, 4)
k0 = InstanceGenerator.apply_swaps(root, 0, swap_seed=1)
k2a = InstanceGenerator.apply_swaps(root, 128, swap_seed=9)
k2b = InstanceGenerator.apply_swaps(root, 128, swap_seed=9)
check("k=0 copy identical to base", set(k0.edges()) == set(root.edges()))
check("swaps deterministic given seed", set(k2a.edges()) == set(k2b.edges()))
check("k=128 differs from base", set(k2a.edges()) != set(root.edges()))
i1, m1 = InstanceGenerator.iso_pair_with_rng(root, 55)
i2, m2 = InstanceGenerator.iso_pair_with_rng(k2a, 55)
check("same permutation reused across hardening levels", m1 == m2)
l4 = InstanceGenerator.assign_labels(root, 4, label_seed=3)
l8 = InstanceGenerator.assign_labels(root, 8, label_seed=3)
check("label levels share base structure",
      set(l4.edges()) == set(root.edges()) == set(l8.edges()))
check("assign_labels does not mutate input",
      not any('label' in root.nodes[v] for v in root.nodes()))

# ---- 8. Timeout path ------------------------------------------------------
big = InstanceGenerator(seed=88).generate_f1_base(256, 8)
bigiso, _ = InstanceGenerator.iso_pair_with_rng(big, 5)
r = pipe.run_a2_vf2pp(big, bigiso, timeout_sec=0.001)
check("timeout: hit flag set, solver_time None, not verified",
      r["timeout_hit"] and r["solver_time_ms"] is None and not r["map_verified"])


# ======================================================================
# v3 / v4 CORRECTIONS  (previously verified ad hoc; now protected here)
# ======================================================================

# ---- v3.1  Holm cumulative-max (anti-conservative without it) -----------
from stats_analysis import holm_adjust
raw = [0.001, 0.30, 0.30, 0.30, 0.30, 0.30]
adj = holm_adjust(raw)
check("v3 Holm: cumulative-max applied (ties collapse to 1.0)",
      [round(x, 4) for x in adj] == [0.006, 1.0, 1.0, 1.0, 1.0, 1.0])
check("v3 Holm: monotone non-decreasing in p-order",
      all(adj[i] <= adj[j] for i, j in
          zip(sorted(range(6), key=lambda k: raw[k])[:-1],
              sorted(range(6), key=lambda k: raw[k])[1:])))

# ---- v3.2  nested label masks (width isolated from draw) ----------------
gN = InstanceGenerator(seed=4242).generate_f1_base(64, 4)
n2 = InstanceGenerator.assign_labels(gN, 2, label_seed=500 + 2)
n4 = InstanceGenerator.assign_labels(gN, 4, label_seed=500 + 4)
n8 = InstanceGenerator.assign_labels(gN, 8, label_seed=500 + 8)
check("v3 labels: label_2 == label_4 mod 4",
      all(n2.nodes[v]['label'] == (n4.nodes[v]['label'] & 0b11)
          for v in gN.nodes()))
check("v3 labels: label_4 == label_8 mod 16",
      all(n4.nodes[v]['label'] == (n8.nodes[v]['label'] & 0b1111)
          for v in gN.nodes()))
check("v3 labels: 8-bit uses full range (draw is not degenerate)",
      len({n8.nodes[v]['label'] for v in gN.nodes()}) > 8)
check("v3 labels: structure identical across widths",
      set(n2.edges()) == set(n4.edges()) == set(n8.edges()) == set(gN.edges()))

# ---- v3.3  F3 seed decoupled from F1 ------------------------------------
gs = 42 + 1_000_003 * 0 + 0
f1g = InstanceGenerator(seed=gs).generate_f1_base(64, 4)
f3g = InstanceGenerator(seed=gs + 250_000_013).generate_f3_base(64, 4)
check("v3 F3 seed decoupled: F1 and F3 roots differ",
      set(f1g.edges()) != set(f3g.edges()))

# ---- v3.4  solver-error propagates as solver_error ----------------------
from runner import classify
def _boom(G1, G2, use_labels, q):
    raise RuntimeError("simulated native failure")
rb = pipe._run_with_timeout(_boom, f1g, f1g, timeout_sec=5)
check("v3 crash: error surfaced on the record",
      rb["error"] is not None and not rb["map_verified"])
check("v3 crash: classified solver_error (not 'no witness')",
      classify(rb, "iso") == "solver_error")
gi, _ = InstanceGenerator.iso_pair_with_rng(f1g, 11)
check("v3 classify: healthy run -> verified_witness",
      classify(pipe.run_a4_nauty(f1g, gi, timeout_sec=30), "iso")
      == "verified_witness")

# ---- v3.5  cost-per-byte dedup keeps F3 separate -------------------------
from cost_per_byte import byte_footprint
check("v3 bytes: reduces to nd for n<=256 unlabelled",
      byte_footprint(192, 6) == 192 * 6 and byte_footprint(256, 8) == 256 * 8)
check("v3 bytes: 2-byte ids past n=256",
      byte_footprint(1024, 6) == 1024 * 6 * 2)
_rows = [("F1", 64, 4, 0, 0, 2.55), ("F2", 64, 4, 0, 0, 2.63),
         ("F3", 64, 4, 0, 0, 2.71), ("F4", 64, 4, 0, 0, 2.69)]
_structs = [("F1_BASE", n, d) if (fam in ("F1", "F2", "F4") and k == 0 and lb == 0)
            else (fam, n, d, k, lb) for (fam, n, d, k, lb, t) in _rows]
check("v3 dedup: F1/F2/F4 collapse but F3 survives (2 points, not 1)",
      len(set(_structs)) == 2)

# ---- v4.1  module-level imports kill the per-child import cost ----------
import attackers as _atk
check("v4 imports: igraph imported in PARENT (fork inherits)",
      _atk.igraph is not None)
check("v4 imports: vf3py imported in PARENT (fork inherits)",
      _atk.vf3py is not None)
_big = InstanceGenerator(seed=99).generate_f1_base(256, 6)
_bigi, _ = InstanceGenerator.iso_pair_with_rng(_big, 3)
_rb = pipe.run_a5_bliss(_big, _bigi, timeout_sec=30)
_wrap = _rb["end_to_end_time_ms"] - _rb["solver_time_ms"]
check(f"v4 imports: Bliss wrapper {_wrap:.1f} ms < 200 ms (was ~660 ms)",
      _wrap < 200)

# ---- v4.2  Queue path: no Manager, results still correct ----------------
check("v4 queue: Bliss still returns a strictly verified witness",
      _rb["map_verified"])
_to = pipe.run_a2_vf2pp(_big, _bigi, timeout_sec=0.001)
check("v4 queue: timeout path intact (hit flag, no solver time)",
      _to["timeout_hit"] and _to["solver_time_ms"] is None)

# ---- v4.3  derived ladder reproduces the threshold table -----------------
from runner import emit_ladder
_base = {"family": "F1", "n": 64, "d": 4, "k": 0, "label_bits": 0,
         "seed": 1, "base_graph_id": "F1-64-4-0", "instance_id": 0}
_fake = {"map_found": True, "map_verified": True, "solver_time_ms": 91.73,
         "end_to_end_time_ms": 140.0, "time_ms": 140.0, "timeout_hit": False,
         "labels_enforced": False, "error": None, "process_exitcode": 0}
_recs = emit_ladder(_base, "vf2pp", [0.1, 1.0, 5.0], _fake)
check("v4 ladder: one record per threshold from ONE run",
      [r["timeout_sec"] for r in _recs] == [0.1, 1.0, 5.0])
check("v4 ladder: 91.73 ms does NOT time out at tau=0.1 s "
      "(the wrapper artefact the old ladder reported)",
      _recs[0]["timeout_hit"] is False and _recs[0]["success"] is True)
_slow = dict(_fake, solver_time_ms=None, timeout_hit=True)
_recs2 = emit_ladder(_base, "vf2pp", [0.1, 1.0, 5.0], _slow)
check("v4 ladder: cap-exceeding run times out at every threshold",
      all(r["timeout_hit"] and not r["success"] for r in _recs2))
_mid = dict(_fake, solver_time_ms=2500.0)
_recs3 = emit_ladder(_base, "vf3", [0.1, 1.0, 5.0, 30.0], _mid)
check("v4 ladder: 2.5 s exceeds 0.1/1 but not 5/30",
      [r["timeout_hit"] for r in _recs3] == [True, True, False, False])

# ---- v4.4  solvers_ext shim preserves the ORIGINAL contract -------------
from solvers_ext import networkx_to_igraph as _shim_conv
_ig, _idx, _rev = _shim_conv(f1g)
check("v4 shim: returns (ig, idx, rev) as memory_profile.py unpacks",
      isinstance(_idx, dict) and isinstance(_rev, dict)
      and all(_idx[_rev[i]] == i for i in _rev))

print()
print("ALL PASS" if not FAIL else f"FAILURES: {FAIL}")
