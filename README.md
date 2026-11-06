# Exact graph-isomorphism tokens: witness-recovery artifact

Data and code for the article:

> A. Yildirim, S. Duman, H. Bulut. *On the Insufficiency of Exact Graph-Isomorphism Tokens for Lightweight Authentication in Ad Hoc and Sensor Networks.* Ad Hoc Networks (under review, manuscript ADHOC-D-26-01690).

Repository: https://github.com/hasanbulut/gi-token-witness-recovery

This is the **revision 2** artifact, version **1.0.0**. A Zenodo DOI has not yet been assigned. Citation metadata is provided in `CITATION.cff` and `.zenodo.json`; Hasan Bulut's ORCID is https://orcid.org/0000-0002-4872-5698.

Everything here belongs to the **revised** measurement pipeline: monotonic solver-only timing, mandatory witness verification, and label-aware matching in every solver tier.

## Repository layout

```
code/                     experimental pipeline (graph generation, five solver tiers, analysis, plots)
data/main_grid/           raw per-instance records of the main grid (13 gzip-compressed parts)
data/extended/            raw records of the extended grid to n = 1024
data/memory/              raw per-solve memory profiles
data/transport/           output of the CoAP/6LoWPAN transport model
data/summary/             per-cell statistics, significance tests, hard-cell study
data/figures/             summary data behind Figures 2-5 of the article
environment/              results_meta.json and requirements.txt
SHA256SUMS.txt            checksums of every data and environment file
LICENSE                   MIT (code) and CC BY 4.0 (data)
```

## What each file supports in the article

| File | Article location |
|---|---|
| `data/main_grid/results_part_*.jsonl.gz` | All main-grid results (Sections 4.1-4.7, Figures 2-5) and the nauty/Bliss sweep over all 216 isomorphic cells |
| `data/main_grid/SHA256SUMS_uncompressed.txt` | Checksums of the uncompressed parts, as produced on the reference machine |
| `data/summary/vf3_stats_summary.csv` | Per-cell N, median, T90, mean, std, IQR, 95% bootstrap CIs; scaling fit in Section 4.1 |
| `data/summary/vf3_stats_significance.txt` | Paired Wilcoxon tests with Holm correction (swaps, label widths), F1-vs-F3 Mann-Whitney test, effect sizes (Sections 4.4-4.5) |
| `data/summary/hardest_summary.txt` | Canonical-labelling solvers on representative hard cells (Section 4.8, Table 8) |
| `data/extended/results_extended_part_00.jsonl` | Extended grid to n = 1024, 3,200 records (Section 4.9, Table 9) |
| `data/memory/mem_*.jsonl` | Solver memory profile (Section 4.2, Table 4) |
| `data/transport/coap_console_pair.log` | Transport model at the pair level, behind Table 10 |
| `data/transport/coap_console.log` | Transport model at the one-graph level, and the single round-trip figures of the case study |
| `data/figures/fig*_data.csv` | Plotted values of Figures 2-5 |

Coverage of the raw main-grid dataset (729,000 records). Counts are over isomorphic pairs without noise; each pair is solved once at the tier's largest budget and the lower budgets are derived from the measured solver time.

| Attacker tier | Cells | Timeout ladder (s) | Witness found and verified | Solved within the budget |
|---|---|---|---|---|
| WL-1 (A1) | 216 | invariant check | n/a (decision only) | n/a |
| VF2++ (A2) | 216 | 0.1, 1, 5 | 126,414 / 129,600 | 111,220 / 129,600 |
| VF3 (A3) | 216 | 0.1, 1, 5, 30 | 172,800 / 172,800 | 172,791 / 172,800 |
| nauty (A4) | 216 | 0.1, 1, 5, 30 | 172,800 / 172,800 | 172,783 / 172,800 |
| Bliss (A5) | 216 | 0.1, 1, 5, 30 | 172,800 / 172,800 | 172,800 / 172,800 |

The 9 VF3 and 17 nauty attempts outside the budget all occur at τ = 0.1 s; from τ = 1 s upward both tiers solve every attempt.

## Record format

One JSON object per solve attempt. A main-grid record looks like this:

```json
{"family": "F1", "n": 64, "d": 4, "k": 0, "label_bits": 0, "seed": 42,
 "base_graph_id": "F1-64-4-0", "instance_id": 0, "pair_type": "iso", "eps": 0,
 "attack": "bliss", "timeout_sec": 0.1, "map_found": true, "map_verified": true,
 "success": true, "solver_time_ms": 0.392, "end_to_end_time_ms": 3.240,
 "time_ms": 3.240, "timeout_hit": false, "labels_enforced": false,
 "status": "verified_witness", "error": null, "process_exitcode": 0,
 "derived_from_single_run": true, "measured_cap_sec": 30.0}
```

- `solver_time_ms` is the monotonic in-process solver time reported in the article. `time_ms` and `end_to_end_time_ms` include process and harness overhead and are kept for transparency only.
- `success` is `map_found and map_verified` and the solve finished within `timeout_sec`.
- `status` is one of `verified_witness`, `timeout`, or a failure class.
- WL-1 records carry only the decision fields, since WL-1 does not produce a mapping.

## Reproducing the results

The scripts import each other by file name and read `config.yaml` from the working directory, so run them from inside `code/`. The reference run used Python 3.10 on Linux; `pynauty` may need a C compiler (`apt install build-essential`). Use Linux or WSL with Python 3.10 to reproduce the reference environment. The multiprocessing test script assumes the Linux `fork` start method and is not intended for native Windows or the changed multiprocessing defaults in Python 3.14.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r environment/requirements.txt
cd code

# 0. Test gate: must end with ALL PASS
python3 test_fixes.py

# 1. Main grid (about 7 h): writes results.jsonl and results_meta.json
python3 runner.py

# 2. Statistics (reproduces data/summary/vf3_stats_*)
python3 stats_analysis.py results.jsonl --attack vf3 --timeout 5 --ref-n 192 --ref-d 6 --out-prefix vf3_

# 3. Cost per byte (Section 4.1)
python3 cost_per_byte.py --results results.jsonl --attack vf3 --timeout 5 --pair

# 4. Hard cells with nauty and Bliss (Section 4.8)
python3 hardest_cells.py --instances 200 --timeout 5 | tee hardest_summary.txt
python3 plot_hardest.py results_hardest.jsonl --out plot_hardest_cells.png

# 5. Memory profile (Section 4.2)
python3 memory_profile.py --instances 100
python3 plot_memory.py results_memory.jsonl

# 6. Extended grid to n = 1024 (Section 4.9)
python3 extend_grid.py --instances 200 --timeout 30 --solvers vf3 bliss --out results_extended.jsonl
python3 plot_extend.py results_extended.jsonl --out plot_extend_grid.png

# 7. Transport model (deterministic, any machine)
python3 coap_6lowpan.py > coap_console.log
python3 coap_6lowpan.py --pair > coap_console_pair.log
python3 plot_coap.py --out plot_coap_6lowpan.png
```

To recompute the statistics from the archived records instead of re-running the grid:

```bash
cd code
gunzip -c ../data/main_grid/results_part_*.jsonl.gz > results.jsonl
python3 stats_analysis.py results.jsonl --attack vf3 --timeout 5 --ref-n 192 --ref-d 6 --out-prefix vf3_
```

This reproduces `data/summary/vf3_stats_summary.csv` and `vf3_stats_significance.txt` byte for byte. Running the transport model reproduces both files in `data/transport/` exactly, since its Monte-Carlo seed is fixed.

Every per-instance seed is fixed and derived deterministically per cell. Verification outcomes therefore reproduce exactly. Absolute timings, and with them the success/timeout status at the tight τ = 0.1 s budget, depend on the host. The reference machine was an Intel Core i7-11850H (2.50 GHz, 64 GB RAM), Ubuntu 22.04 under Windows Subsystem for Linux 2, Python 3.10.12, NetworkX 3.4.2, vf3py 1.0.2, pynauty 2.8.8.1, igraph 1.0.0 (Bliss); see `environment/results_meta.json`. The memory profile was run on a separate machine: Intel Xeon @ 2.80 GHz, 4 GB RAM, Linux 6.18, glibc 2.39, Python 3.12.3, NetworkX 3.6.1.

## Note on the transport-model constants

The header of the transport logs states 103 bytes available to 6LoWPAN (127 − 11 MAC − 13 security). The article follows the accounting of RFC 4944 instead: 102 octets at the MAC layer, minus 13 octets for AES-CCM-64, leaves 89 octets at the adaptation layer. In both accountings one 64-byte CoAP block fits in one IEEE 802.15.4 frame, so the block and frame counts are unchanged. Transfer times also depend on the model's MAC-header airtime accounting; the archived logs retain the original constants and have not been regenerated with the alternative accounting.

## Verifying integrity

```bash
sha256sum -c SHA256SUMS.txt
```

To check the decompressed main-grid parts against the checksums taken on the reference machine:

```bash
cd data/main_grid
for f in results_part_*.jsonl.gz; do gunzip -k "$f"; done
sha256sum -c SHA256SUMS_uncompressed.txt
```

## Licence

Code (`code/`): MIT. Data (`data/`, `environment/`): CC BY 4.0. See `LICENSE`.

## Archiving on Zenodo

Before publishing the first GitHub release, connect this repository in [Zenodo's GitHub settings](https://zenodo.org/account/settings/github/). Then publish the `v1.0.0` release on GitHub. Zenodo will archive the release using `.zenodo.json` and assign a DOI. Add the assigned DOI to this README and `CITATION.cff`, and use it in the article's data-availability statement.

See Zenodo's guides to [enable a repository](https://help.zenodo.org/docs/github/enable-repository/) and [archive a release](https://help.zenodo.org/docs/github/archive-software/github-upload/).
