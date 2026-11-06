"""
plot_coap.py  ->  figure for Step 10 (Reviewer 2 comments 6 and 7)

Imports the analytical model from coap_6lowpan.py and draws a two-panel figure:
(left) 6LoWPAN link frames and CoAP blocks required per token size for the
paper's configurations; (right) single confirmable CoAP round trip versus radio
duty-cycle rate, with the attacker's ~15 ms offline recovery drawn as a
reference line. This figure is analytical/deterministic and does not depend on
the student's run, but it is included here so the whole figure set is produced
from one place.

Usage:
    python3 plot_coap.py --out plot_coap_6lowpan.png
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from coap_6lowpan import (LinkParams, MacParams, PAPER_CONFIGS, bgraph,
                          token_transfer_profile, single_confirmable_rtt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="plot_coap_6lowpan.png")
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--attacker-ms", type=float, default=15.0)
    args = ap.parse_args()

    lp = LinkParams()
    fig, (axb, axr) = plt.subplots(1, 2, figsize=(11, 4.5))

    # ---- left: fragmentation per token ----
    labels = [lbl for (lbl, n, d) in PAPER_CONFIGS]
    Bs = [bgraph(n, d) for (lbl, n, d) in PAPER_CONFIGS]
    frames = [token_transfer_profile(B, lp, args.block_size)["link_frames"] for B in Bs]
    blocks = [token_transfer_profile(B, lp, args.block_size)["coap_blocks"] for B in Bs]
    x = np.arange(len(labels))
    w = 0.4
    axb.bar(x - w / 2, blocks, w, label="CoAP blocks", color="#1f77b4")
    axb.bar(x + w / 2, frames, w, label="802.15.4 link frames", color="#ff7f0e")
    axb.set_xticks(x)
    axb.set_xticklabels([f"{l}\n({B} B)" for l, B in zip(labels, Bs)],
                        fontsize=7, rotation=0)
    axb.set_ylabel("count")
    axb.set_title(f"Fragmentation per one-graph token (block={args.block_size} B)")
    axb.legend(fontsize=8)
    axb.grid(True, axis="y", alpha=0.3)

    # ---- right: single RTT vs duty-cycle rate ----
    hz = np.array([2, 4, 8, 16, 32], dtype=float)
    rtts = [single_confirmable_rtt(lp, MacParams(channel_check_hz=h)) for h in hz]
    axr.plot(hz, rtts, "-o", color="#2ca02c", label="single confirmable RTT")
    axr.axhline(args.attacker_ms, color="#d62728", ls="--",
                label=f"attacker VF3 recovery ~{args.attacker_ms:.0f} ms")
    axr.set_xscale("log", base=2)
    axr.set_yscale("log")
    axr.set_xticks(hz)
    axr.set_xticklabels([f"{int(h)}" for h in hz])
    axr.set_xlabel("radio duty-cycle channel-check rate (Hz)")
    axr.set_ylabel("latency (ms, log scale)")
    axr.set_title("Honest round trip vs. attacker recovery")
    axr.legend(fontsize=8)
    axr.grid(True, which="both", alpha=0.3)

    fig.suptitle("CoAP/6LoWPAN grounding of token size and round-trip latency")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
