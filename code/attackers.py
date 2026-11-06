"""
attackers.py  (corrected, 2026-07, v4)

Attacker tiers used by the paper:
    A1 WL-1, A2 VF2++, A3 VF3, A4 nauty      <- the SUBMITTED tier set
    A5 Bliss                                  <- added for Reviewer 2 comment 4

v4 changes (runtime + measurement quality only; nothing reported changes):

1. MODULE-LEVEL SOLVER IMPORTS. igraph/vf3py/pynauty are imported in the PARENT
   so forked children inherit them. Previously `import igraph` lived inside the
   Bliss worker, so EVERY child paid a fresh C-extension import: measured 660 ms
   of wrapper per Bliss call at n=384, versus 8 ms for nauty (whose pynauty
   import was already module-level). After hoisting: 14.6 ms. (The student's own
   memory_profile.py already used this trick; the core pipeline did not.)

2. QUEUE INSTEAD OF A PER-CALL MANAGER. multiprocessing.Manager() spawns a whole
   server process and measured 47.2 ms per call -- and because it was created
   BEFORE the timer, it never even appeared in end_to_end_time_ms. A Queue does
   the same job in ~4.8 ms round trip.

   The Queue is drained with a short-poll loop rather than a bare
   q.get(timeout=cap): a bare get would block for the FULL cap if a child died
   natively without putting a result, converting a fast crash into a slow bogus
   "timeout". The loop returns as soon as a result arrives (no added latency on
   fast solves), detects a dead child within ~50 ms, and still enforces the cap.

Retained from v2/v3:
  * split solver_time_ms (in-child) / end_to_end_time_ms (spawn+IPC inclusive)
  * corrected nauty witness (canon_label[pos] is the VERTEX at that position)
  * VF3 first-witness via are_isomorphic(get_mapping=True) -- note that
    get_exact_isomorphisms returns a fully-built List, so next(iter(...)) would
    NOT have stopped the enumeration
  * label-aware matching, enforced only when labels are INFORMATIVE
  * strict witness verification (bijection + edge-set equality + label equality)
  * crash detection via process exitcode
"""

import queue
import time
import multiprocessing
from collections import Counter

import networkx as nx
import pynauty

# Parent-side imports so forked children inherit the C extensions and do not
# each pay a fresh import (see note 1 above).
try:
    import igraph
except Exception:                                   # pragma: no cover
    igraph = None
try:
    import vf3py
except Exception:                                   # pragma: no cover
    vf3py = None

POLL_SEC = 0.05          # queue poll granularity / dead-child detection latency


# --------------------------------------------------------------------------
# Label helpers
# --------------------------------------------------------------------------

def _fully_labelled(G):
    return G.number_of_nodes() > 0 and all(
        'label' in G.nodes[v] for v in G.nodes()
    )


def graphs_are_labelled(G1, G2):
    """True iff BOTH graphs carry a 'label' attribute on every vertex."""
    return _fully_labelled(G1) and _fully_labelled(G2)


def labels_informative(G1, G2):
    """Enforce labels only when they CONSTRAIN the matching, i.e. both graphs
    are fully labelled AND more than one distinct label value exists. F4 with
    label_bits = 0 assigns a constant label: enforcing it changes no outcome
    but adds per-comparison callback cost, which would make F4(lb=0) differ in
    timing from its identical unlabelled root graph."""
    if not graphs_are_labelled(G1, G2):
        return False
    vals = {G1.nodes[v]['label'] for v in G1.nodes()}
    vals |= {G2.nodes[v]['label'] for v in G2.nodes()}
    return len(vals) > 1


def label_colour_classes(G, mapping):
    """Colour classes ordered by label value. Ordering by value in BOTH graphs
    makes the class order correspond, which pynauty's canonical form needs."""
    classes = {}
    for v in G.nodes():
        classes.setdefault(G.nodes[v]['label'], set()).add(mapping[v])
    return [classes[lbl] for lbl in sorted(classes)]


# --------------------------------------------------------------------------
# Conversions
# --------------------------------------------------------------------------

def networkx_to_pynauty(G_nx, use_labels=False):
    mapping = {node: i for i, node in enumerate(sorted(G_nx.nodes()))}
    adjacency = {mapping[v]: [mapping[u] for u in G_nx.neighbors(v)]
                 for v in G_nx.nodes()}
    colouring = label_colour_classes(G_nx, mapping) if use_labels else []
    g = pynauty.Graph(G_nx.number_of_nodes(),
                      adjacency_dict=adjacency,
                      vertex_coloring=colouring)
    return g, mapping


def networkx_to_igraph(G_nx, use_labels=False):
    """Returns (ig, rev, colours).

    NOTE: solvers_ext.networkx_to_igraph exposes the ORIGINAL (ig, idx, rev)
    contract that memory_profile.py unpacks; do not "simplify" that shim away.
    """
    nodes = sorted(G_nx.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    edges = [(idx[u], idx[v]) for u, v in G_nx.edges()]
    ig = igraph.Graph(n=len(nodes), edges=edges, directed=False)
    rev = {i: v for v, i in idx.items()}
    colours = [G_nx.nodes[v]['label'] for v in nodes] if use_labels else None
    return ig, rev, colours


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

class AttackerPipeline:
    def __init__(self):
        pass

    # ---- A1: WL-1 (in-process; no fork) ----------------------------------
    def run_a1_wl1(self, G1, G2):
        def get_wl_stats(G):
            colors = {node: str(G.degree(node)) for node in G.nodes()}
            rounds = 0
            while True:
                rounds += 1
                new_colors = {}
                for node in G.nodes():
                    neighbor_colors = sorted(
                        str(colors[nb]) for nb in G.neighbors(node))
                    new_colors[node] = str(
                        hash(str(colors[node]) + "".join(neighbor_colors)))
                if len(set(new_colors.values())) == len(set(colors.values())):
                    break
                colors = new_colors
            color_counts = Counter(colors.values())
            return (rounds, len(color_counts),
                    max(color_counts.values()) if color_counts else 0, colors)

        rounds1, classes1, max_class1, colors1 = get_wl_stats(G1)
        rounds2, _, _, colors2 = get_wl_stats(G2)
        wl_distinguished = (Counter(colors1.values())
                            != Counter(colors2.values()))
        return {
            "wl_rounds": max(rounds1, rounds2),
            "num_color_classes": classes1,
            "largest_color_class_size": max_class1,
            "wl_distinguished": wl_distinguished,
        }

    # ---- strict witness verification -------------------------------------
    def verify_mapping(self, G1, G2, mapping):
        """Bijection V(G1) -> V(G2), mapped edge set EQUALS E(G2), labels agree
        when both graphs are fully labelled."""
        if not mapping:
            return False
        if len(mapping) != G1.number_of_nodes():
            return False
        if set(mapping.keys()) != set(G1.nodes()):
            return False
        values = list(mapping.values())
        if len(set(values)) != len(values):
            return False
        if set(values) != set(G2.nodes()):
            return False
        mapped_edges = {frozenset((mapping[u], mapping[v]))
                        for u, v in G1.edges()}
        real_edges = {frozenset((u, v)) for u, v in G2.edges()}
        if mapped_edges != real_edges:
            return False
        if graphs_are_labelled(G1, G2):
            for u in G1.nodes():
                if G1.nodes[u]['label'] != G2.nodes[mapping[u]]['label']:
                    return False
        return True

    # ---- generic subprocess runner ---------------------------------------
    def _run_with_timeout(self, worker, G1, G2, timeout_sec):
        use_labels = labels_informative(G1, G2)
        q = multiprocessing.Queue()

        t0 = time.perf_counter()
        p = multiprocessing.Process(
            target=worker, args=(G1, G2, use_labels, q))
        p.start()

        # Short-poll drain: returns immediately when the child puts a result,
        # notices a dead child within POLL_SEC, and enforces the cap exactly.
        # The poll window is clamped to the REMAINING time, otherwise a cap
        # shorter than POLL_SEC would silently overshoot.
        deadline = t0 + timeout_sec
        result, timed_out = None, False
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                timed_out = p.is_alive()    # not alive -> crash, not timeout
                break
            try:
                result = q.get(timeout=min(POLL_SEC, remaining))
                break
            except queue.Empty:
                if not p.is_alive():
                    break                   # died without a result
        end_to_end_ms = (time.perf_counter() - t0) * 1000

        if timed_out:
            p.terminate()
            p.join()
            q.close()
            return {
                "success": False, "timeout_hit": True,
                "time_ms": timeout_sec * 1000,
                "end_to_end_time_ms": timeout_sec * 1000,
                "solver_time_ms": None,
                "map_found": False, "map_verified": False, "mapping": None,
                "labels_enforced": use_labels,
                "error": None, "process_exitcode": None,
            }

        p.join()
        exitcode = p.exitcode
        q.close()

        if result is None:
            # Child exited without producing anything -> native crash/OOM.
            return {
                "success": False, "timeout_hit": False,
                "time_ms": end_to_end_ms,
                "end_to_end_time_ms": end_to_end_ms,
                "solver_time_ms": None,
                "map_found": False, "map_verified": False, "mapping": None,
                "labels_enforced": use_labels,
                "error": f"worker exited with code {exitcode} "
                         "(no result recorded)",
                "process_exitcode": exitcode,
            }

        err = result.get('error')
        if err is None and exitcode not in (0, None):
            err = f"worker exited with code {exitcode}"

        mapping = result.get('mapping')
        map_found = result.get('success', False)
        map_verified = (self.verify_mapping(G1, G2, mapping)
                        if (map_found and mapping) else False)
        return {
            "success": map_found, "timeout_hit": False,
            "time_ms": end_to_end_ms,
            "end_to_end_time_ms": end_to_end_ms,
            "solver_time_ms": result.get('solver_time_ms'),
            "map_found": map_found, "map_verified": map_verified,
            "mapping": mapping,
            "labels_enforced": use_labels,
            "error": err, "process_exitcode": exitcode,
        }

    # ---- A2: VF2++ --------------------------------------------------------
    @staticmethod
    def _vf2pp_worker(G1, G2, use_labels, q):
        out = {'success': False, 'mapping': None,
               'solver_time_ms': None, 'error': None}
        try:
            node_label = "label" if use_labels else None
            t0 = time.perf_counter()
            mapping = nx.vf2pp_isomorphism(G1, G2, node_label=node_label)
            out['solver_time_ms'] = (time.perf_counter() - t0) * 1000
            if mapping is not None:
                out['success'] = True
                out['mapping'] = dict(mapping)
        except Exception as e:
            out['error'] = str(e)
        q.put(out)

    def run_a2_vf2pp(self, G1, G2, timeout_sec):
        return self._run_with_timeout(self._vf2pp_worker, G1, G2, timeout_sec)

    # ---- A3: VF3 -----------------------------------------------------------
    @staticmethod
    def _vf3_worker(G1, G2, use_labels, q):
        out = {'success': False, 'mapping': None,
               'solver_time_ms': None, 'error': None}
        try:
            node_match = None
            if use_labels:
                def node_match(a, b):               # noqa: E306
                    return a.get('label') == b.get('label')
            t0 = time.perf_counter()
            ok, mapping = vf3py.are_isomorphic(
                G1, G2, get_mapping=True, node_match=node_match)
            out['solver_time_ms'] = (time.perf_counter() - t0) * 1000
            if ok and mapping:
                out['success'] = True
                out['mapping'] = dict(mapping)
        except Exception as e:
            out['error'] = str(e)
        q.put(out)

    def run_a3_vf3(self, G1, G2, timeout_sec):
        return self._run_with_timeout(self._vf3_worker, G1, G2, timeout_sec)

    # ---- A4: nauty (corrected witness) -------------------------------------
    @staticmethod
    def _nauty_worker(G1, G2, use_labels, q):
        out = {'success': False, 'mapping': None,
               'solver_time_ms': None, 'error': None}
        try:
            g1, map1 = networkx_to_pynauty(G1, use_labels=use_labels)
            g2, map2 = networkx_to_pynauty(G2, use_labels=use_labels)
            t0 = time.perf_counter()
            is_iso = pynauty.isomorphic(g1, g2)
            if not is_iso:
                out['solver_time_ms'] = (time.perf_counter() - t0) * 1000
                q.put(out)
                return
            c1 = pynauty.canon_label(g1)   # c1[pos] = vertex at canonical pos
            c2 = pynauty.canon_label(g2)
            out['solver_time_ms'] = (time.perf_counter() - t0) * 1000
            rev1 = {v: k for k, v in map1.items()}
            rev2 = {v: k for k, v in map2.items()}
            out['mapping'] = {rev1[c1[i]]: rev2[c2[i]] for i in range(len(c1))}
            out['success'] = True
        except Exception as e:
            out['error'] = str(e)
        q.put(out)

    def run_a4_nauty(self, G1, G2, timeout_sec):
        return self._run_with_timeout(self._nauty_worker, G1, G2, timeout_sec)

    # ---- A5: Bliss (igraph backend; R2 comment 4) --------------------------
    @staticmethod
    def _bliss_worker(G1, G2, use_labels, q):
        out = {'success': False, 'mapping': None,
               'solver_time_ms': None, 'error': None}
        try:
            ig1, rev1, col1 = networkx_to_igraph(G1, use_labels=use_labels)
            ig2, rev2, col2 = networkx_to_igraph(G2, use_labels=use_labels)
            t0 = time.perf_counter()
            ok, m12, _ = ig1.isomorphic_bliss(
                ig2, return_mapping_12=True, return_mapping_21=False,
                color1=col1, color2=col2)
            out['solver_time_ms'] = (time.perf_counter() - t0) * 1000
            if ok:
                out['mapping'] = {rev1[i]: rev2[m12[i]]
                                  for i in range(len(m12))}
                out['success'] = True
        except Exception as e:
            out['error'] = str(e)
        q.put(out)

    def run_a5_bliss(self, G1, G2, timeout_sec):
        return self._run_with_timeout(self._bliss_worker, G1, G2, timeout_sec)


if __name__ == "__main__":
    pass
