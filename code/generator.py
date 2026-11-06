"""
generator.py  (corrected, 2026-07)

Changes vs. the submitted version:

1. PAIRED-DESIGN HELPERS. The swap-significance experiment must compare
   k = 0 / 2n / 5n on the SAME base graph (the submitted runner mixed
   k into the seed, so each swap level saw a different base graph).
   New methods:
     apply_swaps(G, k, swap_seed)          -> swapped COPY of G
     assign_labels(G, label_bits, label_seed) -> labelled COPY of G
   Both are deterministic given their seed and never mutate the input.

2. All original methods are preserved with identical behaviour so that
   existing scripts (hardest_cells.py, memory_profile.py, extend_grid.py)
   keep producing the same instance distributions.
"""

import random

import networkx as nx


class InstanceGenerator:
    def __init__(self, seed=42):
        self.seed = seed
        random.seed(self.seed)

    # ------------------------------------------------------------------
    # Original families (unchanged behaviour)
    # ------------------------------------------------------------------

    def generate_f1_base(self, n, d):
        """F1: connected random d-regular graph."""
        max_attempts = 100
        for _ in range(max_attempts):
            G = nx.random_regular_graph(d, n, seed=self.seed)
            if nx.is_connected(G):
                return G
            self.seed += 1
        raise ValueError(
            f"Baglantili bir {d}-regular graf n={n} icin uretilemedi.")

    def generate_f2_base(self, n, d, k):
        """F2: F1 base + k double-edge swaps (original coupled behaviour)."""
        G = self.generate_f1_base(n, d)
        if k > 0:
            nx.double_edge_swap(G, nswap=k, max_tries=k * 10, seed=self.seed)
            self.seed += 1
        return G

    def generate_f3_base(self, n, d, extra_swaps=0):
        """F3: random 2-lift of an F1 base on n/2 vertices."""
        if n % 2 != 0:
            raise ValueError(
                "2-lift grafi icin dugum sayisi (n) cift sayi olmalidir.")
        b = n // 2
        B = self.generate_f1_base(b, d)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for u, v in B.edges():
            if random.choice([0, 1]) == 0:      # parallel
                G.add_edge(u, v)
                G.add_edge(u + b, v + b)
            else:                                # cross
                G.add_edge(u, v + b)
                G.add_edge(u + b, v)
        if extra_swaps > 0:
            nx.double_edge_swap(G, nswap=extra_swaps,
                                max_tries=extra_swaps * 10, seed=self.seed)
            self.seed += 1
        return G

    def generate_f4_base(self, n, d, label_bits, base_type="F1", k=0):
        """F4: labelled graph (original coupled behaviour)."""
        if base_type == "F2":
            G = self.generate_f2_base(n, d, k)
        else:
            G = self.generate_f1_base(n, d)
        max_label = (1 << label_bits) - 1
        for node in G.nodes():
            G.nodes[node]['label'] = random.randint(0, max_label)
        return G

    # ------------------------------------------------------------------
    # NEW: paired-design helpers (base graph fixed across hardening levels)
    # ------------------------------------------------------------------

    @staticmethod
    def apply_swaps(G, k, swap_seed):
        """Return a COPY of G with k degree-preserving double-edge swaps.
        Deterministic given swap_seed; G is never mutated. k == 0 returns
        an identical copy, so F2(k=0) is literally the same graph as the
        base -- which is what a paired design requires."""
        H = G.copy()
        if k > 0:
            nx.double_edge_swap(H, nswap=k, max_tries=k * 10, seed=swap_seed)
        return H

    @staticmethod
    def assign_labels(G, label_bits, label_seed):
        """Return a COPY of G with 'label' attributes drawn uniformly from
        [0, 2^label_bits - 1] using an isolated RNG (does not disturb the
        module-level random state)."""
        H = G.copy()
        # v3.2 nested label masks: callers (runner.py, test_fixes.py) fold the
        # label WIDTH into the seed (label_seed = base + label_bits). Subtract
        # it back out so every width of the same cell shares ONE master 8-bit
        # per-node draw, then mask to the requested width. This guarantees
        # label_2 == label_4 & 0b11 == label_8 & 0b11 (nested refinement), so
        # the across-width comparison is paired on the labelling too.
        rng = random.Random(label_seed - label_bits)
        mask = (1 << label_bits) - 1
        for node in H.nodes():
            H.nodes[node]['label'] = rng.getrandbits(8) & mask
        return H

    @staticmethod
    def iso_pair_with_rng(G, perm_seed):
        """Like generate_iso_pair but driven by an isolated RNG, so the SAME
        permutation can be reused across hardening levels of the same base
        graph (paired design). Returns (G_iso, mapping)."""
        rng = random.Random(perm_seed)
        nodes = list(G.nodes())
        permuted = list(G.nodes())
        rng.shuffle(permuted)
        mapping = dict(zip(nodes, permuted))
        return nx.relabel_nodes(G, mapping, copy=True), mapping

    # ------------------------------------------------------------------
    # Original pair constructions (unchanged)
    # ------------------------------------------------------------------

    def generate_iso_pair(self, G):
        nodes = list(G.nodes())
        permuted_nodes = list(G.nodes())
        random.shuffle(permuted_nodes)
        mapping = dict(zip(nodes, permuted_nodes))
        G_iso = nx.relabel_nodes(G, mapping, copy=True)
        return G_iso, mapping

    def generate_hard_noniso_pair(self, G, num_swaps=10):
        max_attempts = 50
        for _ in range(max_attempts):
            H = G.copy()
            nx.double_edge_swap(H, nswap=num_swaps,
                                max_tries=num_swaps * 10, seed=self.seed)
            self.seed += 1
            if nx.vf2pp_isomorphism(G, H) is None:
                return H
        raise ValueError(
            "Zorlu non-iso graf uretilemedi (Tum denemeler izomorfik cikti).")

    def generate_noisy_iso(self, G_iso, epsilon):
        if epsilon == 0:
            return G_iso.copy()
        m = G_iso.number_of_edges()
        flip_count = round(epsilon * m)
        G_noisy = G_iso.copy()
        nodes = list(G_noisy.nodes())
        flips_done, attempts = 0, 0
        max_attempts = flip_count * 10
        while flips_done < flip_count and attempts < max_attempts:
            u, v = random.sample(nodes, 2)
            attempts += 1
            if G_noisy.has_edge(u, v):
                G_noisy.remove_edge(u, v)
            else:
                G_noisy.add_edge(u, v)
            flips_done += 1
        return G_noisy


if __name__ == "__main__":
    gen = InstanceGenerator(seed=123)
    G = gen.generate_f1_base(64, 4)
    H0 = gen.apply_swaps(G, 0, swap_seed=1)
    H1 = gen.apply_swaps(G, 128, swap_seed=1)
    print("paired: base edges", G.number_of_edges(),
          "| k=0 identical:", set(G.edges()) == set(H0.edges()),
          "| k=128 differs:", set(G.edges()) != set(H1.edges()))
