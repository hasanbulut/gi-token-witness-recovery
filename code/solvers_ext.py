"""
solvers_ext.py  (compatibility shim, 2026-07, v4)

The corrected nauty witness and the Bliss (A5) solver now live directly in
attackers.py::AttackerPipeline. This module exists ONLY so that the existing
imports in hardest_cells.py / extend_grid.py / memory_profile.py keep working:

    from solvers_ext import ExtendedAttackerPipeline
    from solvers_ext import networkx_to_igraph

IMPORTANT -- why networkx_to_igraph is wrapped rather than re-exported:

The ORIGINAL solvers_ext.networkx_to_igraph returned

    (ig, idx, rev)          # reverse map THIRD

and memory_profile.py unpacks exactly that:

    ig1, _, r1 = networkx_to_igraph(G1)

attackers.networkx_to_igraph returns (ig, rev, colours) -- reverse map SECOND.
Re-exporting it directly silently rebinds `r1` to the colour vector, which is
None for unlabelled graphs, and every Bliss memory run dies with
"TypeError: 'NoneType' object is not subscriptable". A shim must ADAPT the
contract, not just forward the name.
"""

from attackers import (
    AttackerPipeline,
    networkx_to_pynauty,
    networkx_to_igraph as _attackers_networkx_to_igraph,
)

# Backwards-compatible alias: the single corrected pipeline.
ExtendedAttackerPipeline = AttackerPipeline


def networkx_to_igraph(G_nx):
    """ORIGINAL contract: returns (ig, idx, rev).

    idx : original node -> igraph vertex id
    rev : igraph vertex id -> original node
    """
    ig, rev, _colours = _attackers_networkx_to_igraph(G_nx, use_labels=False)
    idx = {v: i for i, v in rev.items()}
    return ig, idx, rev


__all__ = [
    "AttackerPipeline",
    "ExtendedAttackerPipeline",
    "networkx_to_pynauty",
    "networkx_to_igraph",
]
