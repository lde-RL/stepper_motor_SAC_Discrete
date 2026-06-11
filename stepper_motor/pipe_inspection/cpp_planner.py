"""
Chinese Postman Problem (CPP) solver for pipe inspection.

Given a topological map where every edge must be traversed at least once,
find the minimum-length closed walk that covers all edges.

Algorithm:
  1. Find all odd-degree nodes in the multigraph G.
  2. Build a complete auxiliary graph K on those odd nodes, edge weight =
     shortest path distance in G.
  3. Find minimum-weight perfect matching in K  (uses negation trick with
     networkx max_weight_matching).
  4. For each matched pair (u,v): duplicate the edges along the shortest
     path in G, making G Eulerian.
  5. Find Eulerian circuit with Hierholzer's algorithm.

Returns a list of (from_node, to_node, edge_key, is_deadhead) tuples.
is_deadhead=True means this edge was added as a duplicate (extra traversal).
"""

from __future__ import annotations
from itertools import combinations
from typing import List, Optional, Tuple

import networkx as nx

from graph_map import TopologicalMap, EdgeStatus


def _weight_fn(G: nx.MultiGraph):
    """Returns a function that gives minimum edge weight between u and v."""
    def w(u, v, d):
        # d is a dict of {key: attr_dict}  for MultiGraph
        return min(attr['data'].length for attr in d.values())
    return w


class ChinesePostmanPlanner:

    def solve(
        self,
        tmap: TopologicalMap,
        start_node: Optional[int] = None,
        only_uninspected: bool = True,
    ) -> List[Tuple[int, int, int, bool]]:
        """
        Parameters
        ----------
        tmap            : TopologicalMap after exploration is complete
        start_node      : node to begin and end the circuit (default: node 0)
        only_uninspected: if True, only include uninspected edges in cost;
                          already-inspected edges may still be used as
                          deadhead shortcuts but are not required.

        Returns
        -------
        List of (u, v, edge_key, is_deadhead) in traversal order.
        """
        G = tmap.graph

        if G.number_of_nodes() == 0:
            return []

        if start_node is None:
            start_node = list(G.nodes)[0]

        # Build a fresh multigraph with only the *required* edges
        # (for cost calculation).  Keep track of which edges are deadhead.
        MG = nx.MultiGraph()
        MG.add_nodes_from(G.nodes(data=True))

        required_keys: set = set()
        for u, v, k, attrs in G.edges(data=True, keys=True):
            ed = attrs['data']
            is_req = (not only_uninspected) or (ed.status != EdgeStatus.INSPECTED)
            MG.add_edge(u, v, key=k, data=ed, required=is_req)
            if is_req:
                required_keys.add((u, v, k))

        if not required_keys:
            return []   # everything already inspected

        # ---- Step 1: odd-degree nodes (in the REQUIRED subgraph) ----
        req_sub = nx.MultiGraph()
        req_sub.add_nodes_from(MG.nodes())
        for u, v, k, attrs in MG.edges(data=True, keys=True):
            if attrs.get('required', False):
                req_sub.add_edge(u, v, key=k, **attrs)

        odd_nodes = [v for v, d in req_sub.degree() if d % 2 == 1]

        # ---- Step 2: all-pairs shortest path between odd nodes ----
        if odd_nodes:
            path_cache: dict = {}
            complete = nx.Graph()

            for u, v in combinations(odd_nodes, 2):
                try:
                    length = nx.shortest_path_length(
                        MG, u, v,
                        weight=lambda a, b, d: min(
                            attr['data'].length for attr in d.values()
                        )
                    )
                    path = nx.shortest_path(
                        MG, u, v,
                        weight=lambda a, b, d: min(
                            attr['data'].length for attr in d.values()
                        )
                    )
                except nx.NetworkXNoPath:
                    continue

                complete.add_edge(u, v, weight=-length)   # negate for max
                path_cache[(u, v)] = path
                path_cache[(v, u)] = path[::-1]

            # ---- Step 3: minimum-weight perfect matching ----
            matching = nx.max_weight_matching(complete, maxcardinality=True)

            # ---- Step 4: duplicate edges along matched shortest paths ----
            deadhead_keys: set = set()
            for a, b in matching:
                path = path_cache.get((a, b)) or path_cache.get((b, a))
                if path is None:
                    continue
                for i in range(len(path) - 1):
                    pa, pb = path[i], path[i + 1]
                    # duplicate the cheapest edge between pa-pb
                    best_key = min(
                        MG[pa][pb],
                        key=lambda k: MG[pa][pb][k]['data'].length,
                    )
                    old_attrs = {
                        k2: v for k2, v in MG.edges[pa, pb, best_key].items()
                        if k2 not in ('required', 'deadhead')
                    }
                    new_key = MG.number_of_edges() + len(deadhead_keys)
                    MG.add_edge(pa, pb, key=new_key,
                                deadhead=True, required=False, **old_attrs)
                    deadhead_keys.add((pa, pb, new_key))

        # ---- Step 5: Eulerian circuit ----
        if not nx.is_eulerian(MG):
            # fallback: not all components reachable — just return required edges
            return [(u, v, k, False) for u, v, k in required_keys]

        circuit = list(nx.eulerian_circuit(MG, source=start_node, keys=True))

        # Annotate with deadhead flag
        result = []
        for u, v, k in circuit:
            is_dh = MG.edges[u, v, k].get('deadhead', False)
            result.append((u, v, k, is_dh))

        return result

    # ------------------------------------------------------------------
    # Utility: total distance breakdown
    # ------------------------------------------------------------------

    def path_stats(
        self,
        path: List[Tuple[int, int, int, bool]],
        tmap: TopologicalMap,
    ) -> dict:
        total = 0.0
        deadhead = 0.0
        for u, v, k, is_dh in path:
            try:
                length = tmap.edge_data(u, v, k).length
            except KeyError:
                # deadhead duplicate edge — key only exists in planner's MG;
                # use shortest parallel edge in tmap as the best estimate
                G = tmap.graph
                if G.has_edge(u, v):
                    length = min(
                        G.edges[u, v, ek]['data'].length
                        for ek in G[u][v]
                    )
                else:
                    length = 0.0
            total += length
            if is_dh:
                deadhead += length
        return {
            'total_m': round(total, 3),
            'inspection_m': round(total - deadhead, 3),
            'deadhead_m': round(deadhead, 3),
            'efficiency_pct': round(
                100 * (total - deadhead) / total, 1) if total else 0.0,
            'segments': len(path),
        }
