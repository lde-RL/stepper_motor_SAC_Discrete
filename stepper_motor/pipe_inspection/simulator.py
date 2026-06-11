#!/usr/bin/env python3
"""
Pipe Inspection Simulator — runs entirely on Linux without ROS 2.

Simulates a robot navigating a virtual pipe network, builds the
topological map, runs the CPP planner, and prints/saves results.

Usage
-----
  python3 simulator.py              # default pipe network
  python3 simulator.py --pipe complex
  python3 simulator.py --pipe simple --save map.json

Pipe networks
-------------
  simple   : straight pipe with one T-junction
  medium   : two loops, four dead-ends
  complex  : large grid-like manifold with many branches
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import os
import time
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(__file__))

from graph_map import TopologicalMap, JunctionType, EdgeStatus
from cpp_planner import ChinesePostmanPlanner

# ── ANSI colours ────────────────────────────────────────────────────────
GREEN  = '\033[92m'
YELLOW = '\033[93m'
CYAN   = '\033[96m'
RED    = '\033[91m'
BOLD   = '\033[1m'
RESET  = '\033[0m'

def cprint(colour, *args):
    print(colour + ' '.join(str(a) for a in args) + RESET)


# ════════════════════════════════════════════════════════════════════════
# Virtual pipe networks
# Each network is described as a list of segments:
#   (from_label, to_label, length_m, heading_deg)
# ════════════════════════════════════════════════════════════════════════

NETWORKS = {
    'simple': [
        ('START', 'A', 2.0,   0),
        ('A',     'B', 1.5,   0),
        ('A',     'C', 1.0,  90),
        ('B',     'D', 1.0,   0),   # dead-end D
        ('C',     'E', 1.2,  90),   # dead-end E
    ],
    'medium': [
        ('START', 'A', 1.0,   0),
        ('A',     'B', 2.0,   0),
        ('B',     'C', 1.5,  90),
        ('C',     'D', 2.0, 180),
        ('D',     'A', 1.5, 270),   # loop closure A
        ('B',     'E', 1.0,   0),   # dead-end E
        ('C',     'F', 0.8,   0),   # junction F
        ('F',     'G', 1.0,  90),   # dead-end G
        ('F',     'H', 1.2, 270),   # dead-end H
    ],
    'complex': [
        ('START', 'A', 1.0,   0),
        ('A',     'B', 2.0,   0),
        ('B',     'C', 2.0,   0),
        ('C',     'D', 2.0,   0),   # dead-end D
        ('A',     'E', 2.0,  90),
        ('B',     'F', 2.0,  90),
        ('C',     'G', 2.0,  90),
        ('E',     'F', 2.0,   0),   # horizontal connectors
        ('F',     'G', 2.0,   0),
        ('E',     'H', 2.0,  90),   # dead-end H
        ('G',     'I', 2.0,  90),
        ('H',     'I', 2.0,   0),   # loop closure
        ('I',     'J', 1.0,  90),   # dead-end J
        ('F',     'K', 1.0, 270),   # dead-end K
    ],
}


class VirtualRobot:
    """Drives a TopologicalMap by replaying a pipe-network description."""

    def __init__(self, network: List[Tuple]):
        self._network = network
        self._map = TopologicalMap()
        self._label_to_id: dict = {}
        self._events: List[str] = []

    def _get_or_create_node(self, label: str, x: float, y: float,
                            jtype: JunctionType) -> int:
        if label in self._label_to_id:
            return self._label_to_id[label]
        nid = self._map.add_node(x, y, jtype)
        self._label_to_id[label] = nid
        return nid

    def run(self) -> TopologicalMap:
        # Lay out node positions via dead-reckoning
        positions: dict = {}
        positions['START'] = (0.0, 0.0)

        for from_lbl, to_lbl, length, hdg_deg in self._network:
            if from_lbl not in positions:
                positions[from_lbl] = (0.0, 0.0)
            fx, fy = positions[from_lbl]
            rad = math.radians(hdg_deg)
            tx = fx + length * math.cos(rad)
            ty = fy + length * math.sin(rad)
            if to_lbl not in positions:
                positions[to_lbl] = (tx, ty)

        # Count adjacency to determine junction types
        adj: dict = {}
        for from_lbl, to_lbl, *_ in self._network:
            adj[from_lbl] = adj.get(from_lbl, 0) + 1
            adj[to_lbl]   = adj.get(to_lbl,   0) + 1

        def jtype_for(label: str) -> JunctionType:
            if label == 'START':
                return JunctionType.START
            if adj.get(label, 0) == 1:
                return JunctionType.DEAD_END
            return JunctionType.JUNCTION

        # Build map
        for from_lbl, to_lbl, length, hdg_deg in self._network:
            fx, fy = positions[from_lbl]
            tx, ty = positions[to_lbl]

            from_id = self._get_or_create_node(
                from_lbl, fx, fy, jtype_for(from_lbl))
            to_id = self._get_or_create_node(
                to_lbl, tx, ty, jtype_for(to_lbl))

            key = self._map.add_edge(from_id, to_id, length,
                                     math.radians(hdg_deg))
            self._map.mark_edge_status(from_id, to_id, key,
                                       EdgeStatus.EXPLORED)

            self._events.append(
                f'  {from_lbl}({from_id}) ──{length:.1f}m──▶ '
                f'{to_lbl}({to_id})'
            )

        return self._map

    @property
    def label_map(self):
        return {v: k for k, v in self._label_to_id.items()}


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Pipe inspection simulator')
    parser.add_argument('--pipe', choices=list(NETWORKS), default='medium',
                        help='Pipe network to simulate')
    parser.add_argument('--save', metavar='FILE',
                        help='Save map JSON to FILE')
    parser.add_argument('--path-save', metavar='FILE',
                        help='Save CPP path JSON to FILE')
    args = parser.parse_args()

    network = NETWORKS[args.pipe]

    cprint(BOLD + CYAN, f'\n══ Pipe Inspection Simulator — network: {args.pipe} ══')

    # ── Phase 1: Exploration ─────────────────────────────────────────
    cprint(YELLOW, '\n[Phase 1] Exploration (DFS traversal)')
    robot = VirtualRobot(network)
    tmap = robot.run()
    lmap = robot.label_map   # id → label

    for ev in robot._events:
        print(ev)

    cprint(GREEN,
           f'\n  Map built: {tmap.num_nodes()} nodes, {tmap.num_edges()} edges')

    # Print graph summary
    print()
    cprint(BOLD, '  Node list:')
    for nid, attrs in tmap.graph.nodes(data=True):
        nd = attrs['data']
        deg = tmap.graph.degree(nid)
        label = lmap.get(nid, '?')
        print(f'    node {nid} [{label:6s}]  {nd.jtype.name:10s}  '
              f'pos=({nd.x:5.1f}, {nd.y:5.1f})  deg={deg}')

    # ── Phase 2: CPP Path Planning ────────────────────────────────────
    cprint(YELLOW, '\n[Phase 2] Chinese Postman Problem — coverage path')

    planner = ChinesePostmanPlanner()
    start_id = tmap._label_to_id['START'] if hasattr(tmap, '_label_to_id') \
               else list(tmap.graph.nodes)[0]
    # VirtualRobot stores label→id; retrieve START node id
    start_id = robot._label_to_id.get('START', list(tmap.graph.nodes)[0])

    t0 = time.perf_counter()
    path = planner.solve(tmap, start_node=start_id, only_uninspected=False)
    elapsed = (time.perf_counter() - t0) * 1000

    if not path:
        cprint(RED, '  No path found!')
        sys.exit(1)

    stats = planner.path_stats(path, tmap)

    cprint(GREEN, f'\n  CPP solved in {elapsed:.1f} ms')
    print(f'  Segments       : {stats["segments"]}')
    print(f'  Total distance : {stats["total_m"]} m')
    print(f'  Inspection dist: {stats["inspection_m"]} m')
    print(f'  Deadhead dist  : {stats["deadhead_m"]} m')
    print(f'  Efficiency     : {stats["efficiency_pct"]} %')

    print()
    cprint(BOLD, '  Traversal order:')
    for i, (u, v, k, is_dh) in enumerate(path):
        ul = lmap.get(u, str(u))
        vl = lmap.get(v, str(v))
        dh_tag = f'{YELLOW}[deadhead]{RESET}' if is_dh else ''
        try:
            length = tmap.edge_data(u, v, k).length
        except KeyError:
            # deadhead duplicate — find length from any parallel edge in tmap
            G = tmap.graph
            if G.has_edge(u, v):
                length = min(G.edges[u, v, ek]['data'].length for ek in G[u][v])
            else:
                length = 0.0
        print(f'  {i+1:3d}. {ul:6s} → {vl:6s}  ({length:.1f} m) {dh_tag}')

    # ── Optional saves ────────────────────────────────────────────────
    if args.save:
        with open(args.save, 'w') as f:
            json.dump(tmap.to_dict(), f, indent=2)
        cprint(GREEN, f'\n  Map saved → {args.save}')

    if args.path_save:
        path_data = {
            'stats': stats,
            'waypoints': [
                {'from': u, 'to': v, 'edge_key': k, 'deadhead': dh,
                 'from_label': lmap.get(u, str(u)),
                 'to_label': lmap.get(v, str(v))}
                for u, v, k, dh in path
            ],
        }
        with open(args.path_save, 'w') as f:
            json.dump(path_data, f, indent=2)
        cprint(GREEN, f'  Path saved → {args.path_save}')

    cprint(BOLD + CYAN, '\n══ Done ══\n')


if __name__ == '__main__':
    main()
