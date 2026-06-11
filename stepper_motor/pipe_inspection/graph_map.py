"""
Topological graph map for pipe inspection robots.

Nodes = junctions / dead-ends / start point
Edges = pipe segments (length [m], heading [rad], explored flag)

Loop closure: when a new junction position is within CLOSURE_DIST of
an existing node, the new observation is merged into that node instead
of creating a duplicate.
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import networkx as nx


class JunctionType(IntEnum):
    START = 0
    JUNCTION = 1   # ≥2 openings (T, cross, …)
    DEAD_END = 2


class EdgeStatus(IntEnum):
    UNEXPLORED = 0
    EXPLORED = 1
    INSPECTED = 2   # second pass: inspection completed


@dataclass
class NodeData:
    x: float
    y: float
    jtype: JunctionType
    visit_count: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class EdgeData:
    length: float          # metres
    heading: float         # radians, robot heading when traversing u→v
    status: EdgeStatus = EdgeStatus.UNEXPLORED
    traversal_count: int = 0
    timestamp: float = field(default_factory=time.time)


# Maximum distance (m) to consider two junctions the same (loop closure)
CLOSURE_DIST = 0.25


class TopologicalMap:
    """Thread-safe topological map backed by a networkx MultiGraph."""

    def __init__(self):
        self._G: nx.MultiGraph = nx.MultiGraph()
        self._node_counter = 0
        self._edge_key_counter = 0

        # Current robot state
        self.current_node: Optional[int] = None
        self.current_heading: float = 0.0   # accumulated IMU yaw (rad)
        self.x: float = 0.0                 # dead-reckoning X (m)
        self.y: float = 0.0                 # dead-reckoning Y (m)

    # ------------------------------------------------------------------
    # Graph mutation
    # ------------------------------------------------------------------

    def add_node(self, x: float, y: float, jtype: JunctionType) -> int:
        nid = self._node_counter
        self._node_counter += 1
        self._G.add_node(nid, data=NodeData(x=x, y=y, jtype=jtype))
        return nid

    def add_edge(self, u: int, v: int, length: float, heading: float) -> int:
        key = self._edge_key_counter
        self._edge_key_counter += 1
        self._G.add_edge(u, v, key=key,
                         data=EdgeData(length=length, heading=heading))
        return key

    def mark_edge_status(self, u: int, v: int, key: int,
                         status: EdgeStatus) -> None:
        ed: EdgeData = self._G.edges[u, v, key]['data']
        ed.status = status
        ed.traversal_count += 1
        ed.timestamp = time.time()

    def mark_node_visited(self, nid: int) -> None:
        nd: NodeData = self._G.nodes[nid]['data']
        nd.visit_count += 1
        self._G.nodes[nid]['data'] = nd

    # ------------------------------------------------------------------
    # Dead-reckoning position update
    # ------------------------------------------------------------------

    def update_odometry(self, delta_dist: float, heading: float) -> None:
        """Call each time the robot moves a known distance at a known heading."""
        self.current_heading = heading
        self.x += delta_dist * math.cos(heading)
        self.y += delta_dist * math.sin(heading)

    # ------------------------------------------------------------------
    # Loop closure
    # ------------------------------------------------------------------

    def try_loop_closure(self, x: float, y: float,
                         heading: float) -> Optional[int]:
        """
        Returns the existing node ID if (x,y) is within CLOSURE_DIST of it,
        else None.  Heading similarity is used as a tie-breaker.
        """
        best_nid = None
        best_dist = CLOSURE_DIST

        for nid, attrs in self._G.nodes(data=True):
            nd: NodeData = attrs['data']
            d = math.hypot(x - nd.x, y - nd.y)
            if d < best_dist:
                best_dist = d
                best_nid = nid

        return best_nid

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def unexplored_edges(self) -> List[Tuple[int, int, int]]:
        return [
            (u, v, k)
            for u, v, k, data in self._G.edges(data='data', keys=True)
            if data.status == EdgeStatus.UNEXPLORED
        ]

    def uninspected_edges(self) -> List[Tuple[int, int, int]]:
        return [
            (u, v, k)
            for u, v, k, data in self._G.edges(data='data', keys=True)
            if data.status != EdgeStatus.INSPECTED
        ]

    def node_data(self, nid: int) -> NodeData:
        return self._G.nodes[nid]['data']

    def edge_data(self, u: int, v: int, key: int) -> EdgeData:
        return self._G.edges[u, v, key]['data']

    @property
    def graph(self) -> nx.MultiGraph:
        return self._G

    def num_nodes(self) -> int:
        return self._G.number_of_nodes()

    def num_edges(self) -> int:
        return self._G.number_of_edges()

    # ------------------------------------------------------------------
    # Serialisation (simple dict for ROS param / JSON saving)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        nodes = {}
        for nid, attrs in self._G.nodes(data=True):
            nd: NodeData = attrs['data']
            nodes[nid] = {
                'x': nd.x, 'y': nd.y,
                'type': nd.jtype.name,
                'visits': nd.visit_count,
            }

        edges = []
        for u, v, k, attrs in self._G.edges(data=True, keys=True):
            ed: EdgeData = attrs['data']
            edges.append({
                'u': u, 'v': v, 'key': k,
                'length': ed.length,
                'heading': ed.heading,
                'status': ed.status.name,
                'traversals': ed.traversal_count,
            })

        return {'nodes': nodes, 'edges': edges,
                'robot': {'x': self.x, 'y': self.y,
                          'heading': self.current_heading,
                          'current_node': self.current_node}}
