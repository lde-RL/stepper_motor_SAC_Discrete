#!/usr/bin/env python3
"""
Pipe Inspection SLAM + CPP Planner — ROS 2 Node (Linux PC side)

Subscriptions
-------------
/pipe/odom_segment   (Float32MultiArray)  [dist_m, heading_rad]
/pipe/junction_event (Int32MultiArray)    [type, n_openings, f_mm, l_mm, r_mm]

Publications
------------
/pipe/graph_json     (String)   full map as JSON (latched, 1 Hz)
/pipe/path_json      (String)   CPP inspection path as JSON (latched)
/pipe/status         (String)   human-readable status line

Services
--------
/pipe/plan_inspection  (Trigger)  run CPP, publish /pipe/path_json
/pipe/save_map         (Trigger)  write map JSON to /tmp/pipe_map.json
/pipe/reset            (Trigger)  clear map and start fresh

Junction event type encoding (matches Teensy firmware):
  0 = START
  1 = JUNCTION
  2 = DEAD_END
  3 = LOOP_CLOSURE (Teensy-side hint, but PC verifies)
"""

from __future__ import annotations

import json
import math
import sys
import os
import time

# ── allow imports from the same directory even when called as a module ──
sys.path.insert(0, os.path.dirname(__file__))

from graph_map import TopologicalMap, JunctionType, EdgeStatus
from cpp_planner import ChinesePostmanPlanner

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import String, Float32MultiArray, Int32MultiArray
from std_srvs.srv import Trigger


LATCHED_QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)

# Junction event type IDs (must match firmware enum)
EVT_START    = 0
EVT_JUNCTION = 1
EVT_DEAD_END = 2
EVT_CLOSURE  = 3


class PipeSlamNode(Node):

    def __init__(self):
        super().__init__('pipe_slam_node')

        # ── internal state ──────────────────────────────────────────────
        self._map = TopologicalMap()
        self._planner = ChinesePostmanPlanner()
        self._last_inspection_path: list = []
        self._pending_edge: tuple | None = None   # (from_node, heading, dist_so_far)
        self._exploring = False

        # ── subscriptions ───────────────────────────────────────────────
        self.create_subscription(
            Float32MultiArray,
            '/pipe/odom_segment',
            self._odom_cb,
            10,
        )
        self.create_subscription(
            Int32MultiArray,
            '/pipe/junction_event',
            self._junction_cb,
            10,
        )

        # ── publications ────────────────────────────────────────────────
        self._graph_pub = self.create_publisher(String, '/pipe/graph_json', LATCHED_QOS)
        self._path_pub  = self.create_publisher(String, '/pipe/path_json',  LATCHED_QOS)
        self._status_pub = self.create_publisher(String, '/pipe/status', 10)

        # ── services ────────────────────────────────────────────────────
        self.create_service(Trigger, '/pipe/plan_inspection', self._svc_plan)
        self.create_service(Trigger, '/pipe/save_map',        self._svc_save)
        self.create_service(Trigger, '/pipe/reset',           self._svc_reset)

        # ── periodic publish ────────────────────────────────────────────
        self.create_timer(1.0, self._publish_graph)

        self.get_logger().info('PipeSlamNode ready.')

    # ════════════════════════════════════════════════════════════════════
    # Callbacks
    # ════════════════════════════════════════════════════════════════════

    def _odom_cb(self, msg: Float32MultiArray) -> None:
        """
        Receive incremental odometry from Teensy.
        data = [delta_dist_m, heading_rad]
        Accumulates position; does NOT yet close an edge (edge is closed on
        the next junction event).
        """
        if len(msg.data) < 2:
            return
        delta_dist, heading = float(msg.data[0]), float(msg.data[1])
        self._map.update_odometry(delta_dist, heading)

        if self._pending_edge is not None:
            from_node, start_heading, acc_dist = self._pending_edge
            self._pending_edge = (from_node, start_heading, acc_dist + delta_dist)

    def _junction_cb(self, msg: Int32MultiArray) -> None:
        """
        Receive junction detection event from Teensy.
        data = [event_type, n_openings, front_mm, left_mm, right_mm]
        """
        if len(msg.data) < 2:
            return

        etype     = int(msg.data[0])
        n_open    = int(msg.data[1])
        front_mm  = int(msg.data[2]) if len(msg.data) > 2 else 9999
        left_mm   = int(msg.data[3]) if len(msg.data) > 3 else 9999
        right_mm  = int(msg.data[4]) if len(msg.data) > 4 else 9999

        x, y = self._map.x, self._map.y

        self.get_logger().info(
            f'Junction event: type={etype} n_open={n_open} '
            f'f={front_mm} l={left_mm} r={right_mm}  pos=({x:.2f},{y:.2f})'
        )

        # ── handle START event (robot deployed into pipe) ───────────────
        if etype == EVT_START:
            nid = self._map.add_node(x, y, JunctionType.START)
            self._map.current_node = nid
            self._exploring = True
            self._pending_edge = None
            self.get_logger().info(f'Map initialised.  Start node={nid}')
            self._publish_status(f'Exploring — start node={nid}')
            return

        if not self._exploring:
            return

        # ── attempt loop closure ────────────────────────────────────────
        matched_node = self._map.try_loop_closure(x, y, self._map.current_heading)

        if matched_node is not None and matched_node != self._map.current_node:
            self.get_logger().info(f'Loop closure!  → node {matched_node}')
            self._close_current_edge(matched_node)
            self._map.current_node = matched_node
            self._map.mark_node_visited(matched_node)
            self._publish_status(f'Loop closure → node {matched_node}')
            return

        # ── new junction / dead-end ─────────────────────────────────────
        jtype = JunctionType.DEAD_END if etype == EVT_DEAD_END else JunctionType.JUNCTION
        new_node = self._map.add_node(x, y, jtype)

        self._close_current_edge(new_node)
        self._map.current_node = new_node
        self._map.mark_node_visited(new_node)

        self.get_logger().info(
            f'New node {new_node} ({jtype.name})  total={self._map.num_nodes()}'
        )
        self._publish_status(
            f'{"Dead-end" if jtype == JunctionType.DEAD_END else "Junction"} '
            f'node {new_node}  edges={self._map.num_edges()}'
        )

    # ════════════════════════════════════════════════════════════════════
    # Services
    # ════════════════════════════════════════════════════════════════════

    def _svc_plan(self, _req, response: Trigger.Response) -> Trigger.Response:
        if self._map.num_nodes() < 2:
            response.success = False
            response.message = 'Map has fewer than 2 nodes — explore first.'
            return response

        start = self._map.current_node or list(self._map.graph.nodes)[0]
        path = self._planner.solve(self._map, start_node=start,
                                   only_uninspected=True)

        if not path:
            response.success = False
            response.message = 'No uninspected edges found.'
            return response

        stats = self._planner.path_stats(path, self._map)
        self._last_inspection_path = path

        path_data = {
            'stats': stats,
            'waypoints': [
                {'from': u, 'to': v, 'edge_key': k, 'deadhead': dh}
                for u, v, k, dh in path
            ],
        }
        msg = String()
        msg.data = json.dumps(path_data, indent=2)
        self._path_pub.publish(msg)

        self.get_logger().info(
            f'CPP path planned: {stats["segments"]} segments, '
            f'{stats["total_m"]} m total, '
            f'{stats["efficiency_pct"]} % efficiency'
        )
        response.success = True
        response.message = json.dumps(stats)
        return response

    def _svc_save(self, _req, response: Trigger.Response) -> Trigger.Response:
        path = '/tmp/pipe_map.json'
        try:
            with open(path, 'w') as f:
                json.dump(self._map.to_dict(), f, indent=2)
            response.success = True
            response.message = f'Map saved to {path}'
        except Exception as e:
            response.success = False
            response.message = str(e)
        return response

    def _svc_reset(self, _req, response: Trigger.Response) -> Trigger.Response:
        self._map = TopologicalMap()
        self._planner = ChinesePostmanPlanner()
        self._last_inspection_path = []
        self._pending_edge = None
        self._exploring = False
        self.get_logger().info('Map reset.')
        response.success = True
        response.message = 'Map reset.'
        return response

    # ════════════════════════════════════════════════════════════════════
    # Internal helpers
    # ════════════════════════════════════════════════════════════════════

    def _close_current_edge(self, to_node: int) -> None:
        """Finalise the edge from current_node to to_node."""
        from_node = self._map.current_node
        if from_node is None:
            return

        if self._pending_edge is not None:
            _, start_heading, length = self._pending_edge
        else:
            length = 0.05   # minimum segment length (m)
            start_heading = self._map.current_heading

        key = self._map.add_edge(from_node, to_node, length, start_heading)
        self._map.mark_edge_status(from_node, to_node, key, EdgeStatus.EXPLORED)
        self._pending_edge = (to_node, self._map.current_heading, 0.0)

    def _publish_graph(self) -> None:
        msg = String()
        msg.data = json.dumps(self._map.to_dict())
        self._graph_pub.publish(msg)

    def _publish_status(self, text: str) -> None:
        msg = String()
        msg.data = text
        self._status_pub.publish(msg)
        self.get_logger().info(f'[STATUS] {text}')


# ════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = PipeSlamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
