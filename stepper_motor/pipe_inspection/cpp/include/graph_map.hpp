// Topological graph map for pipe inspection robots (C++17, header-only).
//
// Nodes = junctions / dead-ends / start point
// Edges = pipe segments (length [m], heading [rad], status)
//
// Mirrors the Python graph_map.py implementation.

#pragma once

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace pipe {

enum class JunctionType : uint8_t { START = 0, JUNCTION = 1, DEAD_END = 2 };

enum class EdgeStatus : uint8_t {
    UNEXPLORED = 0,
    EXPLORED   = 1,
    INSPECTED  = 2,   // second pass: inspection completed
};

inline const char* to_string(JunctionType t) {
    switch (t) {
        case JunctionType::START:    return "START";
        case JunctionType::JUNCTION: return "JUNCTION";
        case JunctionType::DEAD_END: return "DEAD_END";
    }
    return "?";
}

inline const char* to_string(EdgeStatus s) {
    switch (s) {
        case EdgeStatus::UNEXPLORED: return "UNEXPLORED";
        case EdgeStatus::EXPLORED:   return "EXPLORED";
        case EdgeStatus::INSPECTED:  return "INSPECTED";
    }
    return "?";
}

struct NodeData {
    double x = 0.0;
    double y = 0.0;
    JunctionType jtype = JunctionType::JUNCTION;
    int visit_count = 0;
};

struct EdgeData {
    int u = -1;
    int v = -1;
    double length  = 0.0;   // metres
    double heading = 0.0;   // radians, robot heading when traversing u->v
    EdgeStatus status = EdgeStatus::UNEXPLORED;
    int traversal_count = 0;
};

// Maximum distance (m) to consider two junctions the same (loop closure)
constexpr double CLOSURE_DIST = 0.25;

// Multigraph: nodes are indices into nodes_, edges are indices into edges_.
// adjacency_[node] lists edge indices incident to that node.
class TopologicalMap {
public:
    // ------------------------------------------------------------------
    // Graph mutation
    // ------------------------------------------------------------------

    int add_node(double x, double y, JunctionType jtype) {
        nodes_.push_back({x, y, jtype, 0});
        adjacency_.emplace_back();
        return static_cast<int>(nodes_.size()) - 1;
    }

    int add_edge(int u, int v, double length, double heading) {
        EdgeData e;
        e.u = u; e.v = v;
        e.length = length; e.heading = heading;
        edges_.push_back(e);
        int eid = static_cast<int>(edges_.size()) - 1;
        adjacency_[u].push_back(eid);
        adjacency_[v].push_back(eid);
        return eid;
    }

    void mark_edge_status(int eid, EdgeStatus status) {
        edges_[eid].status = status;
        edges_[eid].traversal_count++;
    }

    void mark_node_visited(int nid) { nodes_[nid].visit_count++; }

    // ------------------------------------------------------------------
    // Dead-reckoning position update
    // ------------------------------------------------------------------

    void update_odometry(double delta_dist, double heading) {
        heading_ = heading;
        x_ += delta_dist * std::cos(heading);
        y_ += delta_dist * std::sin(heading);
    }

    // ------------------------------------------------------------------
    // Loop closure: nearest existing node within CLOSURE_DIST, else nullopt
    // ------------------------------------------------------------------

    std::optional<int> try_loop_closure(double x, double y) const {
        std::optional<int> best;
        double best_dist = CLOSURE_DIST;
        for (size_t i = 0; i < nodes_.size(); ++i) {
            double d = std::hypot(x - nodes_[i].x, y - nodes_[i].y);
            if (d < best_dist) {
                best_dist = d;
                best = static_cast<int>(i);
            }
        }
        return best;
    }

    // ------------------------------------------------------------------
    // Queries
    // ------------------------------------------------------------------

    std::vector<int> uninspected_edges() const {
        std::vector<int> out;
        for (size_t i = 0; i < edges_.size(); ++i)
            if (edges_[i].status != EdgeStatus::INSPECTED)
                out.push_back(static_cast<int>(i));
        return out;
    }

    int num_nodes() const { return static_cast<int>(nodes_.size()); }
    int num_edges() const { return static_cast<int>(edges_.size()); }

    const NodeData& node(int nid) const { return nodes_[nid]; }
    const EdgeData& edge(int eid) const { return edges_[eid]; }
    const std::vector<int>& incident_edges(int nid) const { return adjacency_[nid]; }

    int degree(int nid) const { return static_cast<int>(adjacency_[nid].size()); }

    int other_end(int eid, int from) const {
        const EdgeData& e = edges_[eid];
        return (e.u == from) ? e.v : e.u;
    }

    // Robot pose accessors
    double x() const { return x_; }
    double y() const { return y_; }
    double heading() const { return heading_; }
    std::optional<int> current_node() const { return current_node_; }
    void set_current_node(int nid) { current_node_ = nid; }

private:
    std::vector<NodeData> nodes_;
    std::vector<EdgeData> edges_;
    std::vector<std::vector<int>> adjacency_;

    double x_ = 0.0, y_ = 0.0, heading_ = 0.0;
    std::optional<int> current_node_;
};

}  // namespace pipe
