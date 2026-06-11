// Chinese Postman Problem (CPP) solver for pipe inspection (C++17, header-only).
//
// Algorithm:
//   1. Find odd-degree nodes among the required edges.
//   2. All-pairs shortest paths between odd nodes (Dijkstra).
//   3. Minimum-weight perfect matching — exact bitmask DP, O(2^n * n^2).
//      Pipe networks rarely exceed ~20 odd nodes; DP handles up to 24.
//   4. Duplicate edges along matched shortest paths (deadhead edges).
//   5. Hierholzer's algorithm for the Eulerian circuit.
//
// Mirrors the Python cpp_planner.py implementation.

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
#include <vector>

#include "graph_map.hpp"

namespace pipe {

struct PathSegment {
    int from_node;
    int to_node;
    int edge_id;       // index into TopologicalMap edges (original edge)
    bool deadhead;     // true = duplicate traversal added by the matching
};

struct PathStats {
    double total_m = 0.0;
    double inspection_m = 0.0;
    double deadhead_m = 0.0;
    double efficiency_pct = 0.0;
    int segments = 0;
};

class ChinesePostmanPlanner {
public:
    // Solve CPP on the map.  Returns traversal order starting and ending at
    // start_node.  Throws std::runtime_error if the required edges are not
    // connected to start_node.
    std::vector<PathSegment> solve(const TopologicalMap& map,
                                   int start_node,
                                   bool only_uninspected = true) const {
        const int N = map.num_nodes();
        if (N == 0) return {};

        // ---- Collect required edges -----------------------------------
        std::vector<bool> required(map.num_edges(), false);
        bool any_required = false;
        for (int e = 0; e < map.num_edges(); ++e) {
            bool req = !only_uninspected ||
                       map.edge(e).status != EdgeStatus::INSPECTED;
            required[e] = req;
            any_required |= req;
        }
        if (!any_required) return {};

        // ---- Working multigraph: copy of all edges; deadhead list ------
        // walk_edges holds (edge_id, deadhead) pairs forming the multiset
        // of edges that the Eulerian circuit must cover.
        struct WalkEdge { int eid; bool deadhead; bool used = false; };
        std::vector<WalkEdge> walk;
        for (int e = 0; e < map.num_edges(); ++e)
            if (required[e]) walk.push_back({e, false});

        // ---- Step 1: odd-degree nodes in the required subgraph ---------
        std::vector<int> deg(N, 0);
        for (const auto& w : walk) {
            deg[map.edge(w.eid).u]++;
            deg[map.edge(w.eid).v]++;
        }
        std::vector<int> odd;
        for (int v = 0; v < N; ++v)
            if (deg[v] % 2 == 1) odd.push_back(v);

        // ---- Steps 2-4: matching + deadhead duplication -----------------
        if (!odd.empty()) {
            const int K = static_cast<int>(odd.size());
            if (K > 24)
                throw std::runtime_error(
                    "Too many odd-degree nodes for exact matching (>24)");

            // Pairwise shortest paths between odd nodes (over ALL edges,
            // deadheads may shortcut through inspected pipes).
            std::vector<std::vector<double>> dist(K);
            std::vector<std::vector<std::vector<int>>> spath(K);  // edge ids
            for (int i = 0; i < K; ++i) {
                auto [d, paths] = dijkstra_(map, odd[i], odd);
                dist[i] = std::move(d);
                spath[i] = std::move(paths);
            }

            // Bitmask DP minimum-weight perfect matching
            const int FULL = (1 << K) - 1;
            constexpr double INF = std::numeric_limits<double>::infinity();
            std::vector<double> dp(FULL + 1, INF);
            std::vector<std::pair<int,int>> choice(FULL + 1, {-1, -1});
            dp[0] = 0.0;

            for (int mask = 0; mask <= FULL; ++mask) {
                if (dp[mask] == INF) continue;
                int i = 0;
                while (i < K && (mask & (1 << i))) ++i;   // first unmatched
                if (i >= K) continue;
                for (int j = i + 1; j < K; ++j) {
                    if (mask & (1 << j)) continue;
                    if (dist[i][j] == INF) continue;
                    int nmask = mask | (1 << i) | (1 << j);
                    double cost = dp[mask] + dist[i][j];
                    if (cost < dp[nmask]) {
                        dp[nmask] = cost;
                        choice[nmask] = {i, j};
                    }
                }
            }
            if (dp[FULL] == INF)
                throw std::runtime_error("Odd nodes not mutually reachable");

            // Recover matching, add deadhead edges
            int mask = FULL;
            while (mask) {
                auto [i, j] = choice[mask];
                for (int eid : spath[i][j])
                    walk.push_back({eid, true});
                mask &= ~((1 << i) | (1 << j));
            }
        }

        // ---- Step 5: Hierholzer's Eulerian circuit ----------------------
        return hierholzer_(map, walk, start_node);
    }

    PathStats stats(const std::vector<PathSegment>& path,
                    const TopologicalMap& map) const {
        PathStats s;
        for (const auto& seg : path) {
            double len = map.edge(seg.edge_id).length;
            s.total_m += len;
            if (seg.deadhead) s.deadhead_m += len;
        }
        s.inspection_m = s.total_m - s.deadhead_m;
        s.segments = static_cast<int>(path.size());
        s.efficiency_pct =
            s.total_m > 0 ? 100.0 * s.inspection_m / s.total_m : 0.0;
        return s;
    }

private:
    // Dijkstra from src; returns (distance-to-each-odd-node,
    // edge-id path to each odd node).
    static std::pair<std::vector<double>, std::vector<std::vector<int>>>
    dijkstra_(const TopologicalMap& map, int src,
              const std::vector<int>& targets) {
        const int N = map.num_nodes();
        constexpr double INF = std::numeric_limits<double>::infinity();
        std::vector<double> d(N, INF);
        std::vector<int> prev_edge(N, -1);
        d[src] = 0.0;

        using QE = std::pair<double, int>;
        std::priority_queue<QE, std::vector<QE>, std::greater<>> pq;
        pq.push({0.0, src});

        while (!pq.empty()) {
            auto [du, u] = pq.top(); pq.pop();
            if (du > d[u]) continue;
            for (int eid : map.incident_edges(u)) {
                int v = map.other_end(eid, u);
                double nd = du + map.edge(eid).length;
                if (nd < d[v]) {
                    d[v] = nd;
                    prev_edge[v] = eid;
                    pq.push({nd, v});
                }
            }
        }

        const int K = static_cast<int>(targets.size());
        std::vector<double> tdist(K);
        std::vector<std::vector<int>> tpath(K);
        for (int k = 0; k < K; ++k) {
            int t = targets[k];
            tdist[k] = d[t];
            if (d[t] == INF || t == src) continue;
            // walk back along prev_edge
            int cur = t;
            while (cur != src) {
                int eid = prev_edge[cur];
                tpath[k].push_back(eid);
                cur = map.other_end(eid, cur);
            }
            std::reverse(tpath[k].begin(), tpath[k].end());
        }
        return {std::move(tdist), std::move(tpath)};
    }

    struct WalkEdgeRef { int eid; bool deadhead; bool used; };

    template <typename WalkVec>
    static std::vector<PathSegment> hierholzer_(const TopologicalMap& map,
                                                WalkVec& walk,
                                                int start) {
        const int N = map.num_nodes();

        // adjacency over walk indices
        std::vector<std::vector<int>> adj(N);
        for (size_t w = 0; w < walk.size(); ++w) {
            const EdgeData& e = map.edge(walk[w].eid);
            adj[e.u].push_back(static_cast<int>(w));
            adj[e.v].push_back(static_cast<int>(w));
        }
        std::vector<size_t> ptr(N, 0);

        // If start has no required/deadhead edges, fall back to any node
        // that does.
        if (adj[start].empty()) {
            for (int v = 0; v < N; ++v)
                if (!adj[v].empty()) { start = v; break; }
        }

        std::vector<int> circuit_nodes;          // node sequence
        std::vector<int> circuit_walk;           // walk-edge index sequence
        std::vector<std::pair<int,int>> stack;   // (node, walk idx taken to get here)
        stack.push_back({start, -1});

        while (!stack.empty()) {
            auto [v, via] = stack.back();
            // find unused incident walk edge
            bool advanced = false;
            while (ptr[v] < adj[v].size()) {
                int w = adj[v][ptr[v]++];
                if (walk[w].used) continue;
                walk[w].used = true;
                int next = map.other_end(walk[w].eid, v);
                stack.push_back({next, w});
                advanced = true;
                break;
            }
            if (!advanced) {
                circuit_nodes.push_back(v);
                circuit_walk.push_back(via);
                stack.pop_back();
            }
        }

        // circuit is built in reverse; convert to PathSegment list
        std::reverse(circuit_nodes.begin(), circuit_nodes.end());
        std::reverse(circuit_walk.begin(), circuit_walk.end());

        std::vector<PathSegment> out;
        for (size_t i = 0; i + 1 < circuit_nodes.size(); ++i) {
            int w = circuit_walk[i + 1];
            out.push_back({circuit_nodes[i], circuit_nodes[i + 1],
                           walk[w].eid, walk[w].deadhead});
        }
        return out;
    }
};

}  // namespace pipe
