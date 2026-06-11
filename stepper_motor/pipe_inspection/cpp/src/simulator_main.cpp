// Pipe Inspection Simulator — C++ command-line version for Linux.
//
// Builds the topological map from a virtual pipe network, runs the CPP
// planner, and prints the coverage path.  Mirrors simulator.py.
//
// Usage:
//   pipe_simulator [simple|medium|complex]

#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "cpp_planner.hpp"
#include "graph_map.hpp"

namespace {

constexpr const char* GREEN  = "\033[92m";
constexpr const char* YELLOW = "\033[93m";
constexpr const char* CYAN   = "\033[96m";
constexpr const char* BOLD   = "\033[1m";
constexpr const char* RESET  = "\033[0m";

struct Segment {
    std::string from, to;
    double length;
    double heading_deg;
};

const std::map<std::string, std::vector<Segment>> NETWORKS = {
    {"simple", {
        {"START", "A", 2.0,   0}, {"A", "B", 1.5,   0},
        {"A",     "C", 1.0,  90}, {"B", "D", 1.0,   0},
        {"C",     "E", 1.2,  90},
    }},
    {"medium", {
        {"START", "A", 1.0,   0}, {"A", "B", 2.0,   0},
        {"B",     "C", 1.5,  90}, {"C", "D", 2.0, 180},
        {"D",     "A", 1.5, 270}, {"B", "E", 1.0,   0},
        {"C",     "F", 0.8,   0}, {"F", "G", 1.0,  90},
        {"F",     "H", 1.2, 270},
    }},
    {"complex", {
        {"START", "A", 1.0,   0}, {"A", "B", 2.0,   0},
        {"B",     "C", 2.0,   0}, {"C", "D", 2.0,   0},
        {"A",     "E", 2.0,  90}, {"B", "F", 2.0,  90},
        {"C",     "G", 2.0,  90}, {"E", "F", 2.0,   0},
        {"F",     "G", 2.0,   0}, {"E", "H", 2.0,  90},
        {"G",     "I", 2.0,  90}, {"H", "I", 2.0,   0},
        {"I",     "J", 1.0,  90}, {"F", "K", 1.0, 270},
    }},
};

}  // namespace

int main(int argc, char** argv) {
    std::string net_name = (argc > 1) ? argv[1] : "medium";
    auto it = NETWORKS.find(net_name);
    if (it == NETWORKS.end()) {
        std::fprintf(stderr, "Unknown network '%s' (simple|medium|complex)\n",
                     net_name.c_str());
        return 1;
    }
    const auto& network = it->second;

    std::printf("%s%s\n== Pipe Inspection Simulator (C++) — network: %s ==%s\n",
                BOLD, CYAN, net_name.c_str(), RESET);

    // ── Phase 1: build map ───────────────────────────────────────────
    std::printf("%s\n[Phase 1] Exploration%s\n", YELLOW, RESET);

    pipe::TopologicalMap tmap;
    std::map<std::string, int> label_to_id;
    std::map<int, std::string> id_to_label;
    std::map<std::string, std::pair<double,double>> pos;
    std::map<std::string, int> adjacency_count;

    pos["START"] = {0.0, 0.0};
    for (const auto& s : network) {
        adjacency_count[s.from]++;
        adjacency_count[s.to]++;
        if (!pos.count(s.from)) pos[s.from] = {0.0, 0.0};
        auto [fx, fy] = pos[s.from];
        double rad = s.heading_deg * M_PI / 180.0;
        if (!pos.count(s.to))
            pos[s.to] = {fx + s.length * std::cos(rad),
                         fy + s.length * std::sin(rad)};
    }

    auto get_node = [&](const std::string& label) -> int {
        auto f = label_to_id.find(label);
        if (f != label_to_id.end()) return f->second;
        pipe::JunctionType t =
            label == "START" ? pipe::JunctionType::START :
            adjacency_count[label] == 1 ? pipe::JunctionType::DEAD_END :
            pipe::JunctionType::JUNCTION;
        auto [x, y] = pos[label];
        int nid = tmap.add_node(x, y, t);
        label_to_id[label] = nid;
        id_to_label[nid] = label;
        return nid;
    };

    for (const auto& s : network) {
        int u = get_node(s.from);
        int v = get_node(s.to);
        int eid = tmap.add_edge(u, v, s.length,
                                s.heading_deg * M_PI / 180.0);
        tmap.mark_edge_status(eid, pipe::EdgeStatus::EXPLORED);
        std::printf("  %s(%d) --%.1fm--> %s(%d)\n",
                    s.from.c_str(), u, s.length, s.to.c_str(), v);
    }

    std::printf("%s\n  Map built: %d nodes, %d edges%s\n",
                GREEN, tmap.num_nodes(), tmap.num_edges(), RESET);

    std::printf("\n%s  Node list:%s\n", BOLD, RESET);
    for (int n = 0; n < tmap.num_nodes(); ++n) {
        const auto& nd = tmap.node(n);
        std::printf("    node %d [%-6s] %-10s pos=(%5.1f, %5.1f)  deg=%d\n",
                    n, id_to_label[n].c_str(), pipe::to_string(nd.jtype),
                    nd.x, nd.y, tmap.degree(n));
    }

    // ── Phase 2: CPP ─────────────────────────────────────────────────
    std::printf("%s\n[Phase 2] Chinese Postman Problem — coverage path%s\n",
                YELLOW, RESET);

    pipe::ChinesePostmanPlanner planner;
    int start = label_to_id["START"];

    auto t0 = std::chrono::steady_clock::now();
    std::vector<pipe::PathSegment> path;
    try {
        path = planner.solve(tmap, start, /*only_uninspected=*/false);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Planner error: %s\n", e.what());
        return 1;
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto st = planner.stats(path, tmap);

    std::printf("%s\n  CPP solved in %.2f ms%s\n", GREEN, ms, RESET);
    std::printf("  Segments       : %d\n", st.segments);
    std::printf("  Total distance : %.1f m\n", st.total_m);
    std::printf("  Inspection dist: %.1f m\n", st.inspection_m);
    std::printf("  Deadhead dist  : %.1f m\n", st.deadhead_m);
    std::printf("  Efficiency     : %.1f %%\n", st.efficiency_pct);

    std::printf("\n%s  Traversal order:%s\n", BOLD, RESET);
    int i = 0;
    for (const auto& seg : path) {
        double len = tmap.edge(seg.edge_id).length;
        std::printf("  %3d. %-6s -> %-6s (%.1f m) %s%s%s\n",
                    ++i,
                    id_to_label[seg.from_node].c_str(),
                    id_to_label[seg.to_node].c_str(),
                    len,
                    seg.deadhead ? YELLOW : "",
                    seg.deadhead ? "[deadhead]" : "",
                    seg.deadhead ? RESET : "");
    }

    std::printf("%s%s\n== Done ==%s\n\n", BOLD, CYAN, RESET);
    return 0;
}
