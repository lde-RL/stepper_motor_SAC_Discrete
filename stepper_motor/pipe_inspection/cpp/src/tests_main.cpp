// Unit tests for graph_map.hpp / cpp_planner.hpp — plain asserts.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>

#include "cpp_planner.hpp"
#include "graph_map.hpp"

using namespace pipe;

static int tests_run = 0;
#define CHECK(cond) do { \
    tests_run++; \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        return 1; \
    } \
} while (0)

// Validate that a CPP solution is a contiguous closed walk from start that
// covers every required edge at least once.
static bool validate_circuit(const TopologicalMap& m,
                             const std::vector<PathSegment>& path,
                             int start) {
    if (path.empty()) return false;
    if (path.front().from_node != start) return false;
    if (path.back().to_node != start) return false;

    std::multiset<int> covered;
    int cur = start;
    for (const auto& seg : path) {
        if (seg.from_node != cur) return false;
        const EdgeData& e = m.edge(seg.edge_id);
        // segment must use a real edge between these two nodes
        bool matches = (e.u == seg.from_node && e.v == seg.to_node) ||
                       (e.v == seg.from_node && e.u == seg.to_node);
        if (!matches) return false;
        covered.insert(seg.edge_id);
        cur = seg.to_node;
    }
    // every edge covered at least once
    for (int e = 0; e < m.num_edges(); ++e)
        if (covered.count(e) == 0) return false;
    return true;
}

int test_loop_closure() {
    TopologicalMap m;
    int a = m.add_node(0.0, 0.0, JunctionType::START);
    m.add_node(5.0, 0.0, JunctionType::JUNCTION);

    auto hit = m.try_loop_closure(0.1, 0.1);
    CHECK(hit.has_value() && *hit == a);

    auto miss = m.try_loop_closure(2.5, 2.5);
    CHECK(!miss.has_value());
    return 0;
}

int test_odometry() {
    TopologicalMap m;
    m.update_odometry(1.0, 0.0);          // 1 m east
    m.update_odometry(2.0, M_PI / 2.0);   // 2 m north
    CHECK(std::abs(m.x() - 1.0) < 1e-9);
    CHECK(std::abs(m.y() - 2.0) < 1e-9);
    return 0;
}

// Eulerian graph (all even degrees) — no deadhead needed.
int test_cpp_even_graph() {
    TopologicalMap m;
    int a = m.add_node(0, 0, JunctionType::START);
    int b = m.add_node(1, 0, JunctionType::JUNCTION);
    int c = m.add_node(1, 1, JunctionType::JUNCTION);
    int d = m.add_node(0, 1, JunctionType::JUNCTION);
    m.add_edge(a, b, 1.0, 0);
    m.add_edge(b, c, 1.0, 0);
    m.add_edge(c, d, 1.0, 0);
    m.add_edge(d, a, 1.0, 0);

    ChinesePostmanPlanner p;
    auto path = p.solve(m, a, false);
    CHECK(validate_circuit(m, path, a));
    auto st = p.stats(path, m);
    CHECK(std::abs(st.total_m - 4.0) < 1e-9);
    CHECK(std::abs(st.deadhead_m) < 1e-9);
    return 0;
}

// Single dead-end branch — must be traversed twice (once deadhead).
int test_cpp_dead_end() {
    TopologicalMap m;
    int a = m.add_node(0, 0, JunctionType::START);
    int b = m.add_node(2, 0, JunctionType::DEAD_END);
    m.add_edge(a, b, 2.0, 0);

    ChinesePostmanPlanner p;
    auto path = p.solve(m, a, false);
    CHECK(validate_circuit(m, path, a));
    auto st = p.stats(path, m);
    CHECK(st.segments == 2);
    CHECK(std::abs(st.total_m - 4.0) < 1e-9);
    CHECK(std::abs(st.deadhead_m - 2.0) < 1e-9);
    return 0;
}

// T-junction: START-A trunk plus two branches off A.
int test_cpp_t_junction() {
    TopologicalMap m;
    int s = m.add_node(0, 0, JunctionType::START);
    int a = m.add_node(2, 0, JunctionType::JUNCTION);
    int b = m.add_node(4, 0, JunctionType::DEAD_END);
    int c = m.add_node(2, 2, JunctionType::DEAD_END);
    m.add_edge(s, a, 2.0, 0);
    m.add_edge(a, b, 2.0, 0);
    m.add_edge(a, c, 2.0, 0);

    ChinesePostmanPlanner p;
    auto path = p.solve(m, s, false);
    CHECK(validate_circuit(m, path, s));
    auto st = p.stats(path, m);
    // every edge twice: 12 m total, 6 m deadhead
    CHECK(std::abs(st.total_m - 12.0) < 1e-9);
    CHECK(std::abs(st.deadhead_m - 6.0) < 1e-9);
    return 0;
}

// only_uninspected: already-inspected edges are not required.
int test_only_uninspected() {
    TopologicalMap m;
    int a = m.add_node(0, 0, JunctionType::START);
    int b = m.add_node(1, 0, JunctionType::JUNCTION);
    int c = m.add_node(2, 0, JunctionType::DEAD_END);
    int e1 = m.add_edge(a, b, 1.0, 0);
    m.add_edge(b, c, 1.0, 0);
    m.mark_edge_status(e1, EdgeStatus::INSPECTED);

    ChinesePostmanPlanner p;
    auto path = p.solve(m, b, true);   // start at b, only b-c required
    CHECK(!path.empty());
    bool covered_bc = false;
    for (const auto& seg : path)
        if (seg.edge_id == 1) covered_bc = true;
    CHECK(covered_bc);
    return 0;
}

// Larger network: 'medium' from the simulator, sanity check totals.
int test_cpp_medium_network() {
    TopologicalMap m;
    // ids: 0=START 1=A 2=B 3=C 4=D 5=E 6=F 7=G 8=H
    for (int i = 0; i < 9; ++i)
        m.add_node(i, 0, i == 0 ? JunctionType::START : JunctionType::JUNCTION);
    m.add_edge(0, 1, 1.0, 0);
    m.add_edge(1, 2, 2.0, 0);
    m.add_edge(2, 3, 1.5, 0);
    m.add_edge(3, 4, 2.0, 0);
    m.add_edge(4, 1, 1.5, 0);
    m.add_edge(2, 5, 1.0, 0);
    m.add_edge(3, 6, 0.8, 0);
    m.add_edge(6, 7, 1.0, 0);
    m.add_edge(6, 8, 1.2, 0);

    ChinesePostmanPlanner p;
    auto path = p.solve(m, 0, false);
    CHECK(validate_circuit(m, path, 0));
    auto st = p.stats(path, m);
    CHECK(std::abs(st.inspection_m - 12.0) < 1e-9);
    // optimal deadhead for this network = 5.0 m (matches Python solver)
    CHECK(std::abs(st.deadhead_m - 5.0) < 1e-9);
    return 0;
}

int main() {
    if (test_loop_closure())       return 1;
    if (test_odometry())           return 1;
    if (test_cpp_even_graph())     return 1;
    if (test_cpp_dead_end())       return 1;
    if (test_cpp_t_junction())     return 1;
    if (test_only_uninspected())   return 1;
    if (test_cpp_medium_network()) return 1;

    std::printf("All %d checks passed.\n", tests_run);
    return 0;
}
