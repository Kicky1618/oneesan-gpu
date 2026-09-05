#pragma push_macro("main")
#undef main
#define main two_cell_component_size_distribution_probe_main_unused
#include "two_cell_component_size_distribution_probe.cpp"
#pragma pop_macro("main")

namespace {

// Number of connectivity strands bordering the face incident to the boundary
// interval immediately before symbol `mark` in a one-defect Motzkin word.
//
// Let h[mark] be the level of that interval. The same face extends maximally
// left/right while the path stays at or above this level. Its boundary consists
// of the top-level excursions inside this interval, optionally the enclosing
// arc, or the distinguished root strand on the outer face.
int marked_face_strands(const Word& v, int mark) {
    const int n = static_cast<int>(v.size());
    if (mark < 0 || mark >= n) fail("marked face position");

    std::vector<int> h(static_cast<std::size_t>(n + 1));
    h[0] = 1;
    for (int pos = 0; pos < n; ++pos) {
        h[pos + 1] = h[pos];
        if (v[pos] == L) ++h[pos + 1];
        else if (v[pos] == R) --h[pos + 1];
    }
    const int level = h[mark];

    int left = mark;
    while (left > 0 && h[left - 1] >= level) --left;
    int right = mark;
    while (right < n && h[right + 1] >= level) ++right;

    const LinkState s = decode(v);
    int strands = 0;
    for (int p = left; p < right; ++p) {
        if (v[p] != L || h[p] != level) continue;
        const int q = s.mate[p];
        if (q > p && q < right) ++strands;
    }

    if (left > 0) {
        const int p = left - 1;
        if (v[p] == L && s.mate[p] == right) ++strands;
    }

    if (level == 0 || (left == 0 && right < n && s.root == right)) ++strands;
    if (strands <= 0) fail("marked face without strand");
    return strands;
}

int occupied_strands(const Word& v) {
    int occupied = 0;
    for (char c : v) occupied += c != N;
    if (!(occupied & 1)) fail("one-defect occupied parity");
    return (occupied + 1) / 2; // matched arcs plus distinguished root
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 12;
    if (maxW < 5 || maxW > 15) return 2;

    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 5; W <= maxW; ++W) {
        Rank checked_deep = 0;
        Rank max_pairs = 0;
        int max_face = 0;
        int max_total_strands = 0;

        for (int i = 0; i <= W - 4; ++i) {
            for (const Word& u : words[W - 2]) {
                const auto collapsed = deep_collapse_label(u, i);
                if (!collapsed) continue;

                const Key seed = project_key(Key{'C', u}, i, W);
                const auto component = packed_component_sources(pack_key(seed), W, i);
                const int face = marked_face_strands(*collapsed, i);
                const int total = occupied_strands(*collapsed);
                if (face > total)
                    fail("face exceeds total strands W=" + std::to_string(W));
                if (component.size != 4 + face)
                    fail("component/face formula W=" + std::to_string(W) +
                         " i=" + std::to_string(i));

                ++checked_deep;
                max_pairs = std::max<Rank>(max_pairs, component.size);
                max_face = std::max(max_face, face);
                max_total_strands = std::max(max_total_strands, total);
            }
        }

        const int face_bound = (W - 2) / 2;
        const Rank pair_bound = Rank(W / 2 + 3);
        if (max_face > face_bound || max_pairs > pair_bound)
            fail("marked face bound W=" + std::to_string(W));

        std::cout << "W=" << W
                  << " checked_deep=" << checked_deep
                  << " max_face_strands=" << max_face
                  << " max_total_strands=" << max_total_strands
                  << " max_pairs=" << max_pairs
                  << " pair_bound=floor(W/2)+3"
                  << " local_face_formula=1"
                  << " OK\n";
    }

    std::cout << "W=28_theory face_strands_max=13"
              << " component_pairs_max=17"
              << " u32_component_payload_max_bytes=68"
              << " warp_capacity=32"
              << "\n";
    std::cout << "ALL_OK marked_face_component_size=1\n";
    return 0;
}
