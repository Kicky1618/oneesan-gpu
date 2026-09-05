#pragma push_macro("main")
#undef main
#define main two_cell_component_device_probe_main_unused
#include "two_cell_component_device_probe.cpp"
#pragma pop_macro("main")

#include <map>
#include <set>

namespace {

struct PrimitiveSignature {
    int occupied = 0;
    std::uint32_t compact_left = 0;

    bool operator<(const PrimitiveSignature& o) const {
        return occupied < o.occupied ||
               (occupied == o.occupied && compact_left < o.compact_left);
    }
    bool operator==(const PrimitiveSignature& o) const {
        return occupied == o.occupied && compact_left == o.compact_left;
    }
};

PrimitiveSignature primitive_signature(const Key& k) {
    PrimitiveSignature sig;
    int ordinal = 0;
    for (char c : k.w) {
        if (c == N) continue;
        if (c == L) sig.compact_left |= std::uint32_t(1) << ordinal;
        ++ordinal;
    }
    sig.occupied = ordinal;
    return sig;
}

PrimitiveSignature primitive_signature(const Word& w) {
    return primitive_signature(Key{'C', w});
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 13;
    if (maxW < 5 || maxW > 15) return 2;

    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 5; W <= maxW; ++W) {
        Rank components = 0;
        Rank source_states = 0;
        Rank destination_states = 0;
        Rank source_signatures = 0;
        Rank destination_signatures = 0;
        Rank source_label_signature_hits = 0;
        Rank destination_label_signature_hits = 0;
        Rank max_source_signatures = 0;
        Rank max_destination_signatures = 0;
        Rank max_union_signatures = 0;

        for (int i = 0; i <= W - 4; ++i) {
            for (const Word& u : words[W - 2]) {
                const PrimitiveSignature label_sig = primitive_signature(u);
                const auto packed_sources = packed_direct_component_sources(pack_word(u), W, i);

                std::set<Key> src;
                std::set<Key> dst;
                for (int q = 0; q < packed_sources.size; ++q) {
                    const Key s = unpack_key(packed_sources.value[q]);
                    src.insert(s);
                    for (const auto& [d, c] : K_basis(s, W, i)) {
                        if (c != 1) fail("signature nonunit edge");
                        dst.insert(d);
                    }
                }
                if (src.size() != dst.size()) fail("signature unbalanced component");

                std::set<PrimitiveSignature> ss, ds, us;
                for (const Key& s : src) {
                    const auto sig = primitive_signature(s);
                    ss.insert(sig);
                    us.insert(sig);
                    if (sig == label_sig) ++source_label_signature_hits;
                }
                for (const Key& d : dst) {
                    const auto sig = primitive_signature(d);
                    ds.insert(sig);
                    us.insert(sig);
                    if (sig == label_sig) ++destination_label_signature_hits;
                }

                source_states += src.size();
                destination_states += dst.size();
                source_signatures += ss.size();
                destination_signatures += ds.size();
                max_source_signatures = std::max<Rank>(max_source_signatures, ss.size());
                max_destination_signatures = std::max<Rank>(max_destination_signatures, ds.size());
                max_union_signatures = std::max<Rank>(max_union_signatures, us.size());
                ++components;
            }
        }

        std::cout << "W=" << W
                  << " components=" << components
                  << " avg_source_states=" << double(source_states) / double(components)
                  << " avg_source_signatures=" << double(source_signatures) / double(components)
                  << " avg_destination_signatures=" << double(destination_signatures) / double(components)
                  << " source_signature_reuse=" << double(source_states) / double(source_signatures)
                  << " destination_signature_reuse=" << double(destination_states) / double(destination_signatures)
                  << " source_label_hit_fraction="
                  << double(source_label_signature_hits) / double(source_states)
                  << " destination_label_hit_fraction="
                  << double(destination_label_signature_hits) / double(destination_states)
                  << " max_source_signatures=" << max_source_signatures
                  << " max_destination_signatures=" << max_destination_signatures
                  << " max_union_signatures=" << max_union_signatures
                  << " OK\n";
    }

    std::cout << "ALL_OK component_primitive_signature_reuse=1\n";
    return 0;
}
