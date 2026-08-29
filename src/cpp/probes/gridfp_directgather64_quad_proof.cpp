#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

namespace {
using U64 = std::uint64_t;

struct Desc {
    std::array<std::uint16_t, 7> r{};
    std::uint32_t n = 0;
};

U64 pack_primary(const Desc& d, std::uint16_t rare_ix) {
    return U64(d.r[0]) | (U64(d.r[1]) << 15) | (U64(d.r[2]) << 30) |
           (U64(d.n) << 45) | (U64(rare_ix) << 48);
}
U64 pack_rare(const Desc& d) {
    return U64(d.r[3]) | (U64(d.r[4]) << 15) |
           (U64(d.r[5]) << 30) | (U64(d.r[6]) << 45);
}

Desc decode(U64 p, const std::vector<U64>& rare) {
    Desc d{};
    d.n = std::uint32_t((p >> 45) & 7u);
    d.r[0] = std::uint16_t(p & 0x7fffu);
    d.r[1] = std::uint16_t((p >> 15) & 0x7fffu);
    d.r[2] = std::uint16_t((p >> 30) & 0x7fffu);
    if (d.n > 3) {
        const U64 q = rare.at(std::uint32_t(p >> 48));
        d.r[3] = std::uint16_t(q & 0x7fffu);
        d.r[4] = std::uint16_t((q >> 15) & 0x7fffu);
        d.r[5] = std::uint16_t((q >> 30) & 0x7fffu);
        d.r[6] = std::uint16_t((q >> 45) & 0x7fffu);
    }
    return d;
}

U64 sum_desc(const Desc& d, const std::vector<std::uint32_t>& source) {
    U64 s = 0;
    for (std::uint32_t i = 0; i < d.n; ++i) s += source.at(d.r[i]);
    return s;
}

void fail(const char* what) {
    std::cerr << "gridfp_directgather64_quad_proof FAIL: " << what << '\n';
    std::exit(1);
}

void prove_scheduler_tail() {
    for (std::uint32_t cols = 1; cols <= 4097; ++cols) {
        std::vector<int> seen(cols, 0);
        const std::uint32_t step = 97;
        for (std::uint32_t lane_base = 0; lane_base < step; ++lane_base) {
            for (std::uint32_t base = lane_base; base < cols; base += step * 4u) {
                std::array<std::uint32_t,4> lr{};
                std::array<bool,4> live{};
                for (int t=0;t<4;++t) {
                    lr[t]=base+std::uint32_t(t)*step;
                    live[t]=lr[t]<cols;
                }
                if (live[3]) {
                    for (int t=0;t<4;++t) ++seen[lr[t]];
                } else {
                    for (int t=0;t<4;++t) if (live[t]) ++seen[lr[t]];
                }
            }
        }
        for (int x:seen) if (x!=1) fail("quad/tail scheduler coverage");
    }
}

void prove_dense_sparse_quad() {
    std::mt19937_64 rng(0x51f15e5dULL);
    constexpr std::uint32_t kRanks = 32768;
    std::vector<std::uint32_t> source(kRanks);
    for (auto& x:source) x=std::uint32_t(rng());

    for (std::uint32_t ndesc : {1u,2u,31u,32u,33u,100u,1000u,4097u}) {
        std::vector<Desc> ref(ndesc);
        for (auto& d:ref) {
            d.n=std::uint32_t(rng()%8u);
            for (std::uint32_t i=0;i<d.n;++i) d.r[i]=std::uint16_t(rng()%kRanks);
        }

        std::vector<U64> dense(ndesc), dense_rare;
        for (std::uint32_t i=0;i<ndesc;++i) {
            std::uint16_t ri=0;
            if (ref[i].n>3) { ri=std::uint16_t(dense_rare.size()); dense_rare.push_back(pack_rare(ref[i])); }
            dense[i]=pack_primary(ref[i],ri);
        }

        const std::size_t words=(ndesc+31u)/32u;
        std::vector<U64> index(words);
        std::vector<U64> sparse_primary, sparse_rare;
        std::uint32_t prefix=0;
        for (std::size_t w=0;w<words;++w) {
            std::uint32_t bits=0;
            for (std::uint32_t b=0;b<32;++b) {
                const std::uint32_t i=std::uint32_t(w*32u+b);
                if (i<ndesc && ref[i].n) bits|=1u<<b;
            }
            index[w]=U64(bits)|(U64(prefix)<<32);
            prefix+=std::uint32_t(__builtin_popcount(bits));
        }
        sparse_primary.resize(prefix);
        for (std::uint32_t i=0;i<ndesc;++i) if (ref[i].n) {
            const std::uint32_t bit=i&31u, bits=std::uint32_t(index[i>>5]);
            const std::uint32_t lower=bit ? bits&((1u<<bit)-1u) : 0u;
            const std::uint32_t ci=std::uint32_t(index[i>>5]>>32)+std::uint32_t(__builtin_popcount(lower));
            std::uint16_t ri=0;
            if (ref[i].n>3) { ri=std::uint16_t(sparse_rare.size()); sparse_rare.push_back(pack_rare(ref[i])); }
            sparse_primary[ci]=pack_primary(ref[i],ri);
        }

        auto sparse_desc=[&](std::uint32_t i) {
            const std::uint32_t bit=i&31u, bits=std::uint32_t(index[i>>5]);
            const std::uint32_t flag=1u<<bit;
            if (!(bits&flag)) return Desc{};
            const std::uint32_t lower=bit ? bits&(flag-1u) : 0u;
            const std::uint32_t ci=std::uint32_t(index[i>>5]>>32)+std::uint32_t(__builtin_popcount(lower));
            return decode(sparse_primary.at(ci),sparse_rare);
        };

        for (int trial=0;trial<10000;++trial) {
            std::array<std::uint32_t,4> ix{};
            for (auto& i:ix) i=std::uint32_t(rng()%ndesc);
            for (std::uint32_t j=0;j<4;++j) {
                const Desc dd=decode(dense[ix[j]],dense_rare);
                const Desc sd=sparse_desc(ix[j]);
                if (dd.n!=ref[ix[j]].n || sd.n!=ref[ix[j]].n) fail("descriptor count");
                for (std::uint32_t k=0;k<ref[ix[j]].n;++k)
                    if (dd.r[k]!=ref[ix[j]].r[k] || sd.r[k]!=ref[ix[j]].r[k]) fail("descriptor ranks");
                const U64 expect=sum_desc(ref[ix[j]],source);
                if (sum_desc(dd,source)!=expect || sum_desc(sd,source)!=expect) fail("quad sum");
            }
        }
    }
}
}

int main() {
    prove_scheduler_tail();
    prove_dense_sparse_quad();
    std::cout << "gridfp-directgather64-quad-proof OK"
              << " scheduler_exact=1"
              << " tail_1_2_3_exact=1"
              << " dense64_decode_exact=1"
              << " sparse64_decode_exact=1"
              << " quad4_sum_exact=1\n";
    return 0;
}
