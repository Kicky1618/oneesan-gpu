#include <cuda_runtime.h>
#include <cuda/atomic>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>
#include <utility>

using Count = unsigned long long;
static constexpr int FULL_NRES = 26;
#ifndef RES_BATCH
#define RES_BATCH 4
#endif
static constexpr int NRES = RES_BATCH;
__device__ __constant__ Count D_MODS[FULL_NRES] = {2305843009213693951ULL,2305843009213693921ULL,2305843009213693907ULL,2305843009213693723ULL,2305843009213693693ULL,2305843009213693669ULL,2305843009213693613ULL,2305843009213693561ULL,2305843009213693549ULL,2305843009213693487ULL,2305843009213693421ULL,2305843009213693373ULL,2305843009213693277ULL,2305843009213693193ULL,2305843009213693153ULL,2305843009213693133ULL,2305843009213693123ULL,2305843009213693109ULL,2305843009213693093ULL,2305843009213693013ULL,2305843009213692967ULL,2305843009213692937ULL,2305843009213692799ULL,2305843009213692757ULL,2305843009213692737ULL,2305843009213692671ULL};
static constexpr Count H_MODS[FULL_NRES] = {2305843009213693951ULL,2305843009213693921ULL,2305843009213693907ULL,2305843009213693723ULL,2305843009213693693ULL,2305843009213693669ULL,2305843009213693613ULL,2305843009213693561ULL,2305843009213693549ULL,2305843009213693487ULL,2305843009213693421ULL,2305843009213693373ULL,2305843009213693277ULL,2305843009213693193ULL,2305843009213693153ULL,2305843009213693133ULL,2305843009213693123ULL,2305843009213693109ULL,2305843009213693093ULL,2305843009213693013ULL,2305843009213692967ULL,2305843009213692937ULL,2305843009213692799ULL,2305843009213692757ULL,2305843009213692737ULL,2305843009213692671ULL};
static constexpr int MAXW = 28;
static constexpr int MAXWORK = MAXW + 2;

struct Key128 {
    uint64_t lo, hi;
    __host__ __device__ bool operator==(const Key128& o) const { return lo == o.lo && hi == o.hi; }
};

struct Meta {
    int before_n, work_n, after_n;
    int iu, iv;
    uint8_t endpoint_max_u, endpoint_max_v;
    uint32_t forget_mask;
    uint32_t terminal_mask;
    int8_t after_src[MAXW];
};

__host__ __device__ static inline uint64_t get_bits(const Key128& k, int pos, int bits) {
    const uint64_t mask = (1ULL << bits) - 1;
    if (pos < 64) {
        const int sh = pos;
        uint64_t x = k.lo >> sh;
        if (sh + bits > 64) x |= k.hi << (64 - sh);
        return x & mask;
    }
    return (k.hi >> (pos - 64)) & mask;
}

__host__ __device__ static inline void put_bits(Key128& k, int pos, int bits, uint64_t x) {
    const uint64_t mask = (1ULL << bits) - 1;
    x &= mask;
    if (pos < 64) {
        const int sh = pos;
        k.lo |= x << sh;
        if (sh + bits > 64) k.hi |= x >> (64 - sh);
    } else {
        k.hi |= x << (pos - 64);
    }
}

struct PackedWork {
    uint64_t deg = 0;
    uint64_t comp0 = 0;
    uint64_t comp1 = 0;
    uint64_t comp2 = 0;
    uint64_t flags = 0;
    bool done = false;
};

__device__ __forceinline__ uint8_t ws_deg(const PackedWork& s, int i) {
    return static_cast<uint8_t>((s.deg >> (2 * i)) & 3ULL);
}
__device__ __forceinline__ void ws_set_deg(PackedWork& s, int i, uint8_t v) {
    const uint64_t sh = 2ULL * i;
    s.deg = (s.deg & ~(3ULL << sh)) | (uint64_t(v) << sh);
}
__device__ __forceinline__ uint8_t packed5_get(uint64_t a, uint64_t b, uint64_t c, int i) {
    int bit = 5 * i, wi = bit >> 6, sh = bit & 63;
    uint64_t x;
    if (wi == 0) {
        x = a >> sh;
        if (sh > 59) x |= b << (64 - sh);
    } else if (wi == 1) {
        x = b >> sh;
        if (sh > 59) x |= c << (64 - sh);
    } else {
        x = c >> sh;
    }
    return static_cast<uint8_t>(x & 31ULL);
}
__device__ __forceinline__ void packed5_set(uint64_t& a, uint64_t& b, uint64_t& c, int i, uint8_t v) {
    int bit = 5 * i, wi = bit >> 6, sh = bit & 63;
    uint64_t x = uint64_t(v & 31u);
    uint64_t lowmask = 31ULL << sh;
    if (wi == 0) {
        a = (a & ~lowmask) | (x << sh);
        if (sh > 59) { int hb=sh+5-64; uint64_t hm=(1ULL<<hb)-1; b=(b&~hm)|(x>>(64-sh)); }
    } else if (wi == 1) {
        b = (b & ~lowmask) | (x << sh);
        if (sh > 59) { int hb=sh+5-64; uint64_t hm=(1ULL<<hb)-1; c=(c&~hm)|(x>>(64-sh)); }
    } else {
        c = (c & ~lowmask) | (x << sh);
    }
}
__device__ __forceinline__ uint8_t ws_comp(const PackedWork& s, int i) {
    return packed5_get(s.comp0, s.comp1, s.comp2, i);
}
__device__ __forceinline__ void ws_set_comp(PackedWork& s, int i, uint8_t v) {
    packed5_set(s.comp0, s.comp1, s.comp2, i, v);
}
__device__ __forceinline__ uint8_t ws_flag(const PackedWork& s, int c) {
    return static_cast<uint8_t>((s.flags >> (2 * c)) & 3ULL);
}
__device__ __forceinline__ void ws_set_flag(PackedWork& s, int c, uint8_t v) {
    const uint32_t sh = 2u * c;
    s.flags = (s.flags & ~(3ULL << sh)) | (uint64_t(v) << sh);
}
__device__ __forceinline__ void ws_or_flag(PackedWork& s, int c, uint8_t v) {
    s.flags |= uint64_t(v) << (2 * c);
}

// Non-crossing partition encoding for a planar frontier.
// deg==0 means no component. Active vertices use 2-bit topology symbols:
// SINGLE=0, OPEN=1, MIDDLE=2, CLOSE=3.
__device__ __forceinline__ PackedWork decode_packed(const Key128& k, int n) {
    PackedWork s;
    s.done = get_bits(k, 0, 1);
    int p = 1;
    #pragma unroll
    for (int i = 0; i < MAXW; ++i) {
        if (i < n) { ws_set_deg(s, i, static_cast<uint8_t>(get_bits(k, p, 2))); p += 2; }
    }
    const int conn_base = p;
    p += 2 * n;
    const uint8_t start_pos = static_cast<uint8_t>(get_bits(k, p, 5)); p += 5;
    const uint8_t target_pos = static_cast<uint8_t>(get_bits(k, p, 5));

    uint64_t st0 = 0, st1 = 0, st2 = 0;
    int sp = 0;
    uint8_t next = 1;
    #pragma unroll
    for (int i = 0; i < MAXW; ++i) {
        if (i >= n || ws_deg(s, i) == 0) continue;
        const uint8_t t = static_cast<uint8_t>(get_bits(k, conn_base + 2 * i, 2));
        uint8_t c;
        if (t == 0) {
            c = next++;
        } else if (t == 1) {
            c = next++;
            packed5_set(st0, st1, st2, sp++, c);
        } else if (t == 2) {
            c = packed5_get(st0, st1, st2, sp - 1);
        } else {
            c = packed5_get(st0, st1, st2, --sp);
        }
        ws_set_comp(s, i, c);
    }
    if (start_pos) ws_or_flag(s, ws_comp(s, start_pos - 1), 1);
    if (target_pos) ws_or_flag(s, ws_comp(s, target_pos - 1), 2);
    return s;
}

__device__ __forceinline__ Key128 encode_packed_canonical(int n, PackedWork s) {
    // 5-bit packed first/last frontier position per component. 31 is unused.
    uint64_t f0 = ~0ULL, f1 = ~0ULL, f2 = ~0ULL;
    uint64_t l0 = ~0ULL, l1 = ~0ULL, l2 = ~0ULL;
    #pragma unroll
    for (int i = 0; i < MAXW; ++i) {
        if (i >= n) continue;
        const uint8_t c = ws_comp(s, i);
        if (!c) continue;
        if (packed5_get(f0, f1, f2, c) == 31)
            packed5_set(f0, f1, f2, c, static_cast<uint8_t>(i));
        packed5_set(l0, l1, l2, c, static_cast<uint8_t>(i));
    }

    uint8_t start_pos = 0, target_pos = 0;
    #pragma unroll
    for (int c = 1; c <= MAXW; ++c) {
        const uint8_t fi = packed5_get(f0, f1, f2, c);
        if (fi == 31) continue;
        const uint8_t fl = ws_flag(s, c);
        if (fl & 1) start_pos = fi + 1;
        if (fl & 2) target_pos = fi + 1;
    }

    Key128 k{0, 0};
    put_bits(k, 0, 1, s.done ? 1 : 0);
    int p = 1;
    #pragma unroll
    for (int i = 0; i < MAXW; ++i) {
        if (i < n) { put_bits(k, p, 2, ws_deg(s, i)); p += 2; }
    }
    #pragma unroll
    for (int i = 0; i < MAXW; ++i) {
        if (i >= n) continue;
        const uint8_t c = ws_comp(s, i);
        uint8_t t = 0;
        if (c) {
            const uint8_t fi = packed5_get(f0, f1, f2, c);
            const uint8_t li = packed5_get(l0, l1, l2, c);
            if (fi != li) {
                if (i == fi) t = 1;
                else if (i == li) t = 3;
                else t = 2;
            }
        }
        put_bits(k, p, 2, t); p += 2;
    }
    put_bits(k, p, 5, start_pos); p += 5;
    put_bits(k, p, 5, target_pos);
    return k;
}

__device__ __forceinline__ bool transition_decoded(
    PackedWork s, const Meta& m, int take, Key128& out) {

    if (take) {
        if (s.done) return false;
        uint8_t du = ws_deg(s, m.iu), dv = ws_deg(s, m.iv);
        if (du >= m.endpoint_max_u || dv >= m.endpoint_max_v) return false;
        ws_set_deg(s, m.iu, du + 1);
        ws_set_deg(s, m.iv, dv + 1);
        uint8_t a = ws_comp(s, m.iu), b = ws_comp(s, m.iv);
        if (!a && !b) {
            uint32_t used = 0;
            #pragma unroll
            for (int i = 0; i < MAXWORK; ++i)
                if (i < m.work_n) used |= uint32_t(1u << ws_comp(s, i));
            uint8_t q = 1;
            while (q <= MAXW && (used & (1u << q))) ++q;
            if (q > MAXW) return false;
            ws_set_flag(s, q, 0);
            ws_set_comp(s, m.iu, q); ws_set_comp(s, m.iv, q);
            if (m.terminal_mask & (1u << m.iu)) ws_or_flag(s, q, 1);
            if (m.terminal_mask & (1u << m.iv)) ws_or_flag(s, q, 2);
        } else if (!a || !b) {
            uint8_t q = a ? a : b;
            ws_set_comp(s, m.iu, q); ws_set_comp(s, m.iv, q);
            if (m.terminal_mask & (1u << m.iu)) ws_or_flag(s, q, 1);
            if (m.terminal_mask & (1u << m.iv)) ws_or_flag(s, q, 2);
        } else {
            if (a == b) return false;
            uint8_t keep = a < b ? a : b, kill = a < b ? b : a;
            ws_or_flag(s, keep, ws_flag(s, kill));
            #pragma unroll
            for (int i = 0; i < MAXWORK; ++i)
                if (i < m.work_n && ws_comp(s, i) == kill) ws_set_comp(s, i, keep);
            ws_set_flag(s, kill, 0);
        }
    }

    #pragma unroll
    for (int i = 0; i < MAXWORK; ++i) {
        if (i >= m.work_n || !(m.forget_mask & (1u << i))) continue;
        bool terminal = m.terminal_mask & (1u << i);
        uint8_t d = ws_deg(s, i);
        if ((terminal && d != 1) || (!terminal && d != 0 && d != 2)) return false;
        uint8_t q = ws_comp(s, i);
        ws_set_comp(s, i, 0);
        if (q) {
            bool alive = false;
            #pragma unroll
            for (int j = 0; j < MAXWORK; ++j)
                if (j < m.work_n && ws_comp(s, j) == q) alive = true;
            if (!alive) {
                if (ws_flag(s, q) == 3) s.done = true;
                else return false;
            }
        }
    }

    PackedWork o;
    o.flags = s.flags;
    o.done = s.done;
    #pragma unroll
    for (int i = 0; i < MAXW; ++i) {
        if (i < m.after_n) {
            int src = m.after_src[i];
            ws_set_deg(o, i, ws_deg(s, src));
            ws_set_comp(o, i, ws_comp(s, src));
        }
    }
    out = encode_packed_canonical(m.after_n, o);
    return true;
}

__device__ static inline uint64_t mix64(uint64_t x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

__device__ static inline uint64_t hash_key(Key128 k) {
    return mix64(k.lo ^ mix64(k.hi + 0x9e3779b97f4a7c15ULL));
}

__device__ static inline void atomic_add_mod(Count* p, Count v, Count mod) {
    Count old = atomicAdd(p, 0ULL);
    for (;;) {
        Count neu = old >= mod - v ? old - (mod - v) : old + v;
        Count seen = atomicCAS(p, old, neu);
        if (seen == old) return;
        old = seen;
    }
}

// ctrl encodes generation and state: (generation << 1) | ready_bit.
// For the current generation: LOCKED=(gen<<1), READY=(gen<<1)|1.
// Any other generation is logically empty, so the table never needs a per-layer memset.
__device__ static inline void hash_insert(
    Key128 key, const Count* values,
    uint32_t* table_index,
    Key128* dense_keys, Count* dense_vals, uint32_t* dense_n,
    uint32_t dense_cap, uint32_t* overflow,
    uint32_t mask, int mod_base, int active_res) {
    constexpr uint32_t EMPTY = 0xffffffffu;
    constexpr uint32_t LOCKED = 0xfffffffeu;
    uint32_t slot = static_cast<uint32_t>(hash_key(key)) & mask;
    for (uint32_t probe = 0; probe <= mask; ++probe, slot = (slot + 1) & mask) {
        uint32_t idx = atomicAdd(&table_index[slot], 0u);
        if (idx == EMPTY) {
            uint32_t seen = atomicCAS(&table_index[slot], EMPTY, LOCKED);
            if (seen == EMPTY) {
                uint32_t oi = atomicAdd(dense_n, 1u);
                if (oi >= dense_cap) {
                    atomicExch(overflow, 1u);
                    __threadfence();
                    atomicExch(&table_index[slot], EMPTY);
                    return;
                }
                dense_keys[oi] = key;
                #pragma unroll
                for (int r = 0; r < NRES; ++r)
                    dense_vals[size_t(oi) * NRES + r] = (r < active_res ? values[r] : 0);
                __threadfence();
                atomicExch(&table_index[slot], oi);
                return;
            }
            idx = seen;
        }
        while (idx == LOCKED) idx = atomicAdd(&table_index[slot], 0u);
        if (idx != EMPTY && dense_keys[idx] == key) {
            #pragma unroll
            for (int r = 0; r < NRES; ++r)
                if (r < active_res)
                    atomic_add_mod(&dense_vals[size_t(idx) * NRES + r], values[r], D_MODS[mod_base + r]);
            return;
        }
    }
    atomicExch(overflow, 1u);
}

__global__ void expand_hash_kernel(
    const Key128* keys, const Count* vals, size_t n, Meta m,
    uint32_t* table_index,
    Key128* dense_keys, Count* dense_vals, uint32_t* dense_n,
    uint32_t dense_cap, uint32_t* overflow,
    uint32_t mask, int mod_base, int active_res) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    Count v[NRES];
    #pragma unroll
    for (int r = 0; r < NRES; ++r) v[r] = vals[i * NRES + r];
    Key128 out;
    PackedWork base = decode_packed(keys[i], m.before_n);
    if (transition_decoded(base, m, 0, out))
        hash_insert(out, v, table_index, dense_keys, dense_vals, dense_n, dense_cap, overflow, mask, mod_base, active_res);
    if (transition_decoded(base, m, 1, out))
        hash_insert(out, v, table_index, dense_keys, dense_vals, dense_n, dense_cap, overflow, mask, mod_base, active_res);
}

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::cerr << what << ": " << cudaGetErrorString(e) << "\n"; std::exit(1); }
}

static size_t next_pow2(size_t x) {
    size_t p = 1;
    while (p < x) p <<= 1;
    return p;
}

int main(int argc, char** argv) {
    int n = argc > 1 ? std::stoi(argv[1]) : 8;
    int mod_base = argc > 2 ? std::stoi(argv[2]) : 0;
    int active_res = argc > 3 ? std::stoi(argv[3]) : NRES;
    if (mod_base < 0 || active_res < 1 || active_res > NRES || mod_base + active_res > FULL_NRES) {
        std::cerr << "bad residue batch\n"; return 1;
    }
    int w = n + 1;
    if (w > MAXW) { std::cerr << "n too large for 128-bit planar state; max n=" << (MAXW - 1) << "\n"; return 1; }
    int V = w * w, start = 0, target = V - 1;

    std::vector<std::pair<int,int>> edges;
    for (int r = 0; r < w; ++r) for (int c = 0; c < w; ++c) {
        int u = r*w+c;
        if (c+1 < w) edges.push_back({u,u+1});
        if (r+1 < w) edges.push_back({u,u+w});
    }
    std::vector<int> first(V, 1e9), last(V,-1);
    for (int i=0;i<(int)edges.size();++i) {
        auto [u,v]=edges[i];
        first[u]=std::min(first[u],i); first[v]=std::min(first[v],i);
        last[u]=std::max(last[u],i); last[v]=std::max(last[v],i);
    }
    std::vector<std::vector<int>> active(edges.size()+1);
    for (int e=0;e<=(int)edges.size();++e)
        for (int v=0;v<V;++v) if (first[v] < e && last[v] >= e) active[e].push_back(v);

    Key128* cur_keys = nullptr;
    Count* cur_vals = nullptr;
    Key128* spare_keys = nullptr;
    Count* spare_vals = nullptr;
    size_t in_n = 1, cur_cap = 1, spare_cap = 0;
    ck(cudaMalloc(&cur_keys, sizeof(Key128)), "malloc initial keys");
    ck(cudaMalloc(&cur_vals, NRES * sizeof(Count)), "malloc initial vals");
    ck(cudaMemset(cur_keys, 0, sizeof(Key128)), "init key");
    Count init_vals[NRES]; for (int r=0;r<NRES;++r) init_vals[r]=(r < active_res ? 1 : 0);
    ck(cudaMemcpy(cur_vals, init_vals, sizeof(init_vals), cudaMemcpyHostToDevice), "init vals");
    size_t peak = 1, peak_table = 0;
    size_t peak_alloc_bytes = sizeof(Key128) + NRES * sizeof(Count) + sizeof(uint32_t);

    uint32_t* table_index = nullptr;
    uint32_t* dense_n = nullptr;
    uint32_t* overflow = nullptr;
    size_t allocated_cap = 0;
    ck(cudaMalloc(&dense_n, sizeof(uint32_t)), "malloc dense_n");
    ck(cudaMalloc(&overflow, sizeof(uint32_t)), "malloc overflow");

    cudaEvent_t ev0,ev1; cudaEventCreate(&ev0); cudaEventCreate(&ev1);
    cudaEventRecord(ev0);

    for (int ei=0; ei<(int)edges.size(); ++ei) {
        auto before = active[ei], after = active[ei+1];
        auto [u,v] = edges[ei];
        std::vector<int> work = before;
        auto ensure=[&](int x){ if(std::find(work.begin(),work.end(),x)==work.end()) work.push_back(x); };
        ensure(u); ensure(v);
        if ((int)work.size() > MAXWORK) { std::cerr << "work frontier overflow\n"; return 1; }

        Meta m{};
        m.before_n=before.size(); m.work_n=work.size(); m.after_n=after.size();
        auto pos=[&](int x){ return int(std::find(work.begin(),work.end(),x)-work.begin()); };
        m.iu=pos(u); m.iv=pos(v);
        m.endpoint_max_u=(u==start||u==target)?1:2;
        m.endpoint_max_v=(v==start||v==target)?1:2;
        for(int i=0;i<m.work_n;++i) {
            int x=work[i];
            if(std::find(after.begin(),after.end(),x)==after.end()) m.forget_mask |= (1u<<i);
            if(x==start||x==target) m.terminal_mask |= (1u<<i);
        }
        for(int i=0;i<m.after_n;++i) m.after_src[i]=pos(after[i]);

        size_t cap = next_pow2(std::max<size_t>(1u << 16, in_n * 4));
        if (cap > allocated_cap) {
            if (table_index) { cudaFree(table_index); }
            ck(cudaMalloc(&table_index, cap*sizeof(uint32_t)), "malloc table_index");
            allocated_cap = cap;
        }
        peak_table = std::max(peak_table, cap);
        ck(cudaMemset(dense_n, 0, sizeof(uint32_t)), "clear dense_n");

        // Large layers are very predictable: unique next states are about 1.41x
        // the smaller steady-state layer. Start at 1.45x instead of reserving
        // 2x+power-of-two, and retry the layer at the mathematical 2x bound
        // only if the dense output actually overflows. This is the H200 memory mode.
        const size_t hard_max_out = 2 * in_n;
        size_t wanted_out = in_n < (1u << 20)
            ? hard_max_out
            : std::min(hard_max_out, (in_n * 145 + 99) / 100 + 1024);

        auto set_spare_capacity = [&](size_t need) {
            // spare does not contain the current layer, so it is safe to resize.
            // Also shrink grossly oversized buffers left by an earlier 2x retry.
            const bool too_small = spare_cap < need;
            const bool too_large = need >= (1u << 20) && spare_cap > need + need / 5;
            if (!too_small && !too_large) return;
            if (spare_keys) { cudaFree(spare_keys); cudaFree(spare_vals); }
            spare_cap = need;
            ck(cudaMalloc(&spare_keys, spare_cap * sizeof(Key128)), "malloc spare keys");
            ck(cudaMalloc(&spare_vals, spare_cap * NRES * sizeof(Count)), "malloc spare vals");
        };
        set_spare_capacity(wanted_out);

        int threads=256; int blocks=(in_n+threads-1)/threads;
        uint32_t rn = 0;
        for (int attempt = 0; attempt < 2; ++attempt) {
            ck(cudaMemset(table_index, 0xff, cap*sizeof(uint32_t)), "clear table_index");
            ck(cudaMemset(dense_n, 0, sizeof(uint32_t)), "clear dense_n");
            ck(cudaMemset(overflow, 0, sizeof(uint32_t)), "clear overflow");
            expand_hash_kernel<<<blocks,threads>>>(
                cur_keys, cur_vals, in_n, m,
                table_index, spare_keys, spare_vals, dense_n,
                static_cast<uint32_t>(spare_cap), overflow,
                static_cast<uint32_t>(cap-1), mod_base, active_res);
            ck(cudaGetLastError(), "expand hash kernel");

            uint32_t ov = 0;
            ck(cudaMemcpy(&rn, dense_n, sizeof(rn), cudaMemcpyDeviceToHost), "copy dense_n");
            ck(cudaMemcpy(&ov, overflow, sizeof(ov), cudaMemcpyDeviceToHost), "copy overflow");
            if (!ov && rn <= spare_cap && rn < cap) break;
            if (attempt != 0) { std::cerr << "hash/output overflow after retry\n"; return 2; }
            // dense_n still gives a useful lower bound after overflow. Grow only
            // as much as needed instead of permanently reserving the 2x bound.
            size_t retry_need = std::max(spare_cap + spare_cap / 4 + 4096, size_t(rn) + size_t(rn) / 16 + 4096);
            retry_need = std::min(hard_max_out, retry_need);
            if (retry_need <= spare_cap) retry_need = hard_max_out;
            set_spare_capacity(retry_need);
        }

        {
            const size_t table_bytes = allocated_cap * sizeof(uint32_t);
            const size_t state_bytes = (cur_cap + spare_cap) * (sizeof(Key128) + NRES * sizeof(Count));
            peak_alloc_bytes = std::max(peak_alloc_bytes, table_bytes + state_bytes + 2*sizeof(uint32_t));
        }
        std::swap(cur_keys, spare_keys); std::swap(cur_vals, spare_vals);
        std::swap(cur_cap, spare_cap);
        in_n = rn;
        peak=std::max<size_t>(peak,rn);
        if ((ei+1)%w==0 || ei+1==(int)edges.size())
            std::cerr << "edge " << ei+1 << "/" << edges.size() << " frontier=" << after.size() << " states=" << rn << "\n";
    }
    cudaEventRecord(ev1); cudaEventSynchronize(ev1);
    float ms=0; cudaEventElapsedTime(&ms,ev0,ev1);

    std::vector<Key128> hk(in_n);
    std::vector<Count> hv(in_n * NRES);
    ck(cudaMemcpy(hk.data(), cur_keys, in_n*sizeof(Key128), cudaMemcpyDeviceToHost), "copy final keys");
    ck(cudaMemcpy(hv.data(), cur_vals, in_n*NRES*sizeof(Count), cudaMemcpyDeviceToHost), "copy final vals");
    Count ans[NRES] = {};
    for(size_t i=0;i<hk.size();++i) if(get_bits(hk[i],0,1)) {
        for (int r=0;r<NRES;++r) {
            Count v=hv[i*NRES+r], mod=H_MODS[mod_base + r];
            ans[r] = ans[r] >= mod - v ? ans[r] - (mod - v) : ans[r] + v;
        }
    }
    std::cout << "backend=hopper-nc128 n=" << n << " residues=";
    for(int r=0;r<active_res;++r){ if(r) std::cout << ','; std::cout << ans[r]; }
    std::cout << " peak_states=" << peak << " hash_slots=" << peak_table
              << " peak_alloc_bytes=" << peak_alloc_bytes << " gpu_ms=" << ms << "\n";

    cudaFree(table_index); cudaFree(dense_n); cudaFree(overflow);
    cudaFree(cur_keys); cudaFree(cur_vals); cudaFree(spare_keys); cudaFree(spare_vals);
}
