#include <cuda_runtime.h>
#include <cuda/atomic>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>
#include <utility>

using Count = unsigned long long;
static constexpr int FULL_NRES = 13;
static constexpr int NRES = 4;
__device__ __constant__ Count D_MODS[FULL_NRES] = {2305843009213693951ULL,2305843009213693921ULL,2305843009213693907ULL,2305843009213693723ULL,2305843009213693693ULL,2305843009213693669ULL,2305843009213693613ULL,2305843009213693561ULL,2305843009213693549ULL,2305843009213693487ULL,2305843009213693421ULL,2305843009213693373ULL,2305843009213693277ULL};
static constexpr Count H_MODS[FULL_NRES] = {2305843009213693951ULL,2305843009213693921ULL,2305843009213693907ULL,2305843009213693723ULL,2305843009213693693ULL,2305843009213693669ULL,2305843009213693613ULL,2305843009213693561ULL,2305843009213693549ULL,2305843009213693487ULL,2305843009213693421ULL,2305843009213693373ULL,2305843009213693277ULL};
static constexpr int MAXW = 24;
static constexpr int MAXWORK = MAXW + 2;

struct Key192 {
    uint64_t w0, w1, w2;
    __host__ __device__ bool operator==(const Key192& o) const { return w0 == o.w0 && w1 == o.w1 && w2 == o.w2; }
};

struct Meta {
    int before_n, work_n, after_n;
    int iu, iv;
    uint8_t endpoint_max_u, endpoint_max_v;
    uint32_t forget_mask;
    uint32_t terminal_mask;
    int8_t after_src[MAXW];
};

__host__ __device__ static inline uint64_t get_bits(const Key192& k, int pos, int bits) {
    const uint64_t* w = &k.w0;
    int wi = pos >> 6, sh = pos & 63;
    uint64_t mask = (1ULL << bits) - 1;
    uint64_t x = w[wi] >> sh;
    if (sh + bits > 64) x |= w[wi + 1] << (64 - sh);
    return x & mask;
}

__host__ __device__ static inline void put_bits(Key192& k, int pos, int bits, uint64_t x) {
    uint64_t* w = &k.w0;
    int wi = pos >> 6, sh = pos & 63;
    uint64_t mask = (1ULL << bits) - 1;
    x &= mask;
    w[wi] |= x << sh;
    if (sh + bits > 64) w[wi + 1] |= x >> (64 - sh);
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

__device__ __forceinline__ PackedWork decode_packed(const Key192& k, int n) {
    PackedWork s;
    s.done = get_bits(k, 0, 1);
    int p = 1;
    #pragma unroll
    for (int i = 0; i < MAXW; ++i) {
        if (i < n) {
            ws_set_deg(s, i, static_cast<uint8_t>(get_bits(k, p, 2))); p += 2;
            ws_set_comp(s, i, static_cast<uint8_t>(get_bits(k, p, 5))); p += 5;
        }
    }
    uint8_t start_comp = static_cast<uint8_t>(get_bits(k, p, 5)); p += 5;
    uint8_t target_comp = static_cast<uint8_t>(get_bits(k, p, 5));
    if (start_comp) ws_or_flag(s, start_comp, 1);
    if (target_comp) ws_or_flag(s, target_comp, 2);
    return s;
}

__device__ __forceinline__ Key192 encode_packed_canonical(int n, PackedWork s) {
    uint64_t map0 = 0, map1 = 0, map2 = 0;
    uint8_t next = 1, start_comp = 0, target_comp = 0;
    #pragma unroll
    for (int i = 0; i < MAXW; ++i) {
        if (i >= n) continue;
        uint8_t c = ws_comp(s, i);
        if (!c) continue;
        uint8_t r = packed5_get(map0, map1, map2, c);
        if (!r) {
            r = next++;
            packed5_set(map0, map1, map2, c, r);
            uint8_t f = ws_flag(s, c);
            if (f & 1) start_comp = r;
            if (f & 2) target_comp = r;
        }
        ws_set_comp(s, i, r);
    }

    Key192 k{0, 0, 0};
    put_bits(k, 0, 1, s.done ? 1 : 0);
    int p = 1;
    #pragma unroll
    for (int i = 0; i < MAXW; ++i) {
        if (i < n) {
            put_bits(k, p, 2, ws_deg(s, i)); p += 2;
            put_bits(k, p, 5, ws_comp(s, i)); p += 5;
        }
    }
    put_bits(k, p, 5, start_comp); p += 5;
    put_bits(k, p, 5, target_comp);
    return k;
}

__device__ __forceinline__ bool transition_decoded(
    PackedWork s, const Meta& m, int take, Key192& out) {

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

__device__ static inline uint64_t hash_key(Key192 k) {
    return mix64(k.w0 ^ mix64(k.w1 + 0x9e3779b97f4a7c15ULL) ^ mix64(k.w2 + 0xd1b54a32d192ed03ULL));
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
    Key192 key, const Count* values,
    Key192* table_keys, uint32_t* table_index, unsigned int* table_state,
    Key192* dense_keys, Count* dense_vals, uint32_t* dense_n,
    uint32_t dense_cap, uint32_t* overflow,
    uint32_t mask, uint32_t generation, int mod_base, int active_res) {
    const unsigned int locked = generation << 1;
    const unsigned int ready = locked | 1u;
    uint32_t slot = static_cast<uint32_t>(hash_key(key)) & mask;
    for (uint32_t probe = 0; probe <= mask; ++probe, slot = (slot + 1) & mask) {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> st(table_state[slot]);
        unsigned int s = st.load(cuda::memory_order_acquire);

        for (;;) {
            if (s == ready) {
                if (table_keys[slot] == key) {
                    uint32_t idx = table_index[slot];
                    if (idx == 0xffffffffu) { atomicExch(overflow, 1u); return; }
                    #pragma unroll
                    for (int r = 0; r < NRES; ++r)
                        if (r < active_res) atomic_add_mod(&dense_vals[size_t(idx) * NRES + r], values[r], D_MODS[mod_base + r]);
                    return;
                }
                break; // occupied by another key in this generation
            }
            if (s == locked) {
                do { s = st.load(cuda::memory_order_acquire); } while (s == locked);
                continue;
            }

            // Stale generation: try to claim the slot.
            unsigned int expected = s;
            if (st.compare_exchange_strong(expected, locked, cuda::memory_order_acq_rel)) {
                uint32_t oi = atomicAdd(dense_n, 1u);
                table_keys[slot] = key;
                if (oi >= dense_cap) {
                    table_index[slot] = 0xffffffffu;
                    atomicExch(overflow, 1u);
                    st.store(ready, cuda::memory_order_release);
                    return;
                }
                table_index[slot] = oi;
                dense_keys[oi] = key;
                #pragma unroll
                for (int r = 0; r < NRES; ++r) dense_vals[size_t(oi) * NRES + r] = (r < active_res ? values[r] : 0);
                st.store(ready, cuda::memory_order_release);
                return;
            }
            s = expected;
        }
    }
}

__global__ void expand_hash_kernel(
    const Key192* keys, const Count* vals, size_t n, Meta m,
    Key192* table_keys, uint32_t* table_index, unsigned int* table_state,
    Key192* dense_keys, Count* dense_vals, uint32_t* dense_n,
    uint32_t dense_cap, uint32_t* overflow,
    uint32_t mask, uint32_t generation, int mod_base, int active_res) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    Count v[NRES];
    #pragma unroll
    for (int r = 0; r < NRES; ++r) v[r] = vals[i * NRES + r];
    Key192 out;
    PackedWork base = decode_packed(keys[i], m.before_n);
    if (transition_decoded(base, m, 0, out))
        hash_insert(out, v, table_keys, table_index, table_state, dense_keys, dense_vals, dense_n, dense_cap, overflow, mask, generation, mod_base, active_res);
    if (transition_decoded(base, m, 1, out))
        hash_insert(out, v, table_keys, table_index, table_state, dense_keys, dense_vals, dense_n, dense_cap, overflow, mask, generation, mod_base, active_res);
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
    if (w > MAXW) { std::cerr << "n too large for 192-bit state; max n=" << (MAXW - 1) << "\n"; return 1; }
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

    Key192* cur_keys = nullptr;
    Count* cur_vals = nullptr;
    Key192* spare_keys = nullptr;
    Count* spare_vals = nullptr;
    size_t in_n = 1, cur_cap = 1, spare_cap = 0;
    ck(cudaMalloc(&cur_keys, sizeof(Key192)), "malloc initial keys");
    ck(cudaMalloc(&cur_vals, NRES * sizeof(Count)), "malloc initial vals");
    ck(cudaMemset(cur_keys, 0, sizeof(Key192)), "init key");
    Count init_vals[NRES]; for (int r=0;r<NRES;++r) init_vals[r]=(r < active_res ? 1 : 0);
    ck(cudaMemcpy(cur_vals, init_vals, sizeof(init_vals), cudaMemcpyHostToDevice), "init vals");
    size_t peak = 1, peak_table = 0;
    size_t peak_alloc_bytes = sizeof(Key192) + NRES * sizeof(Count) + sizeof(uint32_t);

    Key192* table_keys = nullptr;
    uint32_t* table_index = nullptr;
    unsigned int* table_state = nullptr;
    uint32_t* dense_n = nullptr;
    uint32_t* overflow = nullptr;
    size_t allocated_cap = 0;
    ck(cudaMalloc(&dense_n, sizeof(uint32_t)), "malloc dense_n");
    ck(cudaMalloc(&overflow, sizeof(uint32_t)), "malloc overflow");
    uint32_t generation_counter = 1;

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
            if (table_keys) { cudaFree(table_keys); cudaFree(table_index); cudaFree(table_state); }
            ck(cudaMalloc(&table_keys, cap*sizeof(Key192)), "malloc table_keys");
            ck(cudaMalloc(&table_index, cap*sizeof(uint32_t)), "malloc table_index");
            ck(cudaMalloc(&table_state, cap*sizeof(unsigned int)), "malloc table_state");
            ck(cudaMemset(table_state, 0, cap*sizeof(unsigned int)), "init state");
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

        auto ensure_spare = [&](size_t need) {
            if (spare_cap >= need) return;
            if (spare_keys) { cudaFree(spare_keys); cudaFree(spare_vals); }
            spare_cap = need; // exact capacity: do not round to a power of two
            ck(cudaMalloc(&spare_keys, spare_cap * sizeof(Key192)), "malloc spare keys");
            ck(cudaMalloc(&spare_vals, spare_cap * NRES * sizeof(Count)), "malloc spare vals");
        };
        ensure_spare(wanted_out);

        int threads=256; int blocks=(in_n+threads-1)/threads;
        uint32_t rn = 0;
        for (int attempt = 0; attempt < 2; ++attempt) {
            ck(cudaMemset(dense_n, 0, sizeof(uint32_t)), "clear dense_n");
            ck(cudaMemset(overflow, 0, sizeof(uint32_t)), "clear overflow");
            const uint32_t gen = generation_counter++;
            expand_hash_kernel<<<blocks,threads>>>(
                cur_keys, cur_vals, in_n, m,
                table_keys, table_index, table_state, spare_keys, spare_vals, dense_n,
                static_cast<uint32_t>(spare_cap), overflow,
                static_cast<uint32_t>(cap-1), gen, mod_base, active_res);
            ck(cudaGetLastError(), "expand hash kernel");

            uint32_t ov = 0;
            ck(cudaMemcpy(&rn, dense_n, sizeof(rn), cudaMemcpyDeviceToHost), "copy dense_n");
            ck(cudaMemcpy(&ov, overflow, sizeof(ov), cudaMemcpyDeviceToHost), "copy overflow");
            if (!ov && rn <= spare_cap && rn < cap) break;
            if (attempt != 0) { std::cerr << "hash/output overflow after retry\n"; return 2; }
            ensure_spare(hard_max_out);
        }

        {
            const size_t table_bytes = allocated_cap * (sizeof(Key192) + sizeof(uint32_t) + sizeof(unsigned int));
            const size_t state_bytes = (cur_cap + spare_cap) * (sizeof(Key192) + NRES * sizeof(Count));
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

    std::vector<Key192> hk(in_n);
    std::vector<Count> hv(in_n * NRES);
    ck(cudaMemcpy(hk.data(), cur_keys, in_n*sizeof(Key192), cudaMemcpyDeviceToHost), "copy final keys");
    ck(cudaMemcpy(hv.data(), cur_vals, in_n*NRES*sizeof(Count), cudaMemcpyDeviceToHost), "copy final vals");
    Count ans[NRES] = {};
    for(size_t i=0;i<hk.size();++i) if(get_bits(hk[i],0,1)) {
        for (int r=0;r<NRES;++r) {
            Count v=hv[i*NRES+r], mod=H_MODS[mod_base + r];
            ans[r] = ans[r] >= mod - v ? ans[r] - (mod - v) : ans[r] + v;
        }
    }
    std::cout << "backend=hopper-lowmem4-key192 n=" << n << " residues=";
    for(int r=0;r<active_res;++r){ if(r) std::cout << ','; std::cout << ans[r]; }
    std::cout << " peak_states=" << peak << " hash_slots=" << peak_table
              << " peak_alloc_bytes=" << peak_alloc_bytes << " gpu_ms=" << ms << "\n";

    cudaFree(table_keys); cudaFree(table_index); cudaFree(table_state); cudaFree(dense_n); cudaFree(overflow);
    cudaFree(cur_keys); cudaFree(cur_vals); cudaFree(spare_keys); cudaFree(spare_vals);
}
