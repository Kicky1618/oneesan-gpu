#include <cuda_runtime.h>
#include <cuda/atomic>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>
#include <utility>

using Count = unsigned long long;
static constexpr int NRES = 8;
__device__ __constant__ Count D_MODS[NRES] = {2305843009213693921ULL,2305843009213692799ULL,2305843009213691767ULL,2305843009213690657ULL,2305843009213689601ULL,2305843009213688569ULL,2305843009213687519ULL,2305843009213687483ULL};
static constexpr Count H_MODS[NRES] = {2305843009213693921ULL,2305843009213692799ULL,2305843009213691767ULL,2305843009213690657ULL,2305843009213689601ULL,2305843009213688569ULL,2305843009213687519ULL,2305843009213687483ULL};
static constexpr int MAXW = 16;
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
    if (pos < 64) {
        if (pos + bits <= 64) return (k.lo >> pos) & ((1ULL << bits) - 1);
        int a = 64 - pos;
        uint64_t x = k.lo >> pos;
        uint64_t y = k.hi & ((1ULL << (bits - a)) - 1);
        return x | (y << a);
    }
    return (k.hi >> (pos - 64)) & ((1ULL << bits) - 1);
}

__host__ __device__ static inline void put_bits(Key128& k, int pos, int bits, uint64_t x) {
    x &= (1ULL << bits) - 1;
    if (pos < 64) {
        if (pos + bits <= 64) { k.lo |= x << pos; return; }
        int a = 64 - pos;
        k.lo |= x << pos;
        k.hi |= x >> a;
        return;
    }
    k.hi |= x << (pos - 64);
}

struct PackedWork {
    uint64_t deg = 0;
    uint64_t comp_lo = 0;
    uint64_t comp_hi = 0;
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
__device__ __forceinline__ uint8_t packed5_get(uint64_t lo, uint64_t hi, int i) {
    int bit = 5 * i;
    if (bit <= 59) return static_cast<uint8_t>((lo >> bit) & 31ULL);
    if (bit < 64) {
        int a = 64 - bit;
        return static_cast<uint8_t>(((lo >> bit) | (hi << a)) & 31ULL);
    }
    return static_cast<uint8_t>((hi >> (bit - 64)) & 31ULL);
}
__device__ __forceinline__ void packed5_set(uint64_t& lo, uint64_t& hi, int i, uint8_t v) {
    uint64_t x = uint64_t(v & 31u);
    int bit = 5 * i;
    if (bit <= 59) {
        lo = (lo & ~(31ULL << bit)) | (x << bit);
    } else if (bit < 64) {
        int a = 64 - bit;
        uint64_t lomask = ((1ULL << a) - 1) << bit;
        lo = (lo & ~lomask) | ((x & ((1ULL << a) - 1)) << bit);
        uint64_t himask = (1ULL << (5 - a)) - 1;
        hi = (hi & ~himask) | (x >> a);
    } else {
        int sh = bit - 64;
        hi = (hi & ~(31ULL << sh)) | (x << sh);
    }
}
__device__ __forceinline__ uint8_t ws_comp(const PackedWork& s, int i) {
    return packed5_get(s.comp_lo, s.comp_hi, i);
}
__device__ __forceinline__ void ws_set_comp(PackedWork& s, int i, uint8_t v) {
    packed5_set(s.comp_lo, s.comp_hi, i, v);
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

__device__ __forceinline__ PackedWork decode_packed(const Key128& k, int n) {
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

__device__ __forceinline__ Key128 encode_packed_canonical(int n, PackedWork s) {
    uint64_t map_lo = 0, map_hi = 0;
    uint8_t next = 1, start_comp = 0, target_comp = 0;
    #pragma unroll
    for (int i = 0; i < MAXW; ++i) {
        if (i >= n) continue;
        uint8_t c = ws_comp(s, i);
        if (!c) continue;
        uint8_t r = packed5_get(map_lo, map_hi, c);
        if (!r) {
            r = next++;
            packed5_set(map_lo, map_hi, c, r);
            uint8_t f = ws_flag(s, c);
            if (f & 1) start_comp = r;
            if (f & 2) target_comp = r;
        }
        ws_set_comp(s, i, r);
    }

    Key128 k{0, 0};
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
    return mix64(k.lo ^ (mix64(k.hi) + 0x9e3779b97f4a7c15ULL));
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
    Key128* table_keys, uint32_t* table_index, unsigned int* table_state,
    Key128* dense_keys, Count* dense_vals, uint32_t* dense_n,
    uint32_t mask, uint32_t generation) {
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
                    #pragma unroll
                    for (int r = 0; r < NRES; ++r)
                        atomic_add_mod(&dense_vals[size_t(idx) * NRES + r], values[r], D_MODS[r]);
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
                table_index[slot] = oi;
                dense_keys[oi] = key;
                #pragma unroll
                for (int r = 0; r < NRES; ++r) dense_vals[size_t(oi) * NRES + r] = values[r];
                st.store(ready, cuda::memory_order_release);
                return;
            }
            s = expected;
        }
    }
}

__global__ void expand_hash_kernel(
    const Key128* keys, const Count* vals, size_t n, Meta m,
    Key128* table_keys, uint32_t* table_index, unsigned int* table_state,
    Key128* dense_keys, Count* dense_vals, uint32_t* dense_n,
    uint32_t mask, uint32_t generation) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    Count v[NRES];
    #pragma unroll
    for (int r = 0; r < NRES; ++r) v[r] = vals[i * NRES + r];
    Key128 out;
    PackedWork base = decode_packed(keys[i], m.before_n);
    if (transition_decoded(base, m, 0, out))
        hash_insert(out, v, table_keys, table_index, table_state, dense_keys, dense_vals, dense_n, mask, generation);
    if (transition_decoded(base, m, 1, out))
        hash_insert(out, v, table_keys, table_index, table_state, dense_keys, dense_vals, dense_n, mask, generation);
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
    int w = n + 1;
    if (w > MAXW) { std::cerr << "n too large for 128-bit state; max n=" << (MAXW - 1) << "\n"; return 1; }
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
    Count init_vals[NRES]; for (int r=0;r<NRES;++r) init_vals[r]=1;
    ck(cudaMemcpy(cur_vals, init_vals, sizeof(init_vals), cudaMemcpyHostToDevice), "init vals");
    size_t peak = 1, peak_table = 0;

    Key128* table_keys = nullptr;
    uint32_t* table_index = nullptr;
    unsigned int* table_state = nullptr;
    uint32_t* dense_n = nullptr;
    size_t allocated_cap = 0;
    ck(cudaMalloc(&dense_n, sizeof(uint32_t)), "malloc dense_n");

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

        size_t cap = next_pow2(std::max<size_t>(1024, in_n * 8));
        if (cap > allocated_cap) {
            if (table_keys) { cudaFree(table_keys); cudaFree(table_index); cudaFree(table_state); }
            ck(cudaMalloc(&table_keys, cap*sizeof(Key128)), "malloc table_keys");
            ck(cudaMalloc(&table_index, cap*sizeof(uint32_t)), "malloc table_index");
            ck(cudaMalloc(&table_state, cap*sizeof(unsigned int)), "malloc table_state");
            ck(cudaMemset(table_state, 0, cap*sizeof(unsigned int)), "init state");
            allocated_cap = cap;
        }
        peak_table = std::max(peak_table, cap);
        ck(cudaMemset(dense_n, 0, sizeof(uint32_t)), "clear dense_n");

        // At most two transitions are emitted per input state. Reuse an uninitialized
        // raw output buffer: the kernel writes every live element, so zero-filling is waste.
        size_t max_out = 2 * in_n;
        if (spare_cap < max_out) {
            if (spare_keys) { cudaFree(spare_keys); cudaFree(spare_vals); }
            spare_cap = next_pow2(max_out);
            ck(cudaMalloc(&spare_keys, spare_cap * sizeof(Key128)), "malloc spare keys");
            ck(cudaMalloc(&spare_vals, spare_cap * NRES * sizeof(Count)), "malloc spare vals");
        }
        int threads=256; int blocks=(in_n+threads-1)/threads;
        expand_hash_kernel<<<blocks,threads>>>(
            cur_keys, cur_vals, in_n, m,
            table_keys, table_index, table_state, spare_keys, spare_vals, dense_n,
            static_cast<uint32_t>(cap-1), static_cast<uint32_t>(ei + 1));
        ck(cudaGetLastError(), "expand hash kernel");

        uint32_t rn = 0;
        ck(cudaMemcpy(&rn, dense_n, sizeof(rn), cudaMemcpyDeviceToHost), "copy dense_n");
        if (rn > max_out || rn >= cap) { std::cerr << "hash/output overflow\n"; return 2; }
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
            Count v=hv[i*NRES+r], mod=H_MODS[r];
            ans[r] = ans[r] >= mod - v ? ans[r] - (mod - v) : ans[r] + v;
        }
    }
    std::cout << "n=" << n << " residues=";
    for(int r=0;r<NRES;++r){ if(r) std::cout << ','; std::cout << ans[r]; }
    std::cout << " peak_states=" << peak << " hash_slots=" << peak_table << " gpu_ms=" << ms << "\n";

    cudaFree(table_keys); cudaFree(table_index); cudaFree(table_state); cudaFree(dense_n);
    cudaFree(cur_keys); cudaFree(cur_vals); cudaFree(spare_keys); cudaFree(spare_vals);
}
