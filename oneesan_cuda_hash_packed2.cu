#include <cuda_runtime.h>
#include <cuda/atomic>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>
#include <utility>

using Count = unsigned long long;
static constexpr int MAXW = 15;
static constexpr int MAXWORK = MAXW + 2;

struct Key128 {
    uint64_t lo, hi;
    __host__ __device__ bool operator==(const Key128& o) const { return lo == o.lo && hi == o.hi; }
};

struct Meta {
    int before_n, work_n, after_n;
    int iu, iv;
    uint8_t endpoint_max_u, endpoint_max_v;
    uint16_t forget_mask;
    uint16_t terminal_mask;
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
    uint32_t flags = 0;
    bool done = false;
};

__device__ __forceinline__ uint8_t ws_deg(const PackedWork& s, int i) {
    return static_cast<uint8_t>((s.deg >> (2 * i)) & 3ULL);
}
__device__ __forceinline__ void ws_set_deg(PackedWork& s, int i, uint8_t v) {
    const uint64_t sh = 2ULL * i;
    s.deg = (s.deg & ~(3ULL << sh)) | (uint64_t(v) << sh);
}
__device__ __forceinline__ uint8_t ws_comp(const PackedWork& s, int i) {
    if (i < 16) return static_cast<uint8_t>((s.comp_lo >> (4 * i)) & 15ULL);
    return static_cast<uint8_t>((s.comp_hi >> (4 * (i - 16))) & 15ULL);
}
__device__ __forceinline__ void ws_set_comp(PackedWork& s, int i, uint8_t v) {
    if (i < 16) {
        const uint64_t sh = 4ULL * i;
        s.comp_lo = (s.comp_lo & ~(15ULL << sh)) | (uint64_t(v) << sh);
    } else {
        const uint64_t sh = 4ULL * (i - 16);
        s.comp_hi = (s.comp_hi & ~(15ULL << sh)) | (uint64_t(v) << sh);
    }
}
__device__ __forceinline__ uint8_t ws_flag(const PackedWork& s, int c) {
    return static_cast<uint8_t>((s.flags >> (2 * c)) & 3u);
}
__device__ __forceinline__ void ws_set_flag(PackedWork& s, int c, uint8_t v) {
    const uint32_t sh = 2u * c;
    s.flags = (s.flags & ~(3u << sh)) | (uint32_t(v) << sh);
}
__device__ __forceinline__ void ws_or_flag(PackedWork& s, int c, uint8_t v) {
    s.flags |= uint32_t(v) << (2 * c);
}

__device__ __forceinline__ PackedWork decode_packed(const Key128& k, int n) {
    PackedWork s;
    s.done = get_bits(k, 0, 1);
    int p = 1;
    #pragma unroll
    for (int i = 0; i < MAXWORK; ++i) {
        if (i < n) {
            ws_set_deg(s, i, static_cast<uint8_t>(get_bits(k, p, 2))); p += 2;
            ws_set_comp(s, i, static_cast<uint8_t>(get_bits(k, p, 4))); p += 4;
        }
    }
    #pragma unroll
    for (int c = 1; c <= MAXW; ++c) {
        if (c <= n) { ws_set_flag(s, c, static_cast<uint8_t>(get_bits(k, p, 2))); p += 2; }
    }
    return s;
}

__device__ __forceinline__ Key128 encode_packed_canonical(int n, PackedWork s) {
    uint64_t remap = 0;   // 4 bits per old label (labels 0..15)
    uint32_t nf = 0;      // 2 bits per new label
    uint8_t next = 1;
    #pragma unroll
    for (int i = 0; i < MAXW; ++i) {
        if (i >= n) continue;
        uint8_t c = ws_comp(s, i);
        if (!c) continue;
        uint8_t r = static_cast<uint8_t>((remap >> (4 * c)) & 15ULL);
        if (!r) {
            r = next++;
            remap |= uint64_t(r) << (4 * c);
            nf |= uint32_t(ws_flag(s, c)) << (2 * r);
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
            put_bits(k, p, 4, ws_comp(s, i)); p += 4;
        }
    }
    #pragma unroll
    for (int c = 1; c <= MAXW; ++c) {
        if (c <= n) { put_bits(k, p, 2, (nf >> (2 * c)) & 3u); p += 2; }
    }
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
            uint16_t used = 0;
            #pragma unroll
            for (int i = 0; i < MAXWORK; ++i)
                if (i < m.work_n) used |= uint16_t(1u << ws_comp(s, i));
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
    Key128 key, Count value, Count mod,
    Key128* table_keys, Count* table_vals, unsigned int* table_state,
    uint32_t* occupied, uint32_t* occupied_n, uint32_t mask, uint32_t generation) {
    const unsigned int locked = generation << 1;
    const unsigned int ready = locked | 1u;
    uint32_t slot = static_cast<uint32_t>(hash_key(key)) & mask;
    for (uint32_t probe = 0; probe <= mask; ++probe, slot = (slot + 1) & mask) {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> st(table_state[slot]);
        unsigned int s = st.load(cuda::memory_order_acquire);

        for (;;) {
            if (s == ready) {
                if (table_keys[slot] == key) {
                    atomic_add_mod(&table_vals[slot], value, mod);
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
                table_keys[slot] = key;
                table_vals[slot] = value;
                uint32_t oi = atomicAdd(occupied_n, 1u);
                occupied[oi] = slot;
                st.store(ready, cuda::memory_order_release);
                return;
            }
            s = expected;
        }
    }
}

__global__ void expand_hash_kernel(
    const Key128* keys, const Count* vals, size_t n, Meta m, Count mod,
    Key128* table_keys, Count* table_vals, unsigned int* table_state,
    uint32_t* occupied, uint32_t* occupied_n, uint32_t mask, uint32_t generation) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    Key128 out;
    PackedWork base = decode_packed(keys[i], m.before_n);
    if (transition_decoded(base, m, 0, out))
        hash_insert(out, vals[i], mod, table_keys, table_vals, table_state, occupied, occupied_n, mask, generation);
    if (transition_decoded(base, m, 1, out))
        hash_insert(out, vals[i], mod, table_keys, table_vals, table_state, occupied, occupied_n, mask, generation);
}

__global__ void gather_kernel(
    const Key128* table_keys, const Count* table_vals, const uint32_t* occupied,
    size_t n, Key128* out_keys, Count* out_vals) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t slot = occupied[i];
    out_keys[i] = table_keys[slot];
    out_vals[i] = table_vals[slot];
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
    Count mod = argc > 2 ? std::stoull(argv[2]) : 2305843009213693921ULL;
    if (mod >= (1ULL << 63)) { std::cerr << "mod must be < 2^63\n"; return 1; }
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

    thrust::device_vector<Key128> keys(1);
    thrust::device_vector<Count> vals(1,1);
    keys[0] = Key128{0,0};
    size_t peak = 1, peak_table = 0;

    Key128* table_keys = nullptr;
    Count* table_vals = nullptr;
    unsigned int* table_state = nullptr;
    uint32_t* occupied = nullptr;
    uint32_t* occupied_n = nullptr;
    size_t allocated_cap = 0;
    ck(cudaMalloc(&occupied_n, sizeof(uint32_t)), "malloc occupied_n");

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

        size_t in_n = keys.size();
        size_t cap = next_pow2(std::max<size_t>(1024, in_n * 8));
        if (cap > allocated_cap) {
            if (table_keys) { cudaFree(table_keys); cudaFree(table_vals); cudaFree(table_state); cudaFree(occupied); }
            ck(cudaMalloc(&table_keys, cap*sizeof(Key128)), "malloc table_keys");
            ck(cudaMalloc(&table_vals, cap*sizeof(Count)), "malloc table_vals");
            ck(cudaMalloc(&table_state, cap*sizeof(unsigned int)), "malloc table_state");
            ck(cudaMalloc(&occupied, cap*sizeof(uint32_t)), "malloc occupied");
            ck(cudaMemset(table_state, 0, cap*sizeof(unsigned int)), "init state");
            allocated_cap = cap;
        }
        peak_table = std::max(peak_table, cap);
        ck(cudaMemset(occupied_n, 0, sizeof(uint32_t)), "clear occupied_n");

        int threads=256; int blocks=(in_n+threads-1)/threads;
        expand_hash_kernel<<<blocks,threads>>>(
            thrust::raw_pointer_cast(keys.data()), thrust::raw_pointer_cast(vals.data()), in_n, m, mod,
            table_keys, table_vals, table_state, occupied, occupied_n, static_cast<uint32_t>(cap-1), static_cast<uint32_t>(ei + 1));
        ck(cudaGetLastError(), "expand hash kernel");

        uint32_t rn = 0;
        ck(cudaMemcpy(&rn, occupied_n, sizeof(rn), cudaMemcpyDeviceToHost), "copy occupied_n");
        if (rn >= cap) { std::cerr << "hash table overflow\n"; return 2; }
        thrust::device_vector<Key128> next_keys(rn);
        thrust::device_vector<Count> next_vals(rn);
        if (rn) {
            int gb=(rn+threads-1)/threads;
            gather_kernel<<<gb,threads>>>(table_keys,table_vals,occupied,rn,
                thrust::raw_pointer_cast(next_keys.data()),thrust::raw_pointer_cast(next_vals.data()));
            ck(cudaGetLastError(), "gather kernel");
        }
        keys.swap(next_keys); vals.swap(next_vals);
        peak=std::max<size_t>(peak,rn);
        if ((ei+1)%w==0 || ei+1==(int)edges.size())
            std::cerr << "edge " << ei+1 << "/" << edges.size() << " frontier=" << after.size() << " states=" << rn << "\n";
    }
    cudaEventRecord(ev1); cudaEventSynchronize(ev1);
    float ms=0; cudaEventElapsedTime(&ms,ev0,ev1);

    thrust::host_vector<Key128> hk=keys;
    thrust::host_vector<Count> hv=vals;
    Count ans=0;
    for(size_t i=0;i<hk.size();++i) if(get_bits(hk[i],0,1)) {
        ans = ans >= mod - hv[i] ? ans - (mod - hv[i]) : ans + hv[i];
    }
    std::cout << "n=" << n << " residue=" << ans << " mod=" << mod
              << " peak_states=" << peak << " hash_slots=" << peak_table << " gpu_ms=" << ms << "\n";

    cudaFree(table_keys); cudaFree(table_vals); cudaFree(table_state); cudaFree(occupied); cudaFree(occupied_n);
}
