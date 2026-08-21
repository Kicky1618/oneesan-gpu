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

__host__ __device__ static inline void decode_state(
    const Key128& k, int n, uint8_t* deg, uint8_t* comp, uint8_t* flags, bool& done) {
    done = get_bits(k, 0, 1);
    int p = 1;
    for (int i = 0; i < n; ++i) {
        deg[i] = get_bits(k, p, 2); p += 2;
        comp[i] = get_bits(k, p, 4); p += 4;
    }
    for (int c = 0; c <= MAXW; ++c) flags[c] = 0;
    for (int c = 1; c <= n; ++c) { flags[c] = get_bits(k, p, 2); p += 2; }
}

__host__ __device__ static inline Key128 encode_canonical(
    int n, uint8_t* deg, uint8_t* comp, uint8_t* flags, bool done) {
    uint8_t remap[MAXW + 1] = {};
    uint8_t nf[MAXW + 1] = {};
    uint8_t next = 1;
    for (int i = 0; i < n; ++i) {
        uint8_t c = comp[i];
        if (!c) continue;
        if (!remap[c]) { remap[c] = next; nf[next] = flags[c]; ++next; }
        comp[i] = remap[c];
    }
    Key128 k{0,0};
    put_bits(k, 0, 1, done ? 1 : 0);
    int p = 1;
    for (int i = 0; i < n; ++i) {
        put_bits(k, p, 2, deg[i]); p += 2;
        put_bits(k, p, 4, comp[i]); p += 4;
    }
    for (int c = 1; c <= n; ++c) { put_bits(k, p, 2, c < next ? nf[c] : 0); p += 2; }
    return k;
}

__device__ static inline bool transition_one(
    const Key128& in, const Meta& m, int take, Key128& out) {
    uint8_t deg[MAXWORK] = {}, comp[MAXWORK] = {}, flags[MAXW + 1] = {};
    bool done = false;
    decode_state(in, m.before_n, deg, comp, flags, done);

    if (take) {
        if (done) return false;
        if (deg[m.iu] >= m.endpoint_max_u || deg[m.iv] >= m.endpoint_max_v) return false;
        ++deg[m.iu]; ++deg[m.iv];
        uint8_t a = comp[m.iu], b = comp[m.iv];
        if (!a && !b) {
            bool used[MAXW + 1] = {};
            for (int i = 0; i < m.work_n; ++i) if (comp[i]) used[comp[i]] = true;
            uint8_t q = 1; while (q <= MAXW && used[q]) ++q;
            if (q > MAXW) return false;
            flags[q] = 0;
            comp[m.iu] = comp[m.iv] = q;
            if (m.terminal_mask & (1u << m.iu)) flags[q] |= 1;
            if (m.terminal_mask & (1u << m.iv)) flags[q] |= 2;
        } else if (!a || !b) {
            uint8_t q = a ? a : b;
            comp[m.iu] = comp[m.iv] = q;
            if (m.terminal_mask & (1u << m.iu)) flags[q] |= 1;
            if (m.terminal_mask & (1u << m.iv)) flags[q] |= 2;
        } else {
            if (a == b) return false;
            uint8_t keep = a < b ? a : b, kill = a < b ? b : a;
            flags[keep] |= flags[kill];
            for (int i = 0; i < m.work_n; ++i) if (comp[i] == kill) comp[i] = keep;
            flags[kill] = 0;
        }
    }

    for (int i = 0; i < m.work_n; ++i) {
        if (!(m.forget_mask & (1u << i))) continue;
        bool terminal = m.terminal_mask & (1u << i);
        if ((terminal && deg[i] != 1) || (!terminal && deg[i] != 0 && deg[i] != 2)) return false;
        uint8_t q = comp[i];
        comp[i] = 0;
        if (q) {
            bool alive = false;
            for (int j = 0; j < m.work_n; ++j) if (comp[j] == q) { alive = true; break; }
            if (!alive) {
                if (flags[q] == 3) done = true;
                else return false;
            }
        }
    }

    uint8_t od[MAXW] = {}, oc[MAXW] = {};
    for (int i = 0; i < m.after_n; ++i) {
        int src = m.after_src[i];
        od[i] = deg[src]; oc[i] = comp[src];
    }
    out = encode_canonical(m.after_n, od, oc, flags, done);
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
    if (transition_one(keys[i], m, 0, out))
        hash_insert(out, vals[i], mod, table_keys, table_vals, table_state, occupied, occupied_n, mask, generation);
    if (transition_one(keys[i], m, 1, out))
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
