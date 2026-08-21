#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>

using Count = unsigned long long;
using MateID = unsigned long long;
using Code = unsigned long long;

static constexpr int MAXW = 28; // n<=27 => width=n+1<=28

enum MateValue : uint8_t { N = 0, R = 1, L = 2, X = 3 };
enum MateValuePair : uint8_t {
    NN=0x0, NR=0x1, NL=0x2, NX=0x3,
    RN=0x4, RR=0x5, RL=0x6, RX=0x7,
    LN=0x8, LR=0x9, LL=0xa, LX=0xb,
    XN=0xc, XR=0xd, XL=0xe, XX=0xf
};

__constant__ unsigned long long D_DP[MAXW+1][MAXW+1];
__constant__ Count D_MOD;

__host__ __device__ static inline MateValue mate_get(MateID m, int k) {
    return MateValue((m >> (2*k)) & 3ULL);
}
__host__ __device__ static inline MateValuePair mate_get_pair(MateID m, int p) {
    return MateValuePair((m >> (2*(p-1))) & 0xfULL);
}
__host__ __device__ static inline MateID mate_set(MateID m, int k, MateValue v) {
    const MateID mask = 3ULL << (2*k);
    return (m & ~mask) | (MateID(v) << (2*k));
}
__host__ __device__ static inline MateID mate_set_pair(MateID m, int p, MateValuePair w) {
    const MateID mask = 0xfULL << (2*(p-1));
    return (m & ~mask) | (MateID(w) << (2*(p-1)));
}
__host__ __device__ static inline MateID mate_shrink(MateID m, int k) {
    // Remove symbol at position k and shift higher positions down by one.
    const MateID lowmask = (k == 0) ? 0ULL : ((1ULL << (2*k)) - 1ULL);
    const MateID lo = m & lowmask;
    const MateID hi = m & ~((1ULL << (2*(k+1))) - 1ULL);
    return lo | (hi >> 2);
}

__device__ __forceinline__ void add_mod(Count& a, Count b) {
    Count mod = D_MOD;
    a = (a >= mod - b) ? (a - (mod - b)) : (a + b);
}

__device__ __forceinline__ Code rank_mate(MateID m, int width) {
    // Exact same N,R,L lexicographic order as GGCount MateCodec.
    Code rank = 0;
    int h = 1;
    #pragma unroll
    for (int pos = MAXW-1; pos >= 0; --pos) {
        if (pos >= width) continue;
        MateValue s = mate_get(m, pos);
        int rem = pos;
        if (s > N) rank += D_DP[rem][h];
        if (s > R && h > 0) rank += D_DP[rem][h-1];
        if (s == R) --h;
        else if (s == L) ++h;
    }
    return rank;
}

__device__ __forceinline__ void update_one(
    MateID mate, int p, int width, Count* value, Count* deferred) {

    Code code = rank_mate(mate, width);
    Count& c = value[code];
    MateValuePair w = mate_get_pair(mate, p);

    switch (w) {
    case NN: { // [..] -> [()] plus blocked-state transfer
        MateID sm = mate_shrink(mate, p);
        Count& d = deferred[rank_mate(sm, width-1)];
        if (c != 0) {
            MateID t = mate_set_pair(mate, p, LR);
            add_mod(value[rank_mate(t, width)], c);
        }
        add_mod(c, d);
        d = 0;
        break;
    }
    case NL: // [.(] <-> [(.]
    case NR: { // [.)] <-> [).]
        MateID sm = mate_shrink(mate, p);
        Count& d = deferred[rank_mate(sm, width-1)];
        MateID t = mate_set_pair(mate, p, (w == NL) ? LN : RN);
        Count& cc = value[rank_mate(t, width)];
        if (p == 1) {
            add_mod(d, cc);
            add_mod(cc, c);
            add_mod(c, d);
            d = 0;
        } else {
            Count tmp = c;
            add_mod(c, cc);
            add_mod(c, d);
            d = tmp;
        }
        break;
    }
    case LL: { // [((---)] -> [..---(]
        MateID t = mate_set_pair(mate, p, NN);
        int q = p - 1;
        int s = 1;
        while (s > 0) {
            --q;
            MateValue v = mate_get(t, q);
            if (v == L) ++s;
            else if (v == R) --s;
        }
        t = mate_set(t, q, L);
        if (p == 1) {
            add_mod(value[rank_mate(t, width)], c);
        } else {
            MateID sm = mate_shrink(t, p-1);
            add_mod(deferred[rank_mate(sm, width-1)], c);
        }
        break;
    }
    case RR: { // [(---))] -> [)---..]
        MateID t = mate_set_pair(mate, p, NN);
        int q = p;
        int s = 1;
        while (s > 0) {
            ++q;
            MateValue v = mate_get(t, q);
            if (v == L) --s;
            else if (v == R) ++s;
        }
        t = mate_set(t, q, R);
        if (p == 1) {
            add_mod(value[rank_mate(t, width)], c);
        } else {
            MateID sm = mate_shrink(t, p-1);
            add_mod(deferred[rank_mate(sm, width-1)], c);
        }
        break;
    }
    case RL: { // [)(] -> [..]
        MateID t = mate_set_pair(mate, p, NN);
        if (p == 1) {
            add_mod(value[rank_mate(t, width)], c);
        } else {
            MateID sm = mate_shrink(t, p-1);
            add_mod(deferred[rank_mate(sm, width-1)], c);
        }
        break;
    }
    default:
        break;
    }
}

// Process one transition-closed group from Sec. 4.4 of Iwashita et al.
// group encodes g(c_k) for all positions except p-1,p, where g(N)=0, g(R/L)=1.
// We enumerate the valid Motzkin states in the same lexicographic order as GGCount.
__device__ void process_group(uint32_t group, int p, int width,
                              Count* value, Count* deferred) {
    // Iterative DFS, high position -> low position. Options are N,R,L order.
    uint8_t nextopt[MAXW];
    int8_t height[MAXW+1];
    #pragma unroll
    for (int i=0;i<MAXW;++i) nextopt[i]=0;
    height[0] = 1;
    int depth = 0;
    MateID mate = 0;

    while (depth >= 0) {
        if (depth == width) {
            if (height[depth] == 0) update_one(mate, p, width, value, deferred);
            --depth;
            continue;
        }

        const int pos = width - 1 - depth;
        const int h = height[depth];
        uint8_t& cursor = nextopt[depth];
        bool descended = false;

        // Determine allowed option set, still preserving global N,R,L order.
        const bool active = (pos == p || pos == p-1);
        bool occupied = false;
        if (!active) {
            const int bit = (pos < p-1) ? pos : (pos - 2);
            occupied = ((group >> bit) & 1u) != 0;
        }

        while (cursor < 3) {
            uint8_t opt = cursor++;
            if (!active) {
                if (!occupied && opt != N) continue;
                if ( occupied && opt == N) continue;
            }

            int nh = h;
            if (opt == R) --nh;
            else if (opt == L) ++nh;
            if (nh < 0 || nh > MAXW) continue;
            // There are 'pos' lower symbols left after choosing this position.
            if (D_DP[pos][nh] == 0) continue;

            mate = mate_set(mate, pos, MateValue(opt));
            ++depth;
            height[depth] = static_cast<int8_t>(nh);
            if (depth < width) nextopt[depth] = 0;
            descended = true;
            break;
        }

        if (!descended) {
            cursor = 0;
            --depth;
        }
    }
}

__global__ void update_groups_kernel(int p, int width, Count* value, Count* deferred,
                                     uint32_t num_groups, uint32_t* next_group) {
    for (;;) {
        uint32_t g = atomicAdd(next_group, 1u);
        if (g >= num_groups) return;
        process_group(g, p, width, value, deferred);
    }
}

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::cerr << what << ": " << cudaGetErrorString(e) << "\n";
        std::exit(1);
    }
}

static unsigned long long hdp[MAXW+1][MAXW+1];
static void build_dp() {
    for (int h=0; h<=MAXW; ++h) hdp[0][h] = (h==0 ? 1ULL : 0ULL);
    for (int w=1; w<=MAXW; ++w) {
        for (int h=0; h<=MAXW; ++h) {
            unsigned long long x = hdp[w-1][h];
            if (h>0) x += hdp[w-1][h-1];
            if (h<MAXW) x += hdp[w-1][h+1];
            hdp[w][h] = x;
        }
    }
}

static Code host_rank(MateID m, int width) {
    Code rank=0; int h=1;
    for (int pos=width-1; pos>=0; --pos) {
        MateValue s=mate_get(m,pos);
        if (s>N) rank += hdp[pos][h];
        if (s>R && h>0) rank += hdp[pos][h-1];
        if (s==R) --h; else if (s==L) ++h;
    }
    return rank;
}

int main(int argc, char** argv) {
    int n = argc > 1 ? std::atoi(argv[1]) : 10;
    Count mod = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 2305843009213693951ULL;
    int width = n + 1;
    if (n < 1 || width > MAXW) {
        std::cerr << "supported n=1.." << (MAXW-1) << "\n";
        return 1;
    }
    if (width < 2) return 1;

    build_dp();
    ck(cudaMemcpyToSymbol(D_DP, hdp, sizeof(hdp)), "copy dp");
    ck(cudaMemcpyToSymbol(D_MOD, &mod, sizeof(mod)), "copy modulus");

    const Code main_n = hdp[width][1];
    const Code deferred_n = hdp[width-1][1];
    const size_t bytes = size_t(main_n + deferred_n) * sizeof(Count);

    Count* value=nullptr; Count* deferred=nullptr;
    ck(cudaMalloc(&value, size_t(main_n)*sizeof(Count)), "malloc value");
    ck(cudaMalloc(&deferred, size_t(deferred_n)*sizeof(Count)), "malloc deferred");
    ck(cudaMemset(value, 0, size_t(main_n)*sizeof(Count)), "clear value");
    ck(cudaMemset(deferred, 0, size_t(deferred_n)*sizeof(Count)), "clear deferred");

    MateID init = MateID(R) << (2*(width-1));
    Count one = 1;
    Code init_code = host_rank(init,width);
    ck(cudaMemcpy(value + init_code, &one, sizeof(one), cudaMemcpyHostToDevice), "init count");

    uint32_t* next_group=nullptr;
    ck(cudaMalloc(&next_group,sizeof(uint32_t)), "malloc next_group");
    const uint32_t num_groups = 1u << (width-2);

    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0);
    for (int i=0; i<width; ++i) {
        for (int j=0; j<width-1; ++j) {
            int p = width - j - 1;
            ck(cudaMemset(next_group,0,sizeof(uint32_t)), "reset group counter");
            // Enough persistent workers to fill the GPU; work is dynamically scheduled.
            int blocks = 256, threads = 128;
            update_groups_kernel<<<blocks,threads>>>(p,width,value,deferred,num_groups,next_group);
            ck(cudaGetLastError(), "update kernel");
            ck(cudaDeviceSynchronize(), "sync update");
        }
        std::cerr << "row " << (i+1) << "/" << width << "\n";
    }
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms=0; cudaEventElapsedTime(&ms,e0,e1);

    MateID finalm = MateID(R); // R at position 0
    Code final_code = host_rank(finalm,width);
    Count ans=0;
    ck(cudaMemcpy(&ans,value+final_code,sizeof(ans),cudaMemcpyDeviceToHost), "copy answer");

    std::cout << "backend=gridfp-gpu n=" << n
              << " residue=" << ans
              << " modulus=" << mod
              << " main_states=" << main_n
              << " blocked_states=" << deferred_n
              << " count_bytes=" << bytes
              << " gpu_ms=" << ms << "\n";

    cudaFree(next_group); cudaFree(value); cudaFree(deferred);
    return 0;
}
