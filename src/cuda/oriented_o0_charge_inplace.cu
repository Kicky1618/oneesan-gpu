#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

constexpr std::uint32_t MOD = 65521;
constexpr std::uint32_t ZETA = 61640;
constexpr std::uint32_t ZETA_INV = 19685;

#define CUDA_CHECK(expr)                                                                       \
    do {                                                                                       \
        const cudaError_t err__ = (expr);                                                      \
        if (err__ != cudaSuccess) {                                                            \
            throw std::runtime_error(std::string(#expr) + ": " + cudaGetErrorString(err__)); \
        }                                                                                      \
    } while (0)

// mode 0: normal, mode 1: right boundary (right=0), mode 2: bottom (down=0).
// B index is B+1 for B in [-1,3].  Matrix layout [d][u].
__constant__ std::uint16_t C_MAT[3][5][9];

__device__ __forceinline__ std::uint32_t add_mod(std::uint32_t a, std::uint32_t b) {
    std::uint32_t x = a + b;
    if (x >= MOD) x -= MOD;
    return x;
}

__device__ __forceinline__ std::uint32_t mul_mod(std::uint32_t a, std::uint32_t b) {
    const std::uint32_t x = a * b;
    std::uint32_t r = (x & 0xffffu) + 15u * (x >> 16);
    r = (r & 0xffffu) + 15u * (r >> 16);
    if (r >= MOD) r -= MOD;
    return r;
}

__device__ __forceinline__ int digit_charge(unsigned c) {
    return c == 1 ? 1 : c == 2 ? -1 : 0;
}

__global__ void init_charge(std::int8_t* charge, std::uint64_t total) {
    const std::uint64_t i = std::uint64_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (i >= total) return;
    std::uint64_t x = i;
    int s = 0;
    while (x) {
        const unsigned d = static_cast<unsigned>(x % 3u);
        s += digit_charge(d);
        x /= 3u;
    }
    charge[i] = static_cast<std::int8_t>(s);
}

__global__ void init_state(std::uint16_t* a, std::uint64_t total) {
    const std::uint64_t i = std::uint64_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (i < total) a[i] = 0;
    if (i == 0) {
        a[0] = 1;          // source -> right; omitted carry is +1
        a[1] = ZETA_INV;   // source -> down; vertical digit 0 is +1
    }
}

// One thread owns the three coefficients that differ only at vertical digit x.
// The horizontal carry is omitted and reconstructed from total charge Q=+1.
__global__ void charge_gate(std::uint16_t* __restrict__ a,
                            const std::int8_t* __restrict__ charge,
                            std::uint64_t groups,
                            std::uint64_t stride,
                            int mode) {
    const std::uint64_t g = std::uint64_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (g >= groups) return;

    const std::uint64_t low = g % stride;
    const std::uint64_t high = g / stride;
    const std::uint64_t base = high * (3 * stride) + low; // digit x = 0
    const int B = static_cast<int>(charge[g]);

    if (B < -1 || B > 3) {
        a[base] = 0;
        a[base + stride] = 0;
        a[base + 2 * stride] = 0;
        return;
    }

    const std::uint32_t v0 = a[base];
    const std::uint32_t v1 = a[base + stride];
    const std::uint32_t v2 = a[base + 2 * stride];
    const auto* m = C_MAT[mode][B + 1];

    std::uint32_t o0 = 0, o1 = 0, o2 = 0;
#define MAD(OUT, D, U, V) do { const std::uint32_t w = m[(D) * 3 + (U)]; if (w) OUT = add_mod(OUT, mul_mod((V), w)); } while (0)
    MAD(o0, 0, 0, v0); MAD(o0, 0, 1, v1); MAD(o0, 0, 2, v2);
    MAD(o1, 1, 0, v0); MAD(o1, 1, 1, v1); MAD(o1, 1, 2, v2);
    MAD(o2, 2, 0, v0); MAD(o2, 2, 1, v1); MAD(o2, 2, 2, v2);
#undef MAD

    a[base] = static_cast<std::uint16_t>(o0);
    a[base + stride] = static_cast<std::uint16_t>(o1);
    a[base + 2 * stride] = static_cast<std::uint16_t>(o2);
}

int charge_host(unsigned c) {
    return c == 1 ? 1 : c == 2 ? -1 : 0;
}
int code_from_charge(int q) {
    if (q == 0) return 0;
    if (q == 1) return 1;
    if (q == -1) return 2;
    return -1;
}

struct EdgeFlow { bool used=false; bool incoming=false; int dir=0; };
EdgeFlow left_edge(unsigned c)  { if(c==1)return{true,true,0};  if(c==2)return{true,false,2}; return{}; }
EdgeFlow up_edge(unsigned c)    { if(c==1)return{true,true,3};  if(c==2)return{true,false,1}; return{}; }
EdgeFlow down_edge(unsigned c)  { if(c==1)return{true,false,3}; if(c==2)return{true,true,1};  return{}; }
EdgeFlow right_edge(unsigned c) { if(c==1)return{true,false,0}; if(c==2)return{true,true,2};  return{}; }

std::uint32_t turn_weight(int in_dir, int out_dir) {
    const int d = (out_dir - in_dir + 4) & 3;
    if (d == 0) return 1;
    if (d == 1) return ZETA;
    if (d == 3) return ZETA_INV;
    return 0;
}

std::uint32_t vertex_weight(unsigned l, unsigned u, unsigned d, unsigned r) {
    const std::array<EdgeFlow,4> e = {left_edge(l), up_edge(u), down_edge(d), right_edge(r)};
    int used=0,nin=0,nout=0,idir=0,odir=0;
    for (const auto& x : e) {
        if (!x.used) continue;
        ++used;
        if (x.incoming) { ++nin; idir=x.dir; }
        else { ++nout; odir=x.dir; }
    }
    if (used == 0) return 1;
    if (used == 2 && nin == 1 && nout == 1) return turn_weight(idir, odir);
    return 0;
}

void upload_matrices() {
    std::uint16_t h[3][5][9]{};
    for (int mode=0; mode<3; ++mode) {
        for (int B=-1; B<=3; ++B) {
            for (unsigned u=0; u<3; ++u) {
                const int S = B + charge_host(u);
                const int l = code_from_charge(1-S);
                if (l < 0) continue;
                for (unsigned d=0; d<3; ++d) {
                    if (mode == 2 && d != 0) continue;
                    const int Sp = B + charge_host(d);
                    const int r = code_from_charge(1-Sp);
                    if (r < 0) continue;
                    if (mode == 1 && r != 0) continue;
                    h[mode][B+1][d*3+u] = static_cast<std::uint16_t>(
                        vertex_weight(static_cast<unsigned>(l), u, d, static_cast<unsigned>(r)));
                }
            }
        }
    }
    CUDA_CHECK(cudaMemcpyToSymbol(C_MAT, h, sizeof(h)));
}

std::uint64_t pow3(unsigned e) {
    std::uint64_t x=1;
    while(e--) x*=3;
    return x;
}

struct SolveResult {
    std::uint32_t residue=0;
    float kernel_ms=0;
    double wall_ms=0;
    std::uint64_t states=0;
    std::uint64_t bytes=0;
};

SolveResult solve(unsigned n) {
    if (n < 2 || n > 19) throw std::runtime_error("n must be in [2,19]");
    const std::uint64_t total = pow3(n);
    const std::uint64_t groups = total / 3;
    const std::uint64_t state_bytes = total * sizeof(std::uint16_t);
    const std::uint64_t charge_bytes = groups * sizeof(std::int8_t);

    std::uint16_t* a=nullptr;
    std::int8_t* q=nullptr;
    CUDA_CHECK(cudaMalloc(&a, state_bytes));
    CUDA_CHECK(cudaMalloc(&q, charge_bytes));

    constexpr int threads=256;
    auto blocks_for=[&](std::uint64_t count){ return static_cast<unsigned>((count+threads-1)/threads); };

    init_charge<<<blocks_for(groups),threads>>>(q,groups);
    CUDA_CHECK(cudaGetLastError());
    init_state<<<blocks_for(total),threads>>>(a,total);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t ev0{},ev1{};
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    const auto wall0=std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventRecord(ev0));

    auto gate=[&](unsigned x,int mode){
        const std::uint64_t stride=pow3(x);
        charge_gate<<<blocks_for(groups),threads>>>(a,q,groups,stride,mode);
        CUDA_CHECK(cudaGetLastError());
    };

    // Top row after source.
    for (unsigned x=1; x<n; ++x) gate(x, x+1<n ? 0 : 1);
    // Interior rows.
    for (unsigned y=1; y+1<n; ++y)
        for (unsigned x=0; x<n; ++x) gate(x, x+1<n ? 0 : 1);
    // Bottom row before target.
    for (unsigned x=0; x+1<n; ++x) gate(x, 2);

    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    const auto wall1=std::chrono::steady_clock::now();

    std::uint16_t left16=0,up16=0;
    const std::uint64_t up_idx=pow3(n-1);
    CUDA_CHECK(cudaMemcpy(&left16,a,sizeof(left16),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&up16,a+up_idx,sizeof(up16),cudaMemcpyDeviceToHost));
    const std::uint32_t left=left16,up=up16;
    std::uint32_t residue=left+static_cast<std::uint32_t>(std::uint64_t{up}*ZETA%MOD);
    if(residue>=MOD)residue-=MOD;

    float ms=0;
    CUDA_CHECK(cudaEventElapsedTime(&ms,ev0,ev1));
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(q));
    return {residue,ms,std::chrono::duration<double,std::milli>(wall1-wall0).count(),total,state_bytes+charge_bytes};
}

} // namespace

int main(int argc,char**argv){
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
    upload_matrices();
    std::cerr<<"gpu="<<prop.name<<" mod="<<MOD<<" zeta="<<ZETA<<'\n';
    const unsigned first=argc>1?std::strtoul(argv[1],nullptr,10):12;
    const unsigned last=argc>2?std::strtoul(argv[2],nullptr,10):first;
    const unsigned reps=argc>3?std::strtoul(argv[3],nullptr,10):3;
    (void)solve(2);
    for(unsigned n=first;n<=last;++n){
        SolveResult best{};best.kernel_ms=1e30f;double sum=0;
        for(unsigned rep=0;rep<reps;++rep){auto r=solve(n);sum+=r.kernel_ms;if(r.kernel_ms<best.kernel_ms)best=r;}
        std::cout<<"n="<<n<<" residue="<<best.residue<<" states="<<best.states
                 <<" workspace_mib="<<std::fixed<<std::setprecision(1)<<(double(best.bytes)/(1<<20))
                 <<" best_ms="<<std::setprecision(3)<<best.kernel_ms<<" avg_ms="<<(sum/reps)<<'\n';
    }
}
