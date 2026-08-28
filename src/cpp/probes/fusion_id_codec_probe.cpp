#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

using u8 = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

static u64 C[29][29];
static void build_binom() {
    for (int n=0;n<=28;++n) {
        C[n][0]=C[n][n]=1;
        for (int k=1;k<n;++k) C[n][k]=C[n-1][k-1]+C[n-1][k];
    }
}
static u64 catalan(int n) { return C[2*n][n]/u64(n+1); }

// Colexicographic rank of a fixed-popcount occupancy mask.  For a mask with
// selected positions p_1 < ... < p_m this is sum_j binom(p_j,j).
static u64 mask_rank_slow(u32 mask, int W) {
    u64 r=0; int j=0;
    for (int p=0;p<W;++p) if ((mask>>p)&1u) {
        ++j; r += C[p][j];
    }
    return r;
}

struct MaskChunk { u64 delta=0; u8 dk=0; };
static MaskChunk MASK7[4][29][128];
static void build_mask7(int W) {
    for (int ch=0;ch<4;++ch) {
        int base=7*ch, len=std::min(7,W-base);
        if (len<=0) continue;
        for (int kin=0;kin<=W;++kin) for (int pat=0;pat<128;++pat) {
            if (pat >= (1<<len)) continue;
            int k=kin; u64 d=0;
            for (int b=0;b<len;++b) if ((pat>>b)&1) {
                ++k; d += C[base+b][k];
            }
            MASK7[ch][kin][pat] = {d,u8(k-kin)};
        }
    }
}
static u64 mask_rank_chunked(u32 mask, int W) {
    int k=0; u64 r=0;
    for (int ch=0;ch<4 && 7*ch<W;++ch) {
        int len=std::min(7,W-7*ch);
        int pat=(mask>>(7*ch))&((1u<<len)-1u);
        auto e=MASK7[ch][k][pat]; r+=e.delta; k+=e.dk;
    }
    return r;
}

// Two-colour Motzkin fusion path alphabet:
// A=-1, B=0, C=0, D=+1, encoded as 0,1,2,3.
static int delta(int s) { return s==0?-1:s==3?+1:0; }
static u64 WAYS[15][16]; // WAYS[len][height] -> end at height 0
static void build_ways() {
    WAYS[0][0]=1;
    for (int len=1;len<=14;++len) for (int h=0;h<15;++h) {
        u64 z=2*WAYS[len-1][h];
        if (h) z+=WAYS[len-1][h-1];
        if (h+1<16) z+=WAYS[len-1][h+1];
        WAYS[len][h]=z;
    }
}

static u32 fusion_rank_slow(u32 code, int r) {
    u64 rank=0; int h=0;
    for (int pos=0;pos<r;++pos) {
        int s=(code>>(2*pos))&3u;
        int rem=r-pos-1;
        for (int t=0;t<s;++t) {
            int hh=h+delta(t);
            if (hh>=0) rank += WAYS[rem][hh];
        }
        h += delta(s);
        assert(h>=0);
    }
    assert(h==0 && rank < catalan(r+1));
    return u32(rank);
}

struct FusionChunk { u32 delta_rank=0; u8 end_height=0; u8 valid=0; };
// The largest table is tiny; allocate a uniform upper bound for the probe.
static FusionChunk F4[14][4][15][256];
static u64 fusion_chunk_entries=0;

static void build_fusion4() {
    for (int r=0;r<=13;++r) {
        int nch=(r+3)/4;
        for (int ch=0;ch<nch;++ch) {
            int pos=4*ch, len=std::min(4,r-pos), npat=1<<(2*len);
            for (int hin=0;hin<=r;++hin) for (int pat=0;pat<npat;++pat) {
                ++fusion_chunk_entries;
                u64 dr=0; int h=hin; bool ok=true;
                for (int q=0;q<len;++q) {
                    int s=(pat>>(2*q))&3;
                    int rem=r-(pos+q)-1;
                    for (int t=0;t<s;++t) {
                        int hh=h+delta(t);
                        if (hh>=0) dr += WAYS[rem][hh];
                    }
                    h+=delta(s);
                    if (h<0) {ok=false;break;}
                }
                assert(dr <= 0xffffffffULL);
                F4[r][ch][hin][pat]={u32(dr),u8(std::max(h,0)),u8(ok)};
            }
        }
    }
}

static u32 fusion_rank_chunked(u32 code, int r) {
    u64 rank=0; int h=0;
    int nch=(r+3)/4;
    for (int ch=0;ch<nch;++ch) {
        int len=std::min(4,r-4*ch);
        int pat=(code>>(8*ch))&((1<<(2*len))-1);
        auto e=F4[r][ch][h][pat];
        assert(e.valid); rank+=e.delta_rank; h=e.end_height;
    }
    assert(h==0 && rank<catalan(r+1));
    return u32(rank);
}

static void gen_paths_rec(int r,int pos,int h,u32 code,std::vector<u32>&out) {
    if (pos==r) { if (h==0) out.push_back(code); return; }
    // Lexicographic A,B,C,D order, matching the ranker above.
    if (h>0) gen_paths_rec(r,pos+1,h-1,code|(0u<<(2*pos)),out);
    gen_paths_rec(r,pos+1,h,code|(1u<<(2*pos)),out);
    gen_paths_rec(r,pos+1,h,code|(2u<<(2*pos)),out);
    gen_paths_rec(r,pos+1,h+1,code|(3u<<(2*pos)),out);
}

static u64 state_count(int W) {
    u64 z=0;
    for (int m=1;m<=W;m+=2) z += C[W][m]*catalan((m+1)/2);
    return z;
}

int main() {
    build_binom(); build_ways();

    // Mask rank: four 7-bit automaton lookups at W=28.
    build_mask7(28);
    std::mt19937 rng(1234567);
    for (int t=0;t<200000;++t) {
        u32 m=rng()&((1u<<28)-1u);
        assert(mask_rank_slow(m,28)==mask_rank_chunked(m,28));
    }

    // Fusion rank: at most four four-symbol lookups for r<=13.  Verify every
    // valid fusion word, not just random samples.
    build_fusion4();
    u64 unrank_entries=0;
    for (int r=0;r<=13;++r) {
        std::vector<u32> words;
        gen_paths_rec(r,0,0,0,words);
        assert(words.size()==catalan(r+1));
        for (u32 i=0;i<words.size();++i) {
            u32 a=fusion_rank_slow(words[i],r);
            u32 b=fusion_rank_chunked(words[i],r);
            assert(a==i && b==i);
        }
        unrank_entries += words.size();
        std::cout<<"r="<<r<<" fusion_states="<<words.size()<<" OK\n";
    }

    const u64 main28=state_count(28), blocked27=state_count(27);
    assert(main28==385719506620ULL);
    assert(blocked27==135015505407ULL);
    assert(unrank_entries==3707851ULL);

    // A FusionChunk occupies 8 bytes on normal ABIs.  MASK7 is about 114 KiB.
    std::cout<<"main_W28="<<main28<<" blocked_W27="<<blocked27<<"\n";
    std::cout<<"mask7_bytes="<<sizeof(MASK7)<<"\n";
    std::cout<<"fusion4_entries="<<fusion_chunk_entries
             <<" fusion4_bytes_live="<<fusion_chunk_entries*sizeof(FusionChunk)<<"\n";
    std::cout<<"fusion_unrank_entries="<<unrank_entries
             <<" fusion_unrank_u32_bytes="<<unrank_entries*4<<"\n";
    std::cout<<"max_rank_lookups: mask=4 fusion=4\n";
    return 0;
}
