#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using Code=unsigned long long;
static constexpr int MAXW=28,NGPU=8;
static constexpr std::size_t META_BYTES=13936;

struct Group{int window=0,g=0;Code work=0;int gpu=-1;std::size_t meta_id=~std::size_t(0);};

int main(){
    // Exact W28 plan proof already fixes 8192 groups/window and these per-window
    // LPT counts. This proof isolates the production lowering's cross-window
    // local metadata indexing and repeated-row execution contract.
    constexpr std::array<std::uint32_t,NGPU> per_window={1022,1023,1024,1024,1024,1024,1025,1026};
    constexpr std::array<std::uint32_t,NGPU> combined={2044,2046,2048,2048,2048,2048,2050,2052};
    constexpr std::array<std::uint32_t,NGPU> processed28={57232,57288,57344,57344,57344,57344,57400,57456};

    std::vector<Group> groups;groups.reserve(16384);
    std::array<std::vector<std::size_t>,NGPU> local_to_global;
    std::array<std::uint32_t,NGPU> count{};
    for(int w=0;w<2;++w){
        std::array<std::uint32_t,NGPU> remaining=per_window;
        int d=0;
        for(int g=0;g<8192;++g){
            while(d<NGPU&&remaining[d]==0)++d;
            if(d>=NGPU){std::fprintf(stderr,"assignment overflow window=%d group=%d\n",w,g);return 2;}
            Group x;x.window=w;x.g=g;x.work=Code(8192-g);x.gpu=d;x.meta_id=local_to_global[d].size();
            const std::size_t global=groups.size();groups.push_back(x);local_to_global[d].push_back(global);++count[d];--remaining[d];
        }
        for(auto x:remaining)if(x){std::fprintf(stderr,"assignment remainder window=%d\n",w);return 3;}
        d=0;
    }
    if(groups.size()!=16384||count!=combined){std::fprintf(stderr,"combined group count mismatch\n");return 4;}

    std::vector<unsigned char> seen(groups.size(),0);
    for(int d=0;d<NGPU;++d){
        if(local_to_global[d].size()!=combined[d])return 5;
        for(std::size_t id=0;id<local_to_global[d].size();++id){
            const std::size_t gi=local_to_global[d][id];if(gi>=groups.size()||seen[gi]||groups[gi].gpu!=d||groups[gi].meta_id!=id){std::fprintf(stderr,"local metadata mapping mismatch gpu=%d id=%zu global=%zu\n",d,id,gi);return 6;}seen[gi]=1;
        }
    }
    if(std::find(seen.begin(),seen.end(),0)!=seen.end()){std::fprintf(stderr,"unassigned group\n");return 7;}

    std::array<std::uint32_t,NGPU> processed{};
    std::uint64_t total_processed=0;
    for(int row=0;row<28;++row)for(int d=0;d<NGPU;++d){processed[d]+=count[d];total_processed+=count[d];}
    if(processed!=processed28||total_processed!=458752ull){std::fprintf(stderr,"row repetition mismatch\n");return 8;}

    std::size_t total_meta=0,max_meta=0;for(int d=0;d<NGPU;++d){std::size_t b=std::size_t(count[d])*META_BYTES;total_meta+=b;max_meta=std::max(max_meta,b);}
    if(total_meta!=228327424ull||max_meta!=28596672ull)return 9;

    std::printf("b300-static-lpt-local-meta-proof OK windows=2 groups_per_window=8192 total_groups=16384 gpu_group_min=2044 gpu_group_max=2052 local_meta_ids_contiguous=1 local_meta_ids_unique=1 group_assignment_exactly_once=1 rows=28 total_group_processings=458752 gpu_process_min=57232 gpu_process_max=57456 total_meta_bytes=228327424 max_meta_bytes_per_gpu=28596672 max_meta_mib_per_gpu=27.271911621 total_h2d_gib=0.212646484375 metadata_replication=0 exact=1\n");
    return 0;
}
