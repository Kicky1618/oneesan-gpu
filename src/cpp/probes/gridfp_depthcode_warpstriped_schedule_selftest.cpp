#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

static bool check_case(int total,int cols,int gx,int gy,int threads){
    if(threads<=0||threads>1024||(threads&31))return false;
    const int nwarps=threads/32;
    std::vector<uint16_t> original(size_t(total)*cols),striped(size_t(total)*cols);
    auto hit=[&](std::vector<uint16_t>&v,int k,int lr){
        if(k<0||k>=total||lr<0||lr>=cols){std::cerr<<"schedule index overflow\n";std::exit(2);}
        ++v[size_t(k)*cols+lr];
    };
    for(int by=0;by<gy;++by)for(int bx=0;bx<gx;++bx)for(int t=0;t<threads;++t)
        for(int k=by;k<total;k+=gy)
            for(int lr=bx*threads+t;lr<cols;lr+=gx*threads)hit(original,k,lr);
    for(int by=0;by<gy;++by)for(int bx=0;bx<gx;++bx)for(int w=0;w<nwarps;++w)for(int lane=0;lane<32;++lane)
        for(int k=by*nwarps+w;k<total;k+=gy*nwarps)
            for(int lr=bx*32+lane;lr<cols;lr+=gx*32)hit(striped,k,lr);
    for(int k=0;k<total;++k)for(int lr=0;lr<cols;++lr){
        auto a=original[size_t(k)*cols+lr],b=striped[size_t(k)*cols+lr];
        if(a!=1||b!=1){std::cerr<<"schedule coverage mismatch total="<<total<<" cols="<<cols<<" gx="<<gx<<" gy="<<gy<<" threads="<<threads<<" k="<<k<<" lr="<<lr<<" original="<<a<<" striped="<<b<<'\n';return false;}
    }
    return original==striped;
}

int main(){
    uint64_t cases=0,cells=0;
    for(int threads: {32,64,96,128,160,192,224,256,512,1024})
        for(int gx: {1,2,3,7,16})for(int gy: {1,2,5,8})
            for(int total: {1,2,7,8,9,31,32,33,67})
                for(int cols: {1,2,31,32,33,63,64,65,257}){
                    if(!check_case(total,cols,gx,gy,threads))return 1;
                    ++cases;cells+=uint64_t(total)*uint64_t(cols);
                }
    std::cout<<"gridfp-depthcode-warpstriped-schedule-selftest OK cases="<<cases<<" cells="<<cells<<" full_warp_required=1 exact_orbit_column_cover=1 no_duplicate_work_items=1\n";
    return 0;
}
