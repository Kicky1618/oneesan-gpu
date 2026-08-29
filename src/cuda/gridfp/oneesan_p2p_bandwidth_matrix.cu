#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

static void ck(cudaError_t e,const char* what){if(e!=cudaSuccess){std::cerr<<what<<": "<<cudaGetErrorString(e)<<'\n';std::exit(200);}}

int main(int argc,char**argv){
    int mib=argc>1?std::atoi(argv[1]):256;
    int reps=argc>2?std::atoi(argv[2]):8;
    if(mib<=0||reps<=0)return 1;
    int visible=0;ck(cudaGetDeviceCount(&visible),"device count");
    int ngpu=std::min(visible,8);if(ngpu<2){std::cerr<<"need at least 2 GPUs\n";return 2;}
    std::size_t bytes=std::size_t(mib)<<20;
    std::vector<void*> buf(ngpu,nullptr);
    for(int d=0;d<ngpu;++d){ck(cudaSetDevice(d),"set alloc device");ck(cudaMalloc(&buf[d],bytes),"alloc probe buffer");ck(cudaMemset(buf[d],d+1,bytes),"init probe buffer");}
    for(int d=0;d<ngpu;++d)for(int s=0;s<ngpu;++s)if(d!=s){int can=0;ck(cudaDeviceCanAccessPeer(&can,d,s),"can peer");if(!can){std::cerr<<"P2P unavailable "<<s<<" -> "<<d<<'\n';return 3;}ck(cudaSetDevice(d),"set peer device");cudaError_t e=cudaDeviceEnablePeerAccess(s,0);if(e==cudaErrorPeerAccessAlreadyEnabled)cudaGetLastError();else ck(e,"enable peer");}

    std::vector<double> gbps(std::size_t(ngpu)*ngpu,0.0);
    for(int d=0;d<ngpu;++d){
        ck(cudaSetDevice(d),"set measure device");cudaStream_t st{};cudaEvent_t a{},b{};ck(cudaStreamCreateWithFlags(&st,cudaStreamNonBlocking),"create stream");ck(cudaEventCreate(&a),"create start event");ck(cudaEventCreate(&b),"create stop event");
        for(int s=0;s<ngpu;++s)if(s!=d){
            ck(cudaMemcpyPeerAsync(buf[d],d,buf[s],s,bytes,st),"warmup peer copy");ck(cudaStreamSynchronize(st),"warmup sync");
            ck(cudaEventRecord(a,st),"record start");for(int r=0;r<reps;++r)ck(cudaMemcpyPeerAsync(buf[d],d,buf[s],s,bytes,st),"measure peer copy");ck(cudaEventRecord(b,st),"record stop");ck(cudaEventSynchronize(b),"stop sync");float ms=0.0f;ck(cudaEventElapsedTime(&ms,a,b),"elapsed");double z=double(bytes)*reps/(double(ms)*1.0e6);gbps[std::size_t(d)*ngpu+s]=z;
        }
        cudaEventDestroy(a);cudaEventDestroy(b);cudaStreamDestroy(st);
    }

    std::cerr<<"P2P bandwidth matrix GB/s, rows=destination cols=source, bytes="<<bytes<<" reps="<<reps<<'\n';
    std::cerr<<std::fixed<<std::setprecision(2);
    for(int d=0;d<ngpu;++d){for(int s=0;s<ngpu;++s){if(s)std::cerr<<' ';std::cerr<<std::setw(9)<<gbps[std::size_t(d)*ngpu+s];}std::cerr<<'\n';}

    std::cout<<"export ONEESAN_P2P_GBPS='"<<std::setprecision(9);
    for(int d=0;d<ngpu;++d)for(int s=0;s<ngpu;++s){if(d||s)std::cout<<',';std::cout<<gbps[std::size_t(d)*ngpu+s];}
    std::cout<<"'\n";

    for(int d=0;d<ngpu;++d){ck(cudaSetDevice(d),"set free device");cudaFree(buf[d]);}
    return 0;
}
