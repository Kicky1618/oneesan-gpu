#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void ck(cudaError_t e,const char*where){if(e!=cudaSuccess){std::fprintf(stderr,"%s: %s\n",where,cudaGetErrorString(e));std::exit(2);}}
__global__ void mark_kernel(int*p,int v){if(threadIdx.x==0&&blockIdx.x==0)*p=v;}

int main(int argc,char**argv){
    int requested=argc>1?std::atoi(argv[1]):8,visible=0;ck(cudaGetDeviceCount(&visible),"cudaGetDeviceCount");
    int ng=requested>0?requested:visible;if(ng<2||ng>visible||ng>8){std::fprintf(stderr,"need 2..8 visible GPUs, requested=%d visible=%d\n",ng,visible);return 2;}
    std::vector<int*> flag(ng,nullptr);std::vector<cudaStream_t> stream(ng,nullptr);std::vector<cudaEvent_t> done(ng,nullptr);
    for(int d=0;d<ng;++d){ck(cudaSetDevice(d),"set device init");ck(cudaMalloc(&flag[d],sizeof(int)),"flag alloc");ck(cudaMemset(flag[d],0,sizeof(int)),"flag zero");ck(cudaStreamCreateWithFlags(&stream[d],cudaStreamNonBlocking),"stream create");ck(cudaEventCreateWithFlags(&done[d],cudaEventDisableTiming),"event create");mark_kernel<<<1,1,0,stream[d]>>>(flag[d],d+1);ck(cudaGetLastError(),"mark launch");ck(cudaEventRecord(done[d],stream[d]),"event record");}
    ck(cudaSetDevice(0),"set device zero");
    for(int d=1;d<ng;++d)ck(cudaStreamWaitEvent(stream[0],done[d],0),"cross-device stream wait event");
    ck(cudaStreamSynchronize(stream[0]),"gpu0 join sync");
    bool ok=true;
    for(int d=0;d<ng;++d){int got=0;ck(cudaSetDevice(d),"set device verify");ck(cudaMemcpy(&got,flag[d],sizeof(got),cudaMemcpyDeviceToHost),"flag D2H");if(got!=d+1){std::fprintf(stderr,"flag mismatch gpu=%d got=%d expected=%d\n",d,got,d+1);ok=false;}}
    for(int d=0;d<ng;++d){cudaSetDevice(d);if(done[d])cudaEventDestroy(done[d]);if(stream[d])cudaStreamDestroy(stream[d]);if(flag[d])cudaFree(flag[d]);}
    if(!ok)return 3;
    std::printf("b300-cross-device-event-wait-probe OK gpus=%d foreign_event_waits=%d gpu0_host_syncs=1 exact=1\n",ng,ng-1);
    return 0;
}
