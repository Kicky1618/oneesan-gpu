// Exercise the production arena/graph lifetime contract with real CUDA graphs.
#define main oneesan_solver_main
#include "../src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_factorized_reverse2_row6crt20_batch.cu"
#undef main

__global__ void graph_lifetime_write(Count* a, Count* e) { a[0]=123; e[0]=456; }

static void cache_graph(DeviceCtx& c, const DeviceCtx::GraphKey& key) {
    ck(cudaStreamBeginCapture(c.sMain,cudaStreamCaptureModeThreadLocal),"test begin");
    graph_lifetime_write<<<1,1,0,c.sMain>>>(c.dA,c.dE);
    cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
    ck(cudaStreamEndCapture(c.sMain,&graph),"test end");
    ck(cudaGraphInstantiate(&exec,graph,0),"test instantiate");
    ck(cudaGraphDestroy(graph),"test definition destroy");
    c.transition_graphs.emplace(key,exec);
}
static void replay(DeviceCtx& c,const DeviceCtx::GraphKey& key) {
    ck(cudaMemsetAsync(c.dA,0,sizeof(Count),c.sMain),"test clear A");
    ck(cudaMemsetAsync(c.dE,0,sizeof(Count),c.sMain),"test clear E");
    ck(cudaGraphLaunch(c.transition_graphs.at(key),c.sMain),"test replay");
    ck(cudaStreamSynchronize(c.sMain),"test sync");
    Count a=0,e=0;
    ck(cudaMemcpy(&a,c.dA,sizeof(a),cudaMemcpyDeviceToHost),"test read A");
    ck(cudaMemcpy(&e,c.dE,sizeof(e),cudaMemcpyDeviceToHost),"test read E");
    if(a!=123||e!=456)throw std::runtime_error("stale graph arena addresses");
}
int main() {
    DeviceCtx c;Count* nulls[MAXGPU]{};c.init(0,4294967291u,nulls,nulls,1,1,1);
    // Sizes cross the arena's 256-byte alignment, forcing a real allocation growth.
    DeviceCtx::GraphKey small{128,64,9,1,256,true,true,false,true,false,0};
    DeviceCtx::GraphKey large{1024,512,9,1,256,true,true,false,true,false,0};
    c.ensure(128,64,true,0,0,true);cache_graph(c,small);replay(c,small);
    c.ensure(128,64,true,0,0,true);
    if(c.transition_graphs.size()!=1||c.graph_evictions) return 1;
    // Grow while an old graph is still queued, as in pipelined execution.
    ck(cudaGraphLaunch(c.transition_graphs.at(small),c.sMain),"test pending replay");
    c.ensure(1024,512,true,0,0,true);
    if(!c.transition_graphs.empty()||c.graph_evictions!=1) return 2;
    cache_graph(c,large);replay(c,large);
    c.ensure(128,64,true,0,0,true);
    if(c.transition_graphs.size()!=1||c.graph_evictions!=1) return 3;
    c.ensure(1024,512,true,0,0,true);replay(c,large);
    c.destroy();
    std::cout<<"PASS graph eviction on growth and replay after arena layout changes\n";
}
