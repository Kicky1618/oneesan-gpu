#include <cuda_runtime.h>
#include <cstdint>
__global__ void k(unsigned short* p){
    unsigned short old=*p;
    (void)atomicCAS(p,old,(unsigned short)(old+1));
}
int main(){return 0;}
