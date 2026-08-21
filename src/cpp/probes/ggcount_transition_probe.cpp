#include <cstdint>
#define main ggcount_original_main
#include "ggcount_public.cpp"
#undef main
#include <iostream>
#include <vector>

static void clearpc(PathCounter<uint64_t>& pc){
    for(Code i=0;i<pc.mc.codeSize();++i) pc.value[i]=0;
    for(Code i=0;i<pc.wc.codeSize();++i) pc.deferred[i]=0;
}

int main(){
    msg=NONE; modulus=0;
    int width=5;
    PathCounter<uint64_t> pc(width,width,false,false);
    // Probe p=3 => j=width-p-1 = 1.
    int p=1, j=width-p-1;
    std::cout<<"width="<<width<<" p="<<p<<" main="<<pc.mc.codeSize()<<" blocked="<<pc.wc.codeSize()<<"\n";
    for(Code bi=0;bi<pc.mc.codeSizeL();++bi){
        auto const& b=pc.mc.codeTable(bi);
        for(Code k=0;k<b.size;++k){
            Mate m=b.mateL|b.mateR[k];
            Code src=b.base+k;
            clearpc(pc); pc.value[src]=1; pc.update(j,false);
            std::cout<<"M "<<src<<" "<<m<<" pair="<<m.getPair(p)<<" ->";
            for(Code x=0;x<pc.mc.codeSize();++x) if(pc.value[x]) std::cout<<" M"<<x<<":"<<pc.value[x];
            for(Code x=0;x<pc.wc.codeSize();++x) if(pc.deferred[x]) std::cout<<" D"<<x<<":"<<pc.deferred[x];
            std::cout<<"\n";
        }
    }
    std::cout<<"BLOCKED\n";
    for(Code src=0;src<pc.wc.codeSize();++src){
        clearpc(pc); pc.deferred[src]=1; pc.update(j,false);
        std::cout<<"D "<<src<<" ->";
        for(Code x=0;x<pc.mc.codeSize();++x) if(pc.value[x]) std::cout<<" M"<<x<<":"<<pc.value[x];
        for(Code x=0;x<pc.wc.codeSize();++x) if(pc.deferred[x]) std::cout<<" D"<<x<<":"<<pc.deferred[x];
        std::cout<<"\n";
    }
}
