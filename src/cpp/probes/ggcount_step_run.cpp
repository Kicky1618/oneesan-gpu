#include <cstdint>
#define main ggcount_original_main
#include "ggcount_public.cpp"
#undef main
#include <iostream>
int main(){
 msg=NONE; modulus=0; int width=5; PathCounter<uint64_t> pc(width,width,false,false);
 for(Code i=0;i<pc.mc.codeSize();++i)pc.value[i]=0;
 for(Code i=0;i<pc.wc.codeSize();++i)pc.deferred[i]=0;
 pc.value[pc.mc.encode(Mate(width-1,R))]=1;
 for(int step=0;step<12;++step){
   int j=step%(width-1), p=width-j-1; pc.update(j,false);
   std::cout<<"STEP "<<step+1<<" p="<<p<<"\nM";
   for(Code i=0;i<pc.mc.codeSize();++i)if(pc.value[i])std::cout<<" "<<i<<":"<<pc.value[i];
   std::cout<<"\nD";
   for(Code i=0;i<pc.wc.codeSize();++i)if(pc.deferred[i])std::cout<<" "<<i<<":"<<pc.deferred[i];
   std::cout<<"\n";
 }
}
