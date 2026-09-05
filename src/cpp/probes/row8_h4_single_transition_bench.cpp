#define ROW8_RAW_PREFIX_NO_MAIN 1
#include "row8_raw_prefix_vector.cpp"
#include <chrono>
#include <iostream>
int main(){MODP=1000000007u;Vec all;int col=0;if(!load_ck("work/formal-probes/raw_wfa_r8.ck",8,col,all))return 2;std::vector<Packed> h4;for(auto const&p:all)if(unpack(p).sp==4)h4.push_back(p);std::cerr<<"h4="<<h4.size()<<" col="<<col<<"\n";for(int n:{1,10,100}){auto t=std::chrono::steady_clock::now();size_t out=0;for(int i=0;i<n;++i){WVec v{{h4[(size_t)i*h4.size()/n],1}};auto z=wcolumn(std::move(v),8,false,0);out+=z.size();}double s=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();std::cout<<"n="<<n<<" sec="<<s<<" per_ms="<<s*1000/n<<" avgout="<<(double)out/n<<"\n";}
}
