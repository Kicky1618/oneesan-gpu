#include "../../common/gridfp_closure_inverse.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace oneesan::gridfp;

static void enum_words(int len,int pos,MateID cur,std::vector<MateID>&out){
    if(pos==len){out.push_back(cur);return;}
    for(MateValue v:{N,R,L})enum_words(len,pos+1,mset(cur,pos,v),out);
}

int main(){
    uint64_t checked=0;
    for(int len=2;len<=10;++len){
        std::vector<MateID> words;enum_words(len,0,0,words);
        for(MateID d:words){
            for(int p=1;p<len;++p){
                MateID a[32]{},b[32]{};int na=ordinary_closure_preimages_partial<32>(d,len,p,a),nb=ordinary_closure_preimages_partial_reverse<32>(d,len,p,b);
                std::vector<MateID> va(a,a+na),vb(b,b+nb);std::sort(va.begin(),va.end());std::sort(vb.begin(),vb.end());
                if(va!=vb){std::cerr<<"closure inverse reflection mismatch len="<<len<<" p="<<p<<" dest="<<d<<" forward="<<na<<" reverse="<<nb<<'\n';return 1;}
                ++checked;
            }
        }
    }
    std::cout<<"closure-inverse-reflection OK checked="<<checked<<" reverse_equals_forward=1\n";return 0;
}
