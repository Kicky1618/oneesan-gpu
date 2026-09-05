#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

static uint32_t flip_high(uint32_t code,int K,int depth){
    int s=depth;
    for(int pos=0;pos<K;++pos){
        uint32_t v=(code>>(2*pos))&3u;
        if(v==2u){if(--s==0){uint32_t z=3u<<(2*pos);return(code&~z)|(1u<<(2*pos));}}
        else if(v==1u)++s;
    }
    return 0xffffffffu;
}
static uint32_t flip_low(uint32_t code,int K,int depth){
    int s=depth;
    for(int pos=K-1;pos>=0;--pos){
        uint32_t v=(code>>(2*pos))&3u;
        if(v==2u)++s;
        else if(v==1u){if(--s==0){uint32_t z=3u<<(2*pos);return(code&~z)|(2u<<(2*pos));}}
    }
    return 0xffffffffu;
}
static void inv_high(uint32_t dest,int K,int depth,std::vector<uint32_t>&out){
    out.clear();int s=depth;
    for(int pos=0;pos<K;++pos){
        uint32_t v=(dest>>(2*pos))&3u;
        if(v==2u){if(s==1)break;--s;}
        else if(v==1u){
            if(s==1){uint32_t z=3u<<(2*pos);out.push_back((dest&~z)|(2u<<(2*pos)));}
            ++s;
        }
    }
}
static void inv_low(uint32_t dest,int K,int depth,std::vector<uint32_t>&out){
    out.clear();int s=depth;
    for(int pos=K-1;pos>=0;--pos){
        uint32_t v=(dest>>(2*pos))&3u;
        if(v==1u){if(s==1)break;--s;}
        else if(v==2u){
            if(s==1){uint32_t z=3u<<(2*pos);out.push_back((dest&~z)|(1u<<(2*pos)));}
            ++s;
        }
    }
}
static uint32_t code_from_key(uint32_t key,int K){
    uint32_t c=0;
    for(int p=0;p<K;++p){uint32_t d=key%3u;key/=3u;uint32_t v=d==1u?1u:(d==2u?2u:0u);c|=v<<(2*p);}
    return c;
}
static bool check(int K){
    uint32_t n=1;for(int i=0;i<K;++i)n*=3u;
    std::vector<uint32_t> inv;int maxh=0,maxl=0;
    for(uint32_t key=0;key<n;++key){
        uint32_t src=code_from_key(key,K);
        for(int d=1;d<=K;++d){
            uint32_t z=flip_high(src,K,d);
            if(z!=0xffffffffu){inv_high(z,K,d,inv);if(std::find(inv.begin(),inv.end(),src)==inv.end())return false;}
            z=flip_low(src,K,d);
            if(z!=0xffffffffu){inv_low(z,K,d,inv);if(std::find(inv.begin(),inv.end(),src)==inv.end())return false;}
        }
    }
    for(uint32_t key=0;key<n;++key){
        uint32_t dest=code_from_key(key,K);
        for(int d=1;d<=K;++d){
            inv_high(dest,K,d,inv);maxh=std::max(maxh,int(inv.size()));for(uint32_t s:inv)if(flip_high(s,K,d)!=dest)return false;
            inv_low(dest,K,d,inv);maxl=std::max(maxl,int(inv.size()));for(uint32_t s:inv)if(flip_low(s,K,d)!=dest)return false;
        }
    }
    std::cout<<"K="<<K<<" max_inverse_high="<<maxh<<" max_inverse_low="<<maxl<<'\n';
    return true;
}
int main(){
    for(int K:{4,10,13,14})if(!check(K))return 1;
    std::cout<<"cross-inverse-selftest OK\n";
    return 0;
}
