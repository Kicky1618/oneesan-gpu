#define main gridfp_rankformula_plan_main_unused
#include "gridfp_rankformula_plan.cpp"
#undef main

static uint64_t choose_u64(int n,int k){
    if(k<0||k>n)return 0; if(k>n-k)k=n-k; uint64_t z=1;
    for(int i=1;i<=k;++i)z=z*uint64_t(n-k+i)/uint64_t(i);
    return z;
}

static uint32_t lpattern(int n,int h,uint32_t local){
    uint32_t lp=0; int s=h,rem=n;
    for(int ord=0;ord<n;++ord){
        const uint32_t rc=s>0?ballot_suffix(rem-1,s-1):0u;
        if(s>0&&local<rc)--s;
        else{if(local<rc)return INVALID;local-=rc;lp|=1u<<ord;++s;}
        --rem;
    }
    return s==0&&local==0?lp:INVALID;
}

int main(){
    uint64_t weighted_codes=0,total_calls=0,total_eager=0,total_lazy=0;
    std::array<uint64_t,26> eager{},lazy{};
    for(int n=0;n<=L;++n){
        const uint64_t weight=choose_u64(L,n);
        for(int h=0;h<16;++h){
            const uint32_t cnt=ballot_suffix(n,h);
            for(uint32_t local=0;local<cnt;++local){
                const uint32_t lp=lpattern(n,h,local); if(lp==INVALID)return 2;
                weighted_codes+=weight;
                for(uint32_t depth=1;depth<=25;++depth){
                    uint32_t state=depth;
                    uint64_t e=0,l=0;
                    for(int ord=0;ord<n;++ord){
                        if((lp>>ord)&1u){
                            ++e;
                            if(state==1u)++l;
                            ++state;
                        }else{
                            if(state==1u)break;
                            --state;
                        }
                    }
                    eager[depth]+=e*weight;
                    lazy[depth]+=l*weight;
                    ++total_calls;
                }
            }
        }
    }
    for(uint32_t d=1;d<=25;++d){total_eager+=eager[d];total_lazy+=lazy[d];}
    if(weighted_codes!=1201917ull||total_calls!=30047925ull||
       total_eager!=88482757ull||total_lazy!=2492769ull||
       eager[1]!=1497681ull||lazy[1]!=891345ull||
       eager[13]!=3720805ull||lazy[13]!=1ull||lazy[14]!=0ull)return 3;
    std::cout<<"gridfp-rankformula-abstract-lazy-load OK"
             <<" production_codes="<<weighted_codes
             <<" depths=25 calls="<<total_calls
             <<" eager_source_loads="<<total_eager
             <<" lazy_source_loads="<<total_lazy
             <<" removed_source_loads="<<(total_eager-total_lazy)
             <<" retained_fraction="<<double(total_lazy)/double(total_eager)
             <<" reduction_fraction="<<1.0-double(total_lazy)/double(total_eager)
             <<" depth1_eager="<<eager[1]<<" depth1_lazy="<<lazy[1]
             <<" depth13_eager="<<eager[13]<<" depth13_lazy="<<lazy[13]
             <<" depth14_lazy="<<lazy[14]
             <<" source_load_only_on_state1=1"
             <<" uniform_depth_model=1\n";
    return 0;
}
