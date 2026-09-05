#include <bits/stdc++.h>
using namespace std;
static void genY(int h,int x,int pos,int prevY,int y, vector<uint16_t>&out){
    if(pos==h){out.push_back((uint16_t)y);return;}
    int xp=(x>>(pos-1))&1, xc=(x>>pos)&1;
    for(int yc=0;yc<2;++yc){
        bool bad=(xp!=xc)&&(prevY!=yc)&&(xp!=prevY);
        if(!bad) genY(h,x,pos+1,yc,y|(yc<<pos),out);
    }
}
int main(int argc,char**argv){
    int h=argc>1?atoi(argv[1]):14, iters=argc>2?atoi(argv[2]):30; int N=1<<h;
    vector<uint64_t> off(N+1); vector<uint16_t> ed; ed.reserve((size_t)N*4000);
    for(int x=0;x<N;++x){off[x]=ed.size(); genY(h,x,1,0,0,ed); genY(h,x,1,1,1,ed);} off[N]=ed.size();
    cerr<<"h="<<h<<" N="<<N<<" edges="<<ed.size()<<" avg="<<double(ed.size())/N<<"\n";
    vector<long double> v(N,1),w(N);
    for(int it=0;it<iters;++it){
        long double mx=0;
        #pragma omp parallel for reduction(max:mx)
        for(int x=0;x<N;++x){long double s=0;for(uint64_t q=off[x];q<off[x+1];++q)s+=v[ed[q]];w[x]=s;mx=max(mx,s);}
        #pragma omp parallel for
        for(int x=0;x<N;++x)v[x]=w[x]/mx;
        if(it%5==4) cerr<<"iter="<<it+1<<" scale="<<(double)mx<<"\n";
    }
    long double mn=*min_element(v.begin(),v.end()),mx=*max_element(v.begin(),v.end());
    const long double SCALE=1e12L/mx;
    vector<uint64_t> qv(N),kv(N); uint64_t minv=ULLONG_MAX,maxv=0; unsigned __int128 sumv=0;
    for(int i=0;i<N;++i){qv[i]=max<uint64_t>(1,(uint64_t)floor(v[i]*SCALE+0.5L));minv=min(minv,qv[i]);maxv=max(maxv,qv[i]);sumv+=qv[i];}
    #pragma omp parallel for
    for(int x=0;x<N;++x){unsigned long long s=0;for(uint64_t q=off[x];q<off[x+1];++q)s+=qv[ed[q]];kv[x]=s;}
    uint64_t besta=0,bestb=1; int besti=-1;
    for(int i=0;i<N;++i){ // compare kv[i]/qv[i]
        if((unsigned __int128)kv[i]*bestb > (unsigned __int128)besta*qv[i]){besta=kv[i];bestb=qv[i];besti=i;}
    }
    auto u128=[](unsigned __int128 x){if(!x)return string("0");string s;while(x){s.push_back('0'+x%10);x/=10;}reverse(s.begin(),s.end());return s;};
    cout<<"h "<<h<<" edges "<<ed.size()<<" a "<<besta<<" b "<<bestb<<" minv "<<minv<<" sumv "<<u128(sumv)<<" arg "<<besti<<" maxv "<<maxv<<"\n";
}
