#include <bits/stdc++.h>
using namespace std;

struct S {
    vector<uint8_t> d,c,f;
    bool done=false;
};
struct K { vector<uint8_t> b; bool operator==(K const&o)const{return b==o.b;} };
struct H { size_t operator()(K const&k)const noexcept{ uint64_t h=1469598103934665603ull; for(auto x:k.b){h^=x;h*=1099511628211ull;} return h; } };

static void canon(S& s){
    array<uint8_t,64> mp{}, nf{}; uint8_t nx=1;
    for(size_t i=0;i<s.c.size();++i){ auto q=s.c[i]; if(!q) continue; if(!mp[q]){mp[q]=nx; nf[nx]=s.f[q]; ++nx;} s.c[i]=mp[q]; }
    s.f.assign(nx,0); for(uint8_t i=1;i<nx;++i) s.f[i]=nf[i];
}
static K key(S s){
    canon(s); K k; k.b.push_back(s.done); for(size_t i=0;i<s.d.size();++i){k.b.push_back(s.d[i]);k.b.push_back(s.c[i]);} k.b.push_back(255); for(size_t i=1;i<s.f.size();++i)k.b.push_back(s.f[i]); return k;
}

int main(int argc,char**argv){
    int n=argc>1?stoi(argv[1]):4, w=n+1, V=w*w, st=0, tt=V-1;
    vector<pair<int,int>> e;
    for(int r=0;r<w;++r)for(int c=0;c<w;++c){int u=r*w+c;if(c+1<w)e.push_back({u,u+1});if(r+1<w)e.push_back({u,u+w});}
    vector<int> last(V,-1); for(int i=0;i<(int)e.size();++i){last[e[i].first]=i;last[e[i].second]=i;}
    S z; z.d.assign(V,0); z.c.assign(V,0); z.f.assign(1,0);
    unordered_map<K,unsigned long long,H> cur,nxt; cur[key(z)]=1;
    auto dec=[&](K const&k){ S s; s.done=k.b[0]; s.d.assign(V,0);s.c.assign(V,0);size_t p=1;uint8_t mc=0;for(int i=0;i<V;++i){s.d[i]=k.b[p++];s.c[i]=k.b[p++];mc=max(mc,s.c[i]);}++p;s.f.assign(mc+1,0);for(uint8_t i=1;i<=mc;++i)s.f[i]=k.b[p++];return s;};
    size_t peak=1;
    for(int ei=0;ei<(int)e.size();++ei){
        nxt.clear(); auto [u,v]=e[ei];
        for(auto const& [kk,ways]:cur){ S base=dec(kk);
            for(int take=0;take<2;++take){ S s=base; bool ok=true;
                if(take){
                    if(s.done)continue; uint8_t mu=(u==st||u==tt)?1:2,mv=(v==st||v==tt)?1:2; if(s.d[u]>=mu||s.d[v]>=mv)continue; ++s.d[u];++s.d[v];
                    uint8_t a=s.c[u],b=s.c[v];
                    if(!a&&!b){uint8_t q=s.f.size();s.f.push_back(0);if(u==st||v==st)s.f[q]|=1;if(u==tt||v==tt)s.f[q]|=2;s.c[u]=s.c[v]=q;}
                    else if(!a||!b){uint8_t q=a?a:b;s.c[u]=s.c[v]=q;if(u==st||v==st)s.f[q]|=1;if(u==tt||v==tt)s.f[q]|=2;}
                    else { if(a==b)continue; uint8_t keep=min(a,b),kill=max(a,b);s.f[keep]|=s.f[kill];for(int x=0;x<V;++x)if(s.c[x]==kill)s.c[x]=keep; }
                }
                for(int x:{u,v}) if(last[x]==ei){
                    bool terminal=(x==st||x==tt); if((terminal&&s.d[x]!=1)||(!terminal&&s.d[x]!=0&&s.d[x]!=2)){ok=false;break;}
                    uint8_t q=s.c[x]; s.c[x]=0; if(q){bool alive=false;for(int y=0;y<V;++y)if(s.c[y]==q){alive=true;break;} if(!alive){if(s.f[q]==3)s.done=true;else{ok=false;break;}}}
                }
                if(!ok)continue; canon(s); nxt[key(s)]+=ways;
            }
        }
        cur.swap(nxt); peak=max(peak,cur.size()); if((ei+1)%w==0||ei+1==(int)e.size())cerr<<"edge "<<ei+1<<"/"<<e.size()<<" states="<<cur.size()<<"\n";
    }
    unsigned long long ans=0; for(auto const& [kk,ways]:cur)if(dec(kk).done)ans+=ways;
    cout<<"n="<<n<<" paths="<<ans<<" peak_states="<<peak<<"\n";
}
