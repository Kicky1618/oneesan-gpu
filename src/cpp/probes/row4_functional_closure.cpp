#define main rowr_macro_embedded_main
#include "rowr_column_macro_wfa.cpp"
#undef main
#include <array>
#include <deque>
#include <iomanip>
#include <memory>
#include <set>
#include <sstream>

static constexpr uint32_t P4=1000000007u;
static uint32_t inv4(uint32_t a){uint64_t r=1,x=a,e=P4-2;while(e){if(e&1)r=(__uint128_t)r*x%P4;x=(__uint128_t)x*x%P4;e>>=1;}return (uint32_t)r;}

static bool encode4(std::string const&t,Key&out){
    if(t.size()!=4)return false;int h=0,bal=0;for(char c:t){if(c=='T'){if(bal)return false;++h;}else if(c=='U')++bal;else if(c=='D')--bal;else if(c!='N')return false;}if(bal)return false;
    State s;s.deg.assign(4,0);s.comp.assign(4,0);s.status.assign(1,0);s.stack.clear();
    std::vector<int>tp;for(int i=0;i<4;++i)if(t[i]=='T')tp.push_back(i);
    int next=1;for(int i:tp){s.status.push_back(0);s.deg[i]=1;s.comp[i]=next;s.stack.push_back(next);++next;}
    std::vector<std::pair<int,int>>op;bal=0;
    for(int i=0;i<4;++i){char c=t[i];if(c=='T'){if(bal||!op.empty())return false;continue;}if(c=='N')continue;int st=c=='U'?1:-1;bool opens=bal==0||(bal>0&&st>0)||(bal<0&&st<0);if(opens){op.push_back({i,st});bal+=st;}else{if(op.empty())return false;auto [j,sg]=op.back();op.pop_back();if(sg!=-st)return false;s.status.push_back(sg<0?1:0);s.deg[j]=s.deg[i]=1;s.comp[j]=s.comp[i]=next;++next;bal+=st;}}
    if(bal||!op.empty())return false;out=key(s);return true;
}
static void rec4(int pos,int nt,int bal,std::string&w,std::vector<std::pair<Key,std::string>>&out){
    if(pos==4){if(!nt&&!bal){Key k;if(encode4(w,k))out.push_back({k,w});}return;}if(nt>4-pos)return;
    w[pos]='N';rec4(pos+1,nt,bal,w,out);w[pos]='U';rec4(pos+1,nt,bal+1,w,out);w[pos]='D';rec4(pos+1,nt,bal-1,w,out);if(nt&&bal==0){w[pos]='T';rec4(pos+1,nt-1,0,w,out);}
}
static bool hard4(std::string const&w){int bal=0,neg=0;auto flush=[&](){bool q=neg>=2;bal=neg=0;return q;};for(char c:w){if(c=='T'){if(flush())return true;continue;}if(c=='N')continue;int d=c=='U'?1:-1;if(bal==0&&d<0)++neg;bal+=d;}return flush();}
static std::string show_state(Key const&k){State s=dec(k);std::ostringstream o;o<<"deg=";for(auto x:s.deg)o<<(int)x;o<<" comp=";for(auto x:s.comp)o<<(int)x<<',';o<<" stack=";for(auto x:s.stack)o<<(int)x<<',';o<<" status=";for(size_t q=1;q<s.status.size();++q)o<<(int)s.status[q];return o.str();}

struct LB {
    int n;std::vector<std::vector<uint32_t>> b;int rank=0;
    explicit LB(int nn):n(nn),b(nn){}
    bool add(std::vector<uint32_t> x,std::vector<uint32_t>*normalized=nullptr){
        for(int p=0;p<n;++p)if(x[p]){
            if(!b[p].empty()){uint32_t f=x[p];for(int j=p;j<n;++j)x[j]=(x[j]+P4-(__uint128_t)f*b[p][j]%P4)%P4;}
            else{uint32_t iv=inv4(x[p]);for(int j=p;j<n;++j)x[j]=(__uint128_t)x[j]*iv%P4;b[p]=x;++rank;if(normalized)*normalized=std::move(x);return true;}
        }return false;
    }
};
int main(){constexpr int r=4;Macro M(r);std::unordered_map<Key,int,KH> id;std::vector<Key> states;std::deque<Key> q;
    auto addstate=[&](Key const&k){auto it=id.find(k);if(it!=id.end())return it->second;int z=(int)states.size();id.emplace(k,z);states.push_back(k);q.push_back(k);return z;};
    auto first=M.step(M.empty_input(),true);for(auto const&o:first)addstate(o.k);
    while(!q.empty()){Key k=q.front();q.pop_front();for(auto const&o:M.step(k,false))addstate(o.k);}int n=states.size();
    std::array<std::vector<std::vector<std::pair<int,uint32_t>>>,3>T;for(int a=0;a<3;++a)T[a].resize(n);
    for(int i=0;i<n;++i)for(auto const&o:M.step(states[i],false)){int j=id.at(o.k);T[o.sym][i].push_back({j,(uint32_t)(o.mult%P4)});} 
    std::vector<std::pair<Key,std::string>> canon;std::string w(4,'N');for(int h=0;h<=4;++h)rec4(0,h,0,w,canon);std::sort(canon.begin(),canon.end(),[](auto const&a,auto const&b){return a.second<b.second;});
    std::set<Key> uniq;for(auto const&x:canon)uniq.insert(x.first);std::cout<<"raw_states="<<n<<" canonical="<<canon.size()<<" unique="<<uniq.size()<<"\n";
    LB B(n);std::vector<std::string> goodWords;int goodReach=0,hard=0,hardReach=0;
    for(auto const&[k,s]:canon){bool hd=hard4(s);if(hd)++hard;auto it=id.find(k);if(it!=id.end()){if(hd)++hardReach;else{std::vector<uint32_t>e(n);e[it->second]=1;B.add(std::move(e));goodWords.push_back(s);++goodReach;}}else if(!hd)std::cerr<<"GOOD UNREACHABLE "<<s<<"\n";}
    std::cout<<"good_reachable="<<goodReach<<" hard="<<hard<<" hard_reachable="<<hardReach<<" initial_rank="<<B.rank<<"\n";
    struct Gen{std::vector<uint32_t>v;std::string why;};std::deque<Gen>todo;
    // Seed todo with the original 55 unit functionals so we can close under T_a.
    for(auto const&[k,s]:canon)if(!hard4(s)){auto it=id.find(k);if(it!=id.end()){std::vector<uint32_t>e(n);e[it->second]=1;todo.push_back({std::move(e),s});}}
    int added=0;
    while(!todo.empty()){
        Gen g=std::move(todo.front());todo.pop_front();
        for(int a=0;a<3;++a){std::vector<uint32_t>y(n);for(int i=0;i<n;++i){uint64_t sum=0;for(auto [j,c]:T[a][i])if(g.v[j])sum=(sum+(uint64_t)c*g.v[j])%P4;y[i]=sum;}
            std::vector<uint32_t>norm;if(B.add(y,&norm)){++added;size_t nz=0;for(auto x:norm)nz+=x!=0;std::string why=g.why+char('0'+a);std::cout<<"added="<<added<<" rank="<<B.rank<<" via="<<why<<" nz="<<nz<<"\n";
                if(added<=4){int shown=0;for(int i=0;i<n&&shown<80;++i)if(norm[i]){int64_t c=norm[i]<=P4/2?norm[i]:(int64_t)norm[i]-P4;std::cout<<"  c="<<c<<" id="<<i<<" "<<show_state(states[i])<<"\n";++shown;}}
                todo.push_back({std::move(norm),why});
            }
        }
        if(B.rank>80){std::cout<<"closure_exceeded_80\n";break;}
    }
    std::cout<<"closure_rank="<<B.rank<<" added="<<added<<"\n";
}
