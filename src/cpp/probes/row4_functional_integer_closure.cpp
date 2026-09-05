#define main rowr_macro_embedded_main
#include "rowr_column_macro_wfa.cpp"
#undef main
#include <array>
#include <deque>
#include <set>
#include <sstream>

static bool enc4i(std::string const&t,Key&out){
 State s;s.deg.assign(4,0);s.comp.assign(4,0);s.status.assign(1,0);int bal=0;std::vector<std::pair<int,int>>op;
 for(int i=0;i<4;++i){char c=t[i];if(c=='T'){if(bal||!op.empty())return false;int q=s.status.size();s.status.push_back(0);s.deg[i]=1;s.comp[i]=q;s.stack.push_back(q);continue;}if(c=='N')continue;int st=c=='U'?1:-1;bool opens=bal==0||(bal>0&&st>0)||(bal<0&&st<0);if(opens){op.push_back({i,st});bal+=st;}else{if(op.empty())return false;auto[j,sg]=op.back();op.pop_back();if(sg!=-st)return false;int q=s.status.size();s.status.push_back(sg<0);s.deg[j]=s.deg[i]=1;s.comp[j]=s.comp[i]=q;bal+=st;}}
 if(bal||!op.empty())return false;out=key(s);return true;
}
static void rec4i(int p,int nt,int b,std::string&w,std::vector<std::pair<Key,std::string>>&o){if(p==4){if(!nt&&!b){Key k;if(enc4i(w,k))o.push_back({k,w});}return;}if(nt>4-p)return;w[p]='N';rec4i(p+1,nt,b,w,o);w[p]='U';rec4i(p+1,nt,b+1,w,o);w[p]='D';rec4i(p+1,nt,b-1,w,o);if(nt&&b==0){w[p]='T';rec4i(p+1,nt-1,0,w,o);}}
static bool hardi(std::string const&w){int b=0,n=0;auto fl=[&](){bool q=n>=2;b=n=0;return q;};for(char c:w){if(c=='T'){if(fl())return true;continue;}if(c=='N')continue;int d=c=='U'?1:-1;if(b==0&&d<0)++n;b+=d;}return fl();}
static Key special(int mode){State s;s.deg.assign(4,0);s.comp.assign(4,0);s.status.assign(1,0);auto add=[&](int a,int b,int neg){int q=s.status.size();s.status.push_back(neg);s.deg[a]=s.deg[b]=1;s.comp[a]=s.comp[b]=q;};if(mode==0){add(0,1,1);add(2,3,1);}else if(mode==1){add(0,3,1);add(1,2,0);}else add(0,3,1);return key(s);}
int main(){Macro M(4);std::unordered_map<Key,int,KH>id;std::vector<Key>st;std::deque<Key>q;auto add=[&](Key const&k){auto it=id.find(k);if(it!=id.end())return;int z=st.size();id[k]=z;st.push_back(k);q.push_back(k);};for(auto&o:M.step(M.empty_input(),true))add(o.k);while(!q.empty()){auto k=q.front();q.pop_front();for(auto&o:M.step(k,false))add(o.k);}int n=st.size();std::array<std::vector<std::vector<std::pair<int,uint64_t>>>,3>T;for(auto&x:T)x.resize(n);for(int i=0;i<n;++i)for(auto&o:M.step(st[i],false))T[o.sym][i].push_back({id.at(o.k),o.mult});
 std::vector<std::pair<Key,std::string>>can;std::string w(4,'N');for(int h=0;h<=4;++h)rec4i(0,h,0,w,can);std::vector<int>good;for(auto&[k,s]:can)if(!hardi(s)){auto it=id.find(k);if(it==id.end()){std::cerr<<"missing good "<<s<<"\n";return 2;}good.push_back(it->second);}std::array<int,2>sp{ id.at(special(0)),id.at(special(1))}; int dnnu=id.at(special(2));
 std::set<int>gset(good.begin(),good.end()),sset(sp.begin(),sp.end());std::cout<<"states="<<n<<" good="<<good.size()<<" special="<<sp[0]<<","<<sp[1]<<" dnnu_good="<<dnnu<<" good_has_dnnu="<<gset.count(dnnu)<<"\n";size_t checks=0;uint64_t maxc=0;std::array<size_t,3>nzBySym{};
 auto check=[&](std::vector<uint64_t>const&f,std::string name){for(int a=0;a<3;++a){std::vector<unsigned long long>y(n);for(int i=0;i<n;++i){unsigned long long z=0;for(auto[j,c]:T[a][i])z+=c*f[j];y[i]=z;}auto c=y[sp[0]];if(y[sp[1]]!=c){std::cerr<<"special disagree basis="<<name<<" a="<<a<<" vals="<<y[sp[0]]<<","<<y[sp[1]]<<"\n";return false;}for(int i=0;i<n;++i)if(!gset.count(i)&&!sset.count(i)&&y[i]){std::cerr<<"outside support basis="<<name<<" a="<<a<<" id="<<i<<" val="<<y[i]<<"\n";return false;}for(int gi:good){maxc=std::max<uint64_t>(maxc,y[gi]);nzBySym[a]+=y[gi]!=0;}maxc=std::max<uint64_t>(maxc,c);++checks;}return true;};
 for(size_t z=0;z<good.size();++z){std::vector<uint64_t>f(n);f[good[z]]=1;if(!check(f,"g"+std::to_string(z)))return 3;}std::vector<uint64_t>ell(n);for(int x:sp)ell[x]=1;if(!check(ell,"ell"))return 4;
 std::cout<<"integer_closure_exact=1 checks="<<checks<<" max_coeff="<<maxc<<" good_target_nz_by_sym="<<nzBySym[0]<<","<<nzBySym[1]<<","<<nzBySym[2]<<"\n";
}
