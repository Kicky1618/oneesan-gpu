#define ROW8_CANONICAL_TRIE_NO_MAIN 1
#include "row8_canonical_trie_verify.cpp"
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <vector>
static constexpr uint32_t Q=1000000007u;
struct IH{char magic[8];uint32_t ver,h,rows,states;uint64_t maxabs;};
struct EH{char magic[8];uint32_t ver,mod,h,rows,states;};
static uint32_t invq(uint32_t a){uint64_t e=Q-2,r=1,x=a;while(e){if(e&1)r=r*x%Q;x=x*x%Q;e>>=1;}return r;}
static uint32_t maskof(Packed p){State s=unpack(p);uint32_t m=0;for(int i=0;i<8;++i)if(s.deg[i])m|=1u<<i;return m;}
static Packed compressp(Packed p){State s=unpack(p),t{};t.sp=s.sp;t.stack=s.stack;t.status=s.status;t.ns=s.ns;int n=0;for(int i=0;i<8;++i)if(s.deg[i]){t.deg[n]=s.deg[i];t.comp[n]=s.comp[i];++n;}t.n=n;return pack(t);}
static std::vector<std::vector<uint32_t>> rref(std::vector<std::vector<uint32_t>> B){int R=B.size(),C=R?B[0].size():0,rk=0;for(int c=0;c<C&&rk<R;++c){int p=rk;while(p<R&&!B[p][c])++p;if(p==R)continue;std::swap(B[p],B[rk]);uint32_t iv=invq(B[rk][c]);for(int j=c;j<C;++j)B[rk][j]=(uint64_t)B[rk][j]*iv%Q;for(int i=0;i<R;++i)if(i!=rk&&B[i][c]){uint32_t f=B[i][c];for(int j=c;j<C;++j)B[i][j]=(B[i][j]+Q-(uint64_t)f*B[rk][j]%Q)%Q;}++rk;}B.resize(rk);return B;}
struct Block{uint32_t mask;std::vector<int> idx;std::vector<Packed> keys;std::vector<std::vector<uint32_t>> rr;std::vector<int> piv;int off=0;};
struct Space{int h=0,dim=0;std::vector<Packed> states;std::vector<Block> blocks;std::vector<int> stateBlock,stateLocal;};
static std::vector<std::vector<uint32_t>> loadA01(int h,size_t states){std::string p="work/formal-probes/dual-basis/Phi_h"+std::to_string(h)+"_integer.bin";std::ifstream in(p,std::ios::binary);IH hh{};in.read((char*)&hh,sizeof(hh));std::vector<std::vector<uint32_t>>A(hh.rows,std::vector<uint32_t>(states));for(int r=0;r<(int)hh.rows;++r)for(int c=0;c<(int)states;++c){int64_t x;in.read((char*)&x,8);x%=Q;if(x<0)x+=Q;A[r][c]=x;}return A;}
static std::vector<std::vector<uint32_t>> loadA2(std::vector<Packed>const&H){auto ws=basis(2);std::set<int>bad;{std::ifstream b("work/formal-probes/canonical-matrix/h2_actual_bad_indices.txt");int x;while(b>>x)bad.insert(x);}std::vector<std::vector<uint32_t>>A(1428,std::vector<uint32_t>(H.size()));int r=0;for(int j=0;j<(int)ws.size();++j)if(!bad.count(j)){int i=std::lower_bound(H.begin(),H.end(),ws[j])-H.begin();A[r++][i]=1;}if(r!=1308)throw std::runtime_error("good1308");std::ifstream in("work/formal-probes/dual-basis/Phi_h2_extra_mod1000000007.bin",std::ios::binary);EH hh{};in.read((char*)&hh,sizeof(hh));for(int q=0;q<120;++q)in.read((char*)A[1308+q].data(),H.size()*4);return A;}
static Space makeSpace(int h,std::vector<Packed> H,std::vector<std::vector<uint32_t>>const&A){Space S;S.h=h;S.states=std::move(H);S.stateBlock.assign(S.states.size(),-1);S.stateLocal.assign(S.states.size(),-1);std::map<uint32_t,std::vector<int>> g;for(int i=0;i<(int)S.states.size();++i)g[maskof(S.states[i])].push_back(i);std::map<int,std::pair<std::vector<Packed>,std::vector<std::vector<uint32_t>>>> templ;int off=0;for(auto const&[m,ids]:g){int j=__builtin_popcount(m);std::vector<std::pair<Packed,int>>z;for(int i:ids)z.push_back({compressp(S.states[i]),i});std::sort(z.begin(),z.end(),[](auto&a,auto&b){return a.first<b.first;});Block b;b.mask=m;b.off=off;for(auto&x:z){b.keys.push_back(x.first);b.idx.push_back(x.second);}if(!templ.count(j)){std::vector<std::vector<uint32_t>>B(A.size(),std::vector<uint32_t>(b.idx.size()));for(int r=0;r<(int)A.size();++r)for(int c=0;c<(int)b.idx.size();++c)B[r][c]=A[r][b.idx[c]];auto rr=rref(std::move(B));templ[j]={b.keys,rr};}b.rr=templ[j].second;if(b.keys!=templ[j].first)throw std::runtime_error("core keys mismatch");for(auto const&row:b.rr){int p=-1;for(int c=0;c<(int)row.size();++c)if(row[c]){p=c;break;}if(p<0||row[p]!=1)throw std::runtime_error("bad rref");b.piv.push_back(p);}for(int c=0;c<(int)b.idx.size();++c){S.stateBlock[b.idx[c]]=S.blocks.size();S.stateLocal[b.idx[c]]=c;}off+=b.rr.size();S.blocks.push_back(std::move(b));}S.dim=off;return S;}
static void runTrans(Space const&S,Space const&T,int a){std::vector<std::vector<std::pair<int,uint32_t>>> eval(T.states.size());for(int bi=0;bi<(int)T.blocks.size();++bi){auto const&b=T.blocks[bi];for(int c=0;c<(int)b.idx.size();++c)for(int r=0;r<(int)b.rr.size();++r)if(b.rr[r][c])eval[b.idx[c]].push_back({b.off+r,b.rr[r][c]});}
 std::vector<std::vector<uint32_t>>Y(T.dim,std::vector<uint32_t>(S.states.size()));size_t rawedges=0;for(int si=0;si<(int)S.states.size();++si){WVec v{{S.states[si],1}};auto z=wcolumn(std::move(v),8,false,a);rawedges+=z.size();for(auto const&e:z){int ti=std::lower_bound(T.states.begin(),T.states.end(),e.p)-T.states.begin();if(ti==(int)T.states.size()||!(T.states[ti]==e.p))throw std::runtime_error("target");for(auto[k,x]:eval[ti])Y[k][si]=(Y[k][si]+(uint64_t)e.v*x)%Q;}}
 size_t nz=0,bad=0;std::map<int,int>rowNz;std::map<int64_t,size_t>coefHist;for(int k=0;k<T.dim;++k){int rn=0;for(auto const&b:S.blocks){std::vector<uint32_t>y(b.idx.size());for(int c=0;c<(int)b.idx.size();++c)y[c]=Y[k][b.idx[c]];for(int r=0;r<(int)b.rr.size();++r){int p=b.piv[r];uint32_t c=y[p];if(!c)continue;++rn;++nz;int64_t sv=c<=Q/2?(int64_t)c:(int64_t)c-Q;++coefHist[sv];for(int q=p;q<(int)y.size();++q)if(b.rr[r][q])y[q]=(y[q]+Q-(uint64_t)c*b.rr[r][q]%Q)%Q;}for(auto x:y)if(x){++bad;break;}}++rowNz[rn];}
 std::cout<<"STRUCT h="<<S.h<<" a="<<a<<" h2="<<T.h<<" dims="<<S.dim<<"x"<<T.dim<<" rawedges="<<rawedges<<" q_nz="<<nz<<" density="<<(double)nz/((size_t)S.dim*T.dim)<<" bad="<<bad<<" rowNz";for(auto[x,c]:rowNz)std::cout<<' '<<x<<':'<<c;std::cout<<" coeff";int sh=0;for(auto[v,c]:coefHist)if(sh++<20)std::cout<<' '<<v<<':'<<c;std::cout<<"\n";if(bad)throw std::runtime_error("span");}

static size_t runTransFast(Space const&S,Space const&T,int a){
 size_t nz=0;std::map<int64_t,size_t> ch;
 for(auto const&sb:S.blocks)for(int sr=0;sr<(int)sb.rr.size();++sr){
   int si=sb.idx[sb.piv[sr]], srow=sb.off+sr;
   std::vector<uint32_t> q(T.dim);
   WVec v{{S.states[si],1}};auto z=wcolumn(std::move(v),8,false,a);
   for(auto const&e:z){int ti=std::lower_bound(T.states.begin(),T.states.end(),e.p)-T.states.begin();if(ti==(int)T.states.size()||!(T.states[ti]==e.p))throw std::runtime_error("fast target");int bi=T.stateBlock[ti],lc=T.stateLocal[ti];auto const&tb=T.blocks[bi];for(int tr=0;tr<(int)tb.rr.size();++tr)if(tb.rr[tr][lc])q[tb.off+tr]=(q[tb.off+tr]+(uint64_t)e.v*tb.rr[tr][lc])%Q;}
   for(auto x:q)if(x){++nz;int64_t y=x<=Q/2?(int64_t)x:(int64_t)x-Q;++ch[y];}
 }
 std::cout<<"FAST h="<<S.h<<" a="<<a<<" h2="<<T.h<<" q_nz="<<nz<<" density="<<(double)nz/((size_t)S.dim*T.dim)<<" coeff";for(auto[v,c]:ch)std::cout<<' '<<v<<':'<<c;std::cout<<"\n";return nz;
}

#ifndef ROW8_STRUCTURAL_NO_MAIN
int main(){MODP=Q;Vec all;int col=0;load_ck("work/formal-probes/raw_wfa_r8.ck",8,col,all);std::vector<Packed>H0,H1,H2;for(auto&p:all){int h=unpack(p).sp;if(h==0)H0.push_back(p);else if(h==1)H1.push_back(p);else if(h==2)H2.push_back(p);}auto A0=loadA01(0,H0.size()),A1=loadA01(1,H1.size()),A2=loadA2(H2);auto S0=makeSpace(0,H0,A0),S1=makeSpace(1,H1,A1),S2=makeSpace(2,H2,A2);std::cout<<"spaces "<<S0.dim<<","<<S1.dim<<","<<S2.dim<<"\n";runTrans(S0,S0,0);runTrans(S0,S1,2);runTrans(S1,S1,0);runTrans(S1,S0,1);runTrans(S1,S2,2);runTrans(S2,S2,0);runTrans(S2,S1,1);}

#endif
