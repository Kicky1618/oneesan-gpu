#define ROW8_GAP_TRANSITION_NO_MAIN 1
#include "row8_gap_transition_exact.cpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>
#include <omp.h>

struct FileHdr2 { char magic[8]; uint32_t version,r,dims[9]; uint64_t total_nnz; };
struct BlockHdr2 { uint32_t sym,h,h2,src_dim,dst_dim; uint64_t nnz; };
struct VecHdr2 { char magic[8]; uint32_t version,r,count; };
struct VecEntry2 { uint32_t tag,sym,h,nz; };
struct ABlock { int h2=-1,src=0,dst=0; std::vector<uint32_t> rp; std::vector<uint16_t> ci; };
struct BaseV { int h=-1; std::vector<uint16_t> idx; };
static constexpr int DEL2[3]={0,-1,1};

static std::array<std::array<ABlock,9>,3> load_graph2(std::array<int,9>&dims){std::ifstream in("work/row8_gap/row8_gap_u01_v1.bin",std::ios::binary);if(!in)throw std::runtime_error("graph");FileHdr2 f{};in.read((char*)&f,sizeof(f));for(int h=0;h<9;++h)dims[h]=f.dims[h];uint32_t nb=0;in.read((char*)&nb,4);std::array<std::array<ABlock,9>,3>g{};for(uint32_t q=0;q<nb;++q){BlockHdr2 b{};in.read((char*)&b,sizeof(b));auto&z=g[b.sym][b.h];z.h2=b.h2;z.src=b.src_dim;z.dst=b.dst_dim;z.rp.resize(z.src+1);z.ci.resize(b.nnz);in.read((char*)z.rp.data(),z.rp.size()*4);in.read((char*)z.ci.data(),z.ci.size()*2);}return g;}
static void load_vecs2(std::array<BaseV,3>&a){std::ifstream in("work/row8_gap/row8_gap_vectors_v1.bin",std::ios::binary);VecHdr2 h{};in.read((char*)&h,sizeof(h));for(uint32_t q=0;q<h.count;++q){VecEntry2 e{};in.read((char*)&e,sizeof(e));std::vector<uint16_t>x(e.nz);in.read((char*)x.data(),x.size()*2);if(e.tag==1){a[e.sym].h=e.h;a[e.sym].idx=std::move(x);}}}
struct VGroup{int rows=0,dim=0;std::vector<uint32_t>d;};
static inline uint32_t madd(uint32_t a,uint32_t b){uint32_t z=a+b;return z>=Q||z<a?uint32_t((uint64_t)a+b-Q):z;}
static std::array<VGroup,9> vstep(std::array<VGroup,9>const&cur,std::array<std::array<ABlock,9>,3>const&G,std::array<int,9>const&dims){std::array<int,9>nr{};for(int h=0;h<9;++h)if(cur[h].rows)for(int a=0;a<3;++a){int h2=h+DEL2[a];if(h2>=0&&h2<9)nr[h2]+=cur[h].rows;}std::array<VGroup,9>nxt{};for(int h=0;h<9;++h){nxt[h].rows=nr[h];nxt[h].dim=dims[h];if(nr[h])nxt[h].d.assign((size_t)nr[h]*dims[h],0);}std::array<int,9>off{};for(int h=0;h<9;++h)if(cur[h].rows)for(int a=0;a<3;++a){int h2=h+DEL2[a];if(h2<0||h2>=9)continue;auto const&b=G[a][h];int oo=off[h2];
#pragma omp parallel for schedule(static)
 for(int r=0;r<cur[h].rows;++r){auto*dst=nxt[h2].d.data()+(size_t)(oo+r)*dims[h2];auto const*src=cur[h].d.data()+(size_t)r*dims[h];for(int i=0;i<dims[h];++i){uint32_t v=src[i];if(!v)continue;for(uint32_t e=b.rp[i];e<b.rp[i+1];++e){auto j=b.ci[e];uint64_t s=(uint64_t)dst[j]+v;dst[j]=s>=Q?s-Q:s;}}}
 off[h2]+=cur[h].rows;}return nxt;}

static bool decode8(State const&s,std::string&tok){if(!gap8(s))return false;tok.assign(8,'?');std::array<int,MAXC> ns{},lo{},hi{};lo.fill(99);hi.fill(-1);for(int i=0;i<s.sp;++i)if(s.stack[i])++ns[s.stack[i]];for(int i=0;i<8;++i)if(s.comp[i]){int q=s.comp[i];lo[q]=std::min(lo[q],i);hi[q]=std::max(hi[q],i);}for(int i=0;i<8;++i)if(!s.comp[i])tok[i]='N';else if(ns[s.comp[i]])tok[i]='T';for(int q=1;q<s.ns;++q)if(!ns[q]&&hi[q]>=0){int a=lo[q],b=hi[q];if(s.status[q]&1){tok[a]='D';tok[b]='U';}else{tok[a]='U';tok[b]='D';}}int bal=0;for(char c:tok){if(c=='T'){if(bal)return false;}else if(c=='U')++bal;else if(c=='D')--bal;else if(c!='N')return false;}return bal==0;}
static uint32_t invm(uint32_t a){uint64_t e=Q-2,x=a,r=1;while(e){if(e&1)r=r*x%Q;x=x*x%Q;e>>=1;}return r;}
static int rankm(std::vector<uint32_t>a,int R,int C){int rk=0;for(int c=0;c<C&&rk<R;++c){int p=rk;while(p<R&&!a[(size_t)p*C+c])++p;if(p==R)continue;for(int j=c;j<C;++j)std::swap(a[(size_t)p*C+j],a[(size_t)rk*C+j]);uint32_t iv=invm(a[(size_t)rk*C+c]);for(int j=c;j<C;++j)a[(size_t)rk*C+j]=(uint64_t)a[(size_t)rk*C+j]*iv%Q;for(int i=rk+1;i<R;++i)if(a[(size_t)i*C+c]){uint32_t f=a[(size_t)i*C+c];for(int j=c;j<C;++j)a[(size_t)i*C+j]=(a[(size_t)i*C+j]+Q-(uint64_t)f*a[(size_t)rk*C+j]%Q)%Q;}++rk;}return rk;}
struct Flat {int total=0;std::vector<int> deg,li,ri,lcnt,rcnt;};
static Flat make_flat(std::vector<std::string>const&tok,int h,int cut){int total=8-h;Flat F;F.total=total;F.deg.resize(tok.size());F.li.resize(tok.size());F.ri.resize(tok.size());F.lcnt.assign(total+1,0);F.rcnt.assign(total+1,0);std::vector<std::map<std::string,int>>LM(total+1),RM(total+1);for(size_t gi=0;gi<tok.size();++gi){std::vector<std::string>seg;std::string cur;for(char c:tok[gi]){if(c=='T'){seg.push_back(cur);cur.clear();}else cur.push_back(c);}seg.push_back(cur);if((int)seg.size()!=h+1)throw std::runtime_error("segments");std::string L,R;int dl=0;for(int q=0;q<cut;++q){if(q)L.push_back('|');L+=seg[q];dl+=seg[q].size();}for(int q=cut;q<=h;++q){if(q>cut)R.push_back('|');R+=seg[q];}auto [il,_l]=LM[dl].emplace(L,LM[dl].size());auto [ir,_r]=RM[dl].emplace(R,RM[dl].size());F.deg[gi]=dl;F.li[gi]=il->second;F.ri[gi]=ir->second;}for(int d=0;d<=total;++d){F.lcnt[d]=LM[d].size();F.rcnt[d]=RM[d].size();}return F;}
static int schmidt(const uint32_t*v,int dim,Flat const&F){int sum=0;for(int d=0;d<=F.total;++d){int R=F.lcnt[d],C=F.rcnt[d];if(!R||!C)continue;std::vector<uint32_t>M((size_t)R*C);for(int i=0;i<dim;++i)if(F.deg[i]==d)M[(size_t)F.li[i]*C+F.ri[i]]=v[i];sum+=rankm(std::move(M),R,C);}return sum;}
int main(int ac,char**av){int lev=ac>1?std::atoi(av[1]):8;std::array<int,9>dims{};auto AG=load_graph2(dims);std::array<BaseV,3>alpha{};load_vecs2(alpha);std::array<VGroup,9>cur{};for(int a=0;a<3;++a){int h=alpha[a].h;if(h<0)continue;int old=cur[h].rows++;cur[h].dim=dims[h];cur[h].d.resize((size_t)cur[h].rows*dims[h]);for(auto j:alpha[a].idx)cur[h].d[(size_t)old*dims[h]+j]=1;}for(int l=2;l<=lev;++l)cur=vstep(cur,AG,dims);
 Vec all;int col=0;load_ck("work/formal-probes/raw_wfa_r8.ck",8,col,all);std::array<std::vector<Packed>,9>H;for(auto&p:all)H[unpack(p).sp].push_back(p);std::array<std::unique_ptr<Space>,9>S;S[0]=std::make_unique<Space>(makeSpace(0,H[0],loadA01(0,H[0].size())));S[1]=std::make_unique<Space>(makeSpace(1,H[1],loadA01(1,H[1].size())));S[2]=std::make_unique<Space>(makeSpace(2,H[2],loadA2(H[2])));for(int h=3;h<=8;++h)S[h]=std::make_unique<Space>(makeSpace(h,H[h],makeAupper(h,H[h])));std::array<std::unique_ptr<GSpace>,9>G;for(int h=0;h<=8;++h)G[h]=std::make_unique<GSpace>(make_gspace(*S[h]));
 for(int h:{1,2,3}){if(!cur[h].rows)continue;std::vector<std::string>tok(dims[h]);for(int si=0;si<(int)S[h]->states.size();++si){int gi=G[h]->raw_gap_global[si];if(gi>=0&&!decode8(unpack(S[h]->states[si]),tok[gi]))throw std::runtime_error("decode");}for(int cut=1;cut<=h;++cut){auto F=make_flat(tok,h,cut);std::map<int,int>hist;int mx=0;long long sum=0;for(int r=0;r<cur[h].rows;++r){int z=schmidt(cur[h].d.data()+(size_t)r*dims[h],dims[h],F);++hist[z];mx=std::max(mx,z);sum+=z;}std::cout<<"lev="<<lev<<" h="<<h<<" cut="<<cut<<" rows="<<cur[h].rows<<" dim="<<dims[h]<<" avg_rank="<<double(sum)/cur[h].rows<<" max_rank="<<mx<<" hist";for(auto[k,c]:hist)std::cout<<' '<<k<<':'<<c;std::cout<<"\n";}}
}
