#define ROW8_STRUCTURAL_NO_MAIN 1
#include "row8_structural_lower_sparsity.cpp"
#include <array>
static void genw(int p,int nt,int bal,std::string&w,std::vector<std::pair<Packed,std::string>>&o){if(p==8){if(!nt&&!bal){Packed x;if(enc(w,x))o.push_back({x,w});}return;}if(nt>8-p)return;w[p]='N';genw(p+1,nt,bal,w,o);w[p]='U';genw(p+1,nt,bal+1,w,o);w[p]='D';genw(p+1,nt,bal-1,w,o);if(nt&&bal==0){w[p]='T';genw(p+1,nt-1,0,w,o);}}
static std::vector<std::pair<Packed,std::string>> wordsS(int h){std::string w(8,'N');std::vector<std::pair<Packed,std::string>>v;genw(0,h,0,w,v);std::sort(v.begin(),v.end(),[](auto&a,auto&b){return a.first<b.first;});v.erase(std::unique(v.begin(),v.end(),[](auto&a,auto&b){return a.first==b.first;}),v.end());return v;}
static bool hard3(std::string const&w){int bal=0,n=0;auto flush=[&](){bool z=n>=2;bal=n=0;return z;};for(char c:w){if(c=='T'){if(flush())return true;continue;}if(c=='N')continue;int d=c=='U'?1:-1;if(bal==0&&d<0)++n;bal+=d;}return flush();}
static Packed mix3(std::string const&w){State s{};s.n=8;int q=1,sk=0;std::array<int,4>z{};bool got=false;int st=0;while(st<8){int en=st;while(en<8&&w[en]!='T')++en;int b=0,ns=-1,c=0;for(int i=st;i<en;++i){if(w[i]=='N')continue;int d=w[i]=='U'?1:-1;if(b==0&&d<0)ns=i;b+=d;if(ns>=0&&b==0){if(c<2){z[2*c]=ns;z[2*c+1]=i;}++c;ns=-1;}}if(c>=2){got=true;break;}st=en+1;}if(!got)throw std::runtime_error("mix3");for(int i=0;i<8;++i)if(w[i]=='T'){int c=q++;s.deg[i]=1;s.comp[i]=c;s.stack[sk++]=c;}auto add=[&](int a,int b,int stt){int c=q++;s.deg[a]=s.deg[b]=1;s.comp[a]=s.comp[b]=c;s.status[c]=stt;};add(z[0],z[3],1);add(z[1],z[2],0);s.sp=sk;s.ns=q;return pack(s);}
static Packed rel4(int gap,bool nested){State s{};s.n=8;s.sp=4;int q=1,sk=0;std::vector<int>f;for(int i=0;i<8;++i){if(gap<=i&&i<gap+4)f.push_back(i);else{int z=q++;s.deg[i]=1;s.comp[i]=z;s.stack[sk++]=z;}}auto add=[&](int a,int b,int st){int z=q++;s.deg[a]=s.deg[b]=1;s.comp[a]=s.comp[b]=z;s.status[z]=st;};if(!nested){add(f[0],f[1],1);add(f[2],f[3],1);}else{add(f[0],f[3],1);add(f[1],f[2],0);}s.ns=q;return pack(s);}
static std::vector<std::vector<uint32_t>> makeAupper(int h,std::vector<Packed>const&H){auto ws=wordsS(h);std::vector<std::vector<uint32_t>>A(D_[h],std::vector<uint32_t>(H.size()));for(int j=0;j<D_[h];++j){int i=std::lower_bound(H.begin(),H.end(),ws[j].first)-H.begin();if(i==(int)H.size()||!(H[i]==ws[j].first))throw std::runtime_error("can missing");A[j][i]=1;if(h==3&&hard3(ws[j].second)){auto m=mix3(ws[j].second);int k=std::lower_bound(H.begin(),H.end(),m)-H.begin();if(k==(int)H.size()||!(H[k]==m))throw std::runtime_error("mix missing");A[j][k]=1;}}
 if(h==4){int hk[5]={337,351,371,397,417};for(int g=0;g<5;++g){if(!(ws[hk[g]].first==rel4(g,false)))throw std::runtime_error("h4 order");int k=std::lower_bound(H.begin(),H.end(),rel4(g,true))-H.begin();A[hk[g]][k]=1;}}
 return A;}
#ifndef ROW8_STRUCTURAL_UPPER_NO_MAIN
int main(int argc,char**argv){MODP=Q;Vec all;int col=0;load_ck("work/formal-probes/raw_wfa_r8.ck",8,col,all);std::array<std::vector<Packed>,9>H;for(auto&p:all){int h=unpack(p).sp;H[h].push_back(p);}std::array<std::unique_ptr<Space>,9>S; // build h2..8
 auto A2=loadA2(H[2]);S[2]=std::make_unique<Space>(makeSpace(2,H[2],A2));
 for(int h=3;h<=8;++h){auto A=makeAupper(h,H[h]);S[h]=std::make_unique<Space>(makeSpace(h,H[h],A));std::cout<<"space h="<<h<<" dim="<<S[h]->dim<<" states="<<H[h].size()<<"\n";}
 int only=argc>1?atoi(argv[1]):-1;bool fast=argc>2&&std::string(argv[2])=="fast";for(int h=2;h<=8;++h){if(only>=0&&h!=only)continue;for(int a=0;a<3;++a){int h2=h+DEL_[a];if(h2<2||h2>8)continue;if(fast)runTransFast(*S[h],*S[h2],a);else runTrans(*S[h],*S[h2],a);}}
}

#endif
