#define ROW8_CANONICAL_TRIE_NO_MAIN 1
#include "row8_canonical_trie_verify.cpp"
#include <fstream>
#include <sstream>

static bool enc_word(std::string const&t,Packed&out){return enc(t,out);} // keep same encoder
static void rec_words(int pos,int nt,int bal,std::string&w,std::vector<std::pair<Packed,std::string>>&out){
    if(pos==8){ if(!nt&&!bal){Packed p;if(enc_word(w,p))out.push_back({p,w});} return; }
    if(nt>8-pos)return;
    w[pos]='N';rec_words(pos+1,nt,bal,w,out);
    w[pos]='U';rec_words(pos+1,nt,bal+1,w,out);
    w[pos]='D';rec_words(pos+1,nt,bal-1,w,out);
    if(nt&&bal==0){w[pos]='T';rec_words(pos+1,nt-1,0,w,out);}
}
static std::vector<std::pair<Packed,std::string>> words(int h){
    std::string w(8,'N');std::vector<std::pair<Packed,std::string>>v;rec_words(0,h,0,w,v);
    std::sort(v.begin(),v.end(),[](auto const&a,auto const&b){return a.first<b.first;});
    v.erase(std::unique(v.begin(),v.end(),[](auto const&a,auto const&b){return a.first==b.first;}),v.end());
    return v;
}
static bool hard(std::string const&w){
    int bal=0, negPrim=0;
    auto flush=[&](){ bool z=negPrim>=2; bal=0;negPrim=0;return z; };
    for(char c:w){
        if(c=='T'){ if(flush())return true; continue; }
        if(c=='N')continue;
        int d=c=='U'?1:-1;
        if(bal==0 && d<0) ++negPrim;
        bal+=d;
    }
    return flush();
}
int main(){
    MODP=1000000007u; constexpr int h=3,a=1,h2=2;
    auto P=loadpc(); BuildCtx c;c.h=h;c.mod=MODP;
    for(int q=0;q<9;++q)c.b[q]=basis(q);
    c.A.resize((size_t)D_[h]*D_[h]);
    for(int aa=0;aa<3;++aa){int hh=h+DEL_[aa];if(0<=hh&&hh<=8)c.E[aa].resize((size_t)D_[h]*D_[hh]);}
    Node root;for(int i=0;i<D_[h];++i){auto d=d9(P[h][i]);Node*t=&root;for(int x:d){if(!t->ch[x])t->ch[x]=std::make_unique<Node>();t=t->ch[x].get();}t->row=i;}
    State z{};z.n=8;WVec init{{pack(z),1}};
    auto t0=std::chrono::steady_clock::now();dfs(root,0,init,c);double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    std::filesystem::create_directories("work/formal-probes/canonical-matrix");
    std::string ep="work/formal-probes/canonical-matrix/E_h3_R_h2_mod1000000007.bin";
    struct EH{char magic[8];uint32_t ver,mod,h,a,h2,rows,cols;uint64_t count,hash;} eh{{'R','8','C','A','N','E','0','1'},1,MODP,h,a,h2,(uint32_t)D_[h],(uint32_t)D_[h2],(uint64_t)c.E[a].size(),fnv(c.E[a].data(),c.E[a].size()*4)};
    std::ofstream out(ep,std::ios::binary|std::ios::trunc);out.write((char*)&eh,sizeof(eh));out.write((char*)c.E[a].data(),c.E[a].size()*4);out.close();
    auto ws=words(h2);if((int)ws.size()!=D_[h2])throw std::runtime_error("word size");
    std::ofstream ho("work/formal-probes/canonical-matrix/h2_hard_indices.txt",std::ios::trunc);int hc=0;
    for(int k=0;k<(int)ws.size();++k)if(hard(ws[k].second)){ho<<k<<' '<<ws[k].second<<'\n';++hc;}
    std::cout<<"emit E rows="<<D_[h]<<" cols="<<D_[h2]<<" hard="<<hc<<" trie_nodes="<<c.nodes<<" build_s="<<sec<<" path="<<ep<<"\n";
    return hc==114?0:3;
}
