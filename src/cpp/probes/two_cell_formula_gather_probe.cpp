#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace {
constexpr char N='N', R='R', L='L';
using Word=std::string;
using Rank=std::uint64_t;
struct Key {
    char type;
    Word w;
    bool operator<(const Key&o)const{return type<o.type||(type==o.type&&w<o.w);}
    bool operator==(const Key&o)const{return type==o.type&&w==o.w;}
};
using CVec=std::map<Key,int64_t>;

std::vector<Word> gen_words(int W){
    std::vector<Word> out; Word cur;
    auto rec=[&](auto&&self,int pos,int h)->void{
        int rem=W-pos; if(h<0||h>rem)return;
        if(pos==W){if(h==0)out.push_back(cur);return;}
        cur.push_back(N);self(self,pos+1,h);cur.pop_back();
        if(h>0){cur.push_back(R);self(self,pos+1,h-1);cur.pop_back();}
        cur.push_back(L);self(self,pos+1,h+1);cur.pop_back();
    }; rec(rec,0,1); return out;
}
bool valid_word(const Word&w){
    int h=1;for(char c:w){if(c==R)--h;else if(c==L)++h;if(h<0)return false;}return h==0;
}
struct LinkState{std::vector<int>mate;int root=-1;};
LinkState decode(const Word&w){
    LinkState s;s.mate.assign(w.size(),-2);std::vector<int>st{-1};
    for(int i=0;i<(int)w.size();++i){
        if(w[i]==N)continue;
        if(w[i]==L)st.push_back(i);
        else {int a=st.back();st.pop_back();if(a==-1){s.root=i;s.mate[i]=-1;}else{s.mate[a]=i;s.mate[i]=a;}}
    }
    assert(st.empty()&&s.root>=0);return s;
}
Word encode(const LinkState&s){
    Word w(s.mate.size(),N);
    for(int i=0;i<(int)w.size();++i){if(i==s.root)w[i]=R;else if(s.mate[i]>=0)w[i]=i<s.mate[i]?L:R;}
    assert(valid_word(w));return w;
}
std::vector<Word> apply_T_basis(const Word&w,int i){
    bool a=w[i]!=N,b=w[i+1]!=N;auto s=decode(w);std::vector<Word>out;
    auto partner=[&](int x){return x==s.root?-1:s.mate[x];};
    if(!a&&!b){out.push_back(w);auto t=s;t.mate[i]=i+1;t.mate[i+1]=i;out.push_back(encode(t));}
    else if(a&&!b){out.push_back(w);auto t=s;int p=partner(i);if(i==s.root){t.mate[i]=-2;t.root=i+1;t.mate[i+1]=-1;}else{t.mate[i]=-2;t.mate[p]=i+1;t.mate[i+1]=p;}out.push_back(encode(t));}
    else if(!a&&b){auto t=s;int p=partner(i+1);if(i+1==s.root){t.mate[i+1]=-2;t.root=i;t.mate[i]=-1;}else{t.mate[i+1]=-2;t.mate[p]=i;t.mate[i]=p;}out.push_back(encode(t));out.push_back(w);}
    else {int p=partner(i),q=partner(i+1);if(p==i+1&&q==i)return out;auto t=s;
        if(i==s.root){t.mate[i]=t.mate[i+1]=-2;t.mate[q]=-1;t.root=q;}
        else if(i+1==s.root){t.mate[i]=t.mate[i+1]=-2;t.mate[p]=-1;t.root=p;}
        else{t.mate[i]=t.mate[i+1]=-2;t.mate[p]=q;t.mate[q]=p;}
        out.push_back(encode(t));
    }
    return out;
}
Word collapse_A(const Word&w,int i){
    bool a=w[i]!=N,b=w[i+1]!=N;assert(!(a&&b));char s=a?w[i]:(b?w[i+1]:N);return w.substr(0,i)+s+w.substr(i+2);
}
void add(CVec&v,const Key&k,int64_t c=1){v[k]+=c;if(v[k]==0)v.erase(k);}
CVec R_raw_basis(const Word&w,int i){
    CVec out;bool a=w[i]!=N,b=w[i+1]!=N;
    if(!a&&!b){add(out,{'A',collapse_A(w,i)});add(out,{'C',w.substr(0,i)+w.substr(i+2)});}
    else if(a!=b)add(out,{'A',collapse_A(w,i)});
    else{auto z=apply_T_basis(w,i);if(!z.empty())add(out,{'A',collapse_A(z[0],i)});}
    return out;
}
std::vector<Word> E_raw_basis(const Key&k,int i){
    if(k.type=='C')return{k.w.substr(0,i)+L+R+k.w.substr(i)};
    char s=k.w[i];if(s==N)return{k.w.substr(0,i)+N+N+k.w.substr(i+1)};
    return{k.w.substr(0,i)+s+N+k.w.substr(i+1),k.w.substr(0,i)+N+s+k.w.substr(i+1)};
}
Key project_key(const Key&k,int i,int W){
    if(k.type=='A')return k;
    if(i<=W-3&&k.w[i]==N)return{'A',k.w.substr(0,i)+L+R+k.w.substr(i+1)};
    return k;
}
CVec K_basis(const Key&src,int W,int i){
    CVec out;
    for(auto const&x:E_raw_basis(src,i))
        for(auto const&[k,c]:R_raw_basis(x,i+1))
            add(out,project_key(k,i+1,W),c);
    return out;
}

struct CountDP{
    int len,fixed;
    std::vector<std::vector<Rank>> memo;
    std::vector<std::vector<uint8_t>> seen;
    CountDP(int len_,int fixed_=-1):len(len_),fixed(fixed_),memo(len_+1,std::vector<Rank>(len_+2)),seen(len_+1,std::vector<uint8_t>(len_+2)){}
    Rank count(int pos,int h){
        int rem=len-pos;if(h<0||h>rem)return 0;if(pos==len)return h==0;
        if(seen[pos][h]) return memo[pos][h];
        seen[pos][h]=1;Rank z=0;
        if(pos!=fixed)z+=count(pos+1,h);
        if(h>0)z+=count(pos+1,h-1);
        z+=count(pos+1,h+1);return memo[pos][h]=z;
    }
};
int ord(char c){return c==N?0:c==R?1:2;}
Rank rank_word(const Word&w,int fixed=-1){
    CountDP dp(w.size(),fixed);Rank rank=0;int h=1;
    for(int pos=0;pos<(int)w.size();++pos){
        const char opts[3]={N,R,L};
        for(char x:opts){
            if(pos==fixed&&x==N)continue;
            if(ord(x)>=ord(w[pos]))break;
            int nh=h+(x==L?1:x==R?-1:0);
            rank+=dp.count(pos+1,nh);
        }
        h+=w[pos]==L?1:w[pos]==R?-1:0;
    }
    assert(h==0);return rank;
}
Word unrank_word(int len,Rank rank,int fixed=-1){
    CountDP dp(len,fixed);assert(rank<dp.count(0,1));Word w;int h=1;
    for(int pos=0;pos<len;++pos){
        bool done=false;
        for(char x:{N,R,L}){
            if(pos==fixed&&x==N)continue;
            int nh=h+(x==L?1:x==R?-1:0);Rank z=dp.count(pos+1,nh);
            if(rank<z){w.push_back(x);h=nh;done=true;break;}
            rank-=z;
        }
        assert(done);
    }
    assert(h==0);return w;
}
Rank a_size(int W){CountDP dp(W-1);return dp.count(0,1);}
Rank reduced_size(int W,int i){CountDP a(W-1),c(W-2,i);return a.count(0,1)+c.count(0,1);}
Rank rank_key(const Key&k,int W,int i){
    if(k.type=='A')return rank_word(k.w);
    assert(k.type=='C'&&k.w[i]!=N);return a_size(W)+rank_word(k.w,i);
}
Key unrank_key(Rank r,int W,int i){
    Rank a=a_size(W);if(r<a)return{'A',unrank_word(W-1,r)};
    return{'C',unrank_word(W-2,r-a,i)};
}

// Exact closure inverse in the probe's left-to-right word convention.
// Production stores the same word in the opposite positional direction, so
// reflect into the gridfp two-bit layout and apply the same balance scans.
std::uint64_t pack_reversed(const Word& w){
    std::uint64_t m=0;
    for(int k=0;k<(int)w.size();++k){
        char c=w[w.size()-1-k];
        std::uint64_t v=c==N?0:c==R?1:2;
        m|=v<<(2*k);
    }
    return m;
}
Word unpack_reversed(std::uint64_t m,int len){
    Word w(len,N);
    for(int k=0;k<len;++k){
        unsigned v=(m>>(2*k))&3u;
        w[len-1-k]=v==0?N:v==1?R:L;
    }
    return w;
}
unsigned mget(std::uint64_t m,int k){return unsigned((m>>(2*k))&3u);}
std::uint64_t mset(std::uint64_t m,int k,unsigned v){
    std::uint64_t z=3ull<<(2*k);return(m&~z)|(std::uint64_t(v)<<(2*k));
}
unsigned mpair(std::uint64_t m,int p){return unsigned((m>>(2*(p-1)))&15u);}
std::uint64_t msetpair(std::uint64_t m,int p,unsigned v){
    std::uint64_t z=15ull<<(2*(p-1));return(m&~z)|(std::uint64_t(v)<<(2*(p-1)));
}
std::vector<Word> closure_preimages(const Word& dest,int j){
    assert(dest[j]==N&&dest[j+1]==N);
    const int len=dest.size();
    const int p=len-1-j;
    std::uint64_t m=pack_reversed(dest);
    assert(p>0&&p<len&&mpair(m,p)==0);
    std::vector<std::uint64_t> cand;
    cand.push_back(msetpair(m,p,0x6)); // RL

    int bal=0;
    for(int q=p-2;q>=0;--q){
        unsigned v=mget(m,q);
        if(bal==0&&v==2){
            auto x=msetpair(m,p,0xa); // LL
            cand.push_back(mset(x,q,1));
        }
        if(v==2)++bal;else if(v==1)--bal;
        if(bal<0)break;
    }
    bal=0;
    for(int q=p+1;q<len;++q){
        unsigned v=mget(m,q);
        if(bal==0&&v==1){
            auto x=msetpair(m,p,0x5); // RR
            cand.push_back(mset(x,q,2));
        }
        if(v==1)++bal;else if(v==2)--bal;
        if(bal<0)break;
    }
    std::vector<Word> out;
    for(auto x:cand){
        Word w=unpack_reversed(x,len);
        if(valid_word(w))out.push_back(w);
    }
    return out;
}
std::vector<Key> project_preimages(const Key&dest,int j,int W){
    std::vector<Key> out{dest};
    if(dest.type=='A'&&j<=W-3&&dest.w.substr(j,2)==Word()+L+R){
        Word u=dest.w.substr(0,j)+N+dest.w.substr(j+2);
        if(valid_word(u))out.push_back({'C',u});
    }
    return out;
}
std::vector<Word> R_preimages(const Key&raw,int j){
    if(raw.type=='C'){
        Word x=raw.w.substr(0,j)+N+N+raw.w.substr(j);assert(valid_word(x));return{x};
    }
    char s=raw.w[j];
    if(s!=N){
        Word a=raw.w.substr(0,j)+s+N+raw.w.substr(j+1);
        Word b=raw.w.substr(0,j)+N+s+raw.w.substr(j+1);
        assert(valid_word(a)&&valid_word(b));return{a,b};
    }
    Word z=raw.w.substr(0,j)+N+N+raw.w.substr(j+1);assert(valid_word(z));
    auto out=closure_preimages(z,j);out.push_back(z);return out;
}
std::vector<Key> E_preimages(const Word&x,int i){
    std::vector<Key> out;bool a=x[i]!=N,b=x[i+1]!=N;
    if(!a&&!b){Word w=x.substr(0,i)+N+x.substr(i+2);if(valid_word(w))out.push_back({'A',w});}
    else if(a!=b){Word w=collapse_A(x,i);if(valid_word(w))out.push_back({'A',w});}
    if(x.substr(i,2)==Word()+L+R){
        Word w=x.substr(0,i)+x.substr(i+2);
        if(valid_word(w)&&w[i]!=N)out.push_back({'C',w});
    }
    return out;
}
std::vector<Key> formula_preimages(const Key&dest,int W,int i){
    std::vector<Key> out;int j=i+1;
    for(auto const&raw:project_preimages(dest,j,W))
        for(auto const&x:R_preimages(raw,j))
            for(auto const&s:E_preimages(x,i))out.push_back(s);
    std::sort(out.begin(),out.end());out.erase(std::unique(out.begin(),out.end()),out.end());
    return out;
}

[[noreturn]]void fail(std::string s){std::cerr<<"FAIL "<<s<<"\n";std::exit(1);}

} // namespace

int main(int argc,char**argv){
    int maxW=argc>1?std::atoi(argv[1]):12;
    std::vector<std::vector<Word>> words(maxW+1);
    for(int W=1;W<=maxW;++W)words[W]=gen_words(W);
    for(int W=4;W<=maxW;++W){
        Rank worst=0,total=0;
        for(int i=0;i<=W-4;++i){
            Rank dim=reduced_size(W,i);
            std::vector<Key> src,dst;
            for(auto const&w:words[W-1])src.push_back({'A',w});
            for(auto const&w:words[W-2])if(w[i]!=N)src.push_back({'C',w});
            for(auto const&w:words[W-1])dst.push_back({'A',w});
            for(auto const&w:words[W-2])if(w[i+1]!=N)dst.push_back({'C',w});
            if(src.size()!=dim||dst.size()!=dim)fail("dim");
            for(Rank r=0;r<dim;++r){
                if(rank_key(src[r],W,i)!=r||unrank_key(r,W,i)!=src[r])fail("src rank");
                if(rank_key(dst[r],W,i+1)!=r||unrank_key(r,W,i+1)!=dst[r])fail("dst rank");
            }
            std::vector<std::vector<Rank>> incoming(dim);
            for(Rank s=0;s<dim;++s)
                for(auto const&[d,c]:K_basis(src[s],W,i)){if(c!=1)fail("coeff");incoming[rank_key(d,W,i+1)].push_back(s);}
            std::vector<uint64_t> in(dim),scatter(dim),gather(dim);
            for(Rank s=0;s<dim;++s)in[s]=1+((s*0x9e3779b97f4a7c15ULL)^(Rank(W)<<32)^Rank(i));
            for(Rank d=0;d<dim;++d){
                auto f=formula_preimages(unrank_key(d,W,i+1),W,i);
                std::vector<Rank> got;for(auto const&s:f)got.push_back(rank_key(s,W,i));
                std::sort(got.begin(),got.end());auto exp=incoming[d];std::sort(exp.begin(),exp.end());
                if(got!=exp)fail("preimage W="+std::to_string(W)+" i="+std::to_string(i)+" d="+std::to_string(d));
                worst=std::max<Rank>(worst,got.size());total+=got.size();
                for(Rank s:exp)scatter[d]+=in[s];
                for(Rank s:got)gather[d]+=in[s];
            }
            if(scatter!=gather)fail("gather");
        }
        std::cout<<"W="<<W<<" reduced="<<reduced_size(W,0)<<" max_preimages="<<worst<<" total_checked="<<total<<" OK\n";
    }
    std::cout<<"ALL_OK\n";
}
