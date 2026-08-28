#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

static int dh(char c){return c=='A'?-1:c=='D'?1:0;}
static int height_before(std::string const&s,int k){int h=0;for(int i=0;i<k;++i)h+=dh(s[i]);return h;}
static bool valid(std::string const&s){int h=0;for(char c:s){h+=dh(c);if(h<0)return false;}return h==0;}

static void gen(int r,int pos,int h,std::string&s,std::vector<std::string>&out){
    if(pos==r){if(h==0)out.push_back(s);return;}
    if(h){s.push_back('A');gen(r,pos+1,h-1,s,out);s.pop_back();}
    s.push_back('B');gen(r,pos+1,h,s,out);s.pop_back();
    s.push_back('C');gen(r,pos+1,h,s,out);s.pop_back();
    s.push_back('D');gen(r,pos+1,h+1,s,out);s.pop_back();
}
static std::vector<std::string> paths(int r){std::vector<std::string>o;std::string s;gen(r,0,0,s,o);return o;}

static std::vector<std::string> insert_arc(std::string const&s,int i){
    int r=int(s.size());std::vector<std::string>out;
    if(i==0)return{s+"C"};
    if(i&1){int j=(i-1)/2,k=r-j;return{s.substr(0,k)+"B"+s.substr(k)};}
    int j=i/2,k=r-j,h=height_before(s,k);char x=s[k];std::vector<std::string>rep;
    if(x=='A')rep={"AC","CA"};
    else if(x=='B'){rep={"BC","CB","DA"};if(h)rep.push_back("AD");}
    else if(x=='C')rep={"CC"};
    else rep={"CD","DC"};
    for(auto const&q:rep){auto t=s.substr(0,k)+q+s.substr(k+1);assert(valid(t));out.push_back(std::move(t));}
    return out;
}

static std::vector<std::string> cap_arc(std::string const&s,int i){
    int r=int(s.size());assert(r>=1);
    if(i==0){if(s.back()!='B')return{};return{s.substr(0,r-1)};}
    if(i&1){int j=(i-1)/2,k=(r-1)-j;if(s[k]!='C')return{};return{s.substr(0,k)+s.substr(k+1)};}
    int j=i/2,k=(r-1)-j;std::string p=s.substr(k,2);char y=0;
    if(p=="AB"||p=="BA")y='A';
    else if(p=="BB")y='B';
    else if(p=="AD"||p=="BC"||p=="CB"||p=="DA")y='C';
    else if(p=="BD"||p=="DB")y='D';
    else return{};
    auto t=s.substr(0,k)+std::string(1,y)+s.substr(k+2);assert(valid(t));return{t};
}

int main(int argc,char**argv){
    int maxr=argc>1?std::atoi(argv[1]):10;if(maxr>11)maxr=11;
    for(int r=0;r<=maxr;++r){
        auto src=paths(r);int ins_max=0;
        for(int i=0;i<=2*r+1;++i){
            std::unordered_map<std::string,int>in;std::uint64_t edges=0;
            for(auto const&s:src)for(auto const&t:insert_arc(s,i)){++in[t];++edges;}
            int mx=0;for(auto const&kv:in)mx=std::max(mx,kv.second);ins_max=std::max(ins_max,mx);assert(mx<=1);
            std::cout<<"r="<<r<<" insert_i="<<i<<" edges="<<edges<<" max_indegree="<<mx<<"\n";
        }
        int cap_max=0;
        if(r){for(int i=0;i<2*r;++i){
            std::unordered_map<std::string,int>in;std::uint64_t edges=0;
            for(auto const&s:src)for(auto const&t:cap_arc(s,i)){++in[t];++edges;}
            int mx=0;for(auto const&kv:in)mx=std::max(mx,kv.second);cap_max=std::max(cap_max,mx);assert(mx<=4);
            std::cout<<"r="<<r<<" cap_i="<<i<<" edges="<<edges<<" max_indegree="<<mx<<"\n";
        }}
        std::cout<<"SUMMARY r="<<r<<" states="<<src.size()<<" insert_max="<<ins_max<<" cap_max="<<cap_max<<"\n";
    }
    return 0;
}
