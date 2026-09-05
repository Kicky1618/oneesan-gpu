#define ROW8_STRUCTURAL_INTEGER_NO_MAIN 1
#include "row8_structural_integer_closure.cpp"
#include <array>
#include <memory>

int main(int argc,char**argv){
  MODP=4294967291u;
  Vec all; int col=0;
  if(!load_ck("work/formal-probes/raw_wfa_r8.ck",8,col,all)) return 2;
  std::array<std::vector<Packed>,9> H;
  for(auto const&p:all) H[unpack(p).sp].push_back(p);
  std::array<std::unique_ptr<Space>,9> S;
  S[0]=std::make_unique<Space>(makeSpace(0,H[0],loadA01(0,H[0].size())));
  S[1]=std::make_unique<Space>(makeSpace(1,H[1],loadA01(1,H[1].size())));
  S[2]=std::make_unique<Space>(makeSpace(2,H[2],loadA2(H[2])));
  for(int h=3;h<=8;++h) S[h]=std::make_unique<Space>(makeSpace(h,H[h],makeAupper(h,H[h])));
  int only=argc>1?std::atoi(argv[1]):-1;
  bool ok=true;
  for(int h=0;h<=8;++h){
    if(only>=0 && h!=only) continue;
    for(int a=0;a<3;++a){
      int h2=h+DEL_[a];
      if(h2<0||h2>8) continue;
      ok &= verifyI(*S[h],*S[h2],a);
    }
  }
  if(only<0 || only==0) ok &= verifyFinalI(*S[0],0);
  if(only<0 || only==1) ok &= verifyFinalI(*S[1],1);
  std::cout << "structural_integer_full_exact=" << ok << " col=" << col << "\n";
  return ok?0:1;
}
