from pathlib import Path
import os
p=Path('src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_factorized_fullprerank_orbit_blockfused_groupbatch_batch.cu')
st=p.stat()
if (int(st.st_mtime),st.st_size)!=(1788064402,264365):
    raise SystemExit(f'ABORT source changed: mtime={st.st_mtime} size={st.st_size}')
s=p.read_text()

def rep(old,new,n=1):
    global s
    c=s.count(old)
    if c!=n: raise SystemExit(f'pattern count {c} != {n}: {old[:80]!r}')
    s=s.replace(old,new,n)

rep('__constant__ Code* D_F_HIGH_BLOCK_BASE;\n', '__constant__ Code* D_F_HIGH_BLOCK_BASE;\n__constant__ unsigned long long* D_CROSS_LOW_MATCH;\n__constant__ unsigned long long* D_CROSS_HIGH_MATCH;\n')

rep('static P1DestTablesHost G_P1_DEST;\nstatic bool groupbatch_p1_dest_mode(){', '''static P1DestTablesHost G_P1_DEST;\nstatic std::vector<unsigned long long> G_CROSS_LOW_MATCH,G_CROSS_HIGH_MATCH;\nstatic bool groupbatch_matchlut_mode(){\n    static int mode=[](){int v=0;if(const char*e=std::getenv("GRIDFP_GROUPBATCH_MATCHLUT"))v=std::atoi(e);if(v<0||v>1)throw std::runtime_error("GRIDFP_GROUPBATCH_MATCHLUT must be 0 or 1");return v;}();\n    return mode!=0;\n}\nstatic void build_cross_match_words(){\n    constexpr int L=LOW_LUT_K,H=HIGH_LUT_K;\n    G_CROSS_LOW_MATCH.resize(G_FACTOR.low_all_codes.size());\n    for(size_t a=0;a<G_FACTOR.low_all_codes.size();++a){uint32_t code=G_FACTOR.low_all_codes[a];unsigned long long w=0;for(int d=1;d<=L&&d<=15;++d){int q=d,match=-1;for(int z=L-1;z>=0;--z){auto v=oneesan::gridfp::MateValue((code>>(2*z))&3u);if(v==oneesan::gridfp::R){if(--q==0){match=z;break;}}else if(v==oneesan::gridfp::L)++q;}if(match>=0)w|=static_cast<unsigned long long>(match+1)<<(4*(d-1));}G_CROSS_LOW_MATCH[a]=w;}\n    G_CROSS_HIGH_MATCH.resize(G_FACTOR.high_all_codes.size());\n    for(size_t a=0;a<G_FACTOR.high_all_codes.size();++a){uint32_t code=G_FACTOR.high_all_codes[a];unsigned long long w=0;for(int d=1;d<=H&&d<=15;++d){int q=d,match=-1;for(int z=0;z<H;++z){auto v=oneesan::gridfp::MateValue((code>>(2*z))&3u);if(v==oneesan::gridfp::L){if(--q==0){match=z;break;}}else if(v==oneesan::gridfp::R)++q;}if(match>=0)w|=static_cast<unsigned long long>(match+1)<<(4*(d-1));}G_CROSS_HIGH_MATCH[a]=w;}\n    std::cerr<<"groupbatch matchlut low_mib="<<double(G_CROSS_LOW_MATCH.size()*8)/(1<<20)<<" high_mib="<<double(G_CROSS_HIGH_MATCH.size()*8)/(1<<20)<<"\\n";\n}\nstatic bool groupbatch_p1_dest_mode(){''')

rep('template<bool WRAP32,bool P1> __global__ void batch_low_highrr_kernel', 'template<bool WRAP32,bool P1,bool MATCHLUT> __global__ void batch_low_highrr_kernel')
old='''uint32_t hc=D_F_HIGH_MASK_CODES[D_F_HIGH_MASK_OFF[size_t(c->mask)*S+x.he]+hr];int match=-1,s=depth;\n#pragma unroll\n        for(int a=0;a<H;++a){auto mv=MateValue((hc>>(2*a))&3u);if(mv==L){if(--s==0){match=a;break;}}else if(mv==R)++s;}if(match<0)continue;'''
new='''size_t hmo=D_F_HIGH_MASK_OFF[size_t(c->mask)*S+x.he];uint32_t hc=D_F_HIGH_MASK_CODES[hmo+hr];int match=-1;if constexpr(MATCHLUT){uint32_t har=D_F_HIGH_MASK_ALL_RANK[hmo+hr];unsigned long long mw=D_CROSS_HIGH_MATCH[D_F_HIGH_ALL_OFF[x.he]+har];match=(depth>=1&&depth<=H)?int((mw>>(4*(depth-1)))&15u)-1:-1;}else{int ss=depth;\n#pragma unroll\n        for(int a=0;a<H;++a){auto mv=MateValue((hc>>(2*a))&3u);if(mv==L){if(--ss==0){match=a;break;}}else if(mv==R)++ss;}}if(match<0)continue;'''
rep(old,new)

rep('template<bool WRAP32> __global__ void batch_high_crossll_kernel', 'template<bool WRAP32,bool MATCHLUT> __global__ void batch_high_crossll_kernel')
old='''uint32_t code=D_F_LOW_MASK_CODES[D_F_LOW_MASK_OFF[size_t(c->mask)*S+x.hs]+lr];int match=-1,s=depth;\n#pragma unroll\n        for(int a=LOWK-1;a>=0;--a){auto mv=MateValue((code>>(2*a))&3u);if(mv==R){if(--s==0){match=a;break;}}else if(mv==oneesan::gridfp::L)++s;}if(match<0)continue;'''
new='''size_t lmo=D_F_LOW_MASK_OFF[size_t(c->mask)*S+x.hs];uint32_t code=D_F_LOW_MASK_CODES[lmo+lr];int match=-1;if constexpr(MATCHLUT){uint32_t lar=D_F_LOW_MASK_ALL_RANK[lmo+lr];unsigned long long mw=D_CROSS_LOW_MATCH[D_F_LOW_ALL_OFF[x.hs]+lar];match=(depth>=1&&depth<=LOWK)?int((mw>>(4*(depth-1)))&15u)-1:-1;}else{int ss=depth;\n#pragma unroll\n        for(int a=LOWK-1;a>=0;--a){auto mv=MateValue((code>>(2*a))&3u);if(mv==R){if(--ss==0){match=a;break;}}else if(mv==oneesan::gridfp::L)++ss;}}if(match<0)continue;'''
rep(old,new)

rep('''G_FACTOR=build_factor_tables();\n    G_PRERANK=build_prerank_orbit_tables();G_HPR=build_high_prerank_tables();''', '''G_FACTOR=build_factor_tables();\n    if(groupbatch_matchlut_mode())build_cross_match_words();\n    G_PRERANK=build_prerank_orbit_tables();G_HPR=build_high_prerank_tables();''')

rep('''uint64_t p1DestBytes=groupbatch_p1_dest_mode()?uint64_t(G_P1_DEST.rec.size())*8+uint64_t(G_P1_DEST.src.size())*4:0;uint64_t sparseBytes=0;''', '''uint64_t p1DestBytes=groupbatch_p1_dest_mode()?uint64_t(G_P1_DEST.rec.size())*8+uint64_t(G_P1_DEST.src.size())*4:0;uint64_t sparseBytes=groupbatch_matchlut_mode()?uint64_t(G_CROSS_LOW_MATCH.size()+G_CROSS_HIGH_MATCH.size())*8:0;''')

rep('''unsigned long long*p1DR[MAXGPU]{};uint32_t*p1DS[MAXGPU]{};''', '''unsigned long long*p1DR[MAXGPU]{},*cmL[MAXGPU]{},*cmH[MAXGPU]{};uint32_t*p1DS[MAXGPU]{};''')

needle='''ck(cudaMemcpyToSymbol(D_F_HIGH_BLOCK_BASE,&fHBB[d],sizeof(fHBB[d])),"f ptr");ck(cudaMemcpyToSymbol(D_F_LOW_ALL_OFF,G_FACTOR.low_all_off.data(),sizeof(uint32_t)*(MAXW+2)),"f low all off");'''
repl='''ck(cudaMemcpyToSymbol(D_F_HIGH_BLOCK_BASE,&fHBB[d],sizeof(fHBB[d])),"f ptr");ck(cudaMemcpyToSymbol(D_F_LOW_ALL_OFF,G_FACTOR.low_all_off.data(),sizeof(uint32_t)*(MAXW+2)),"f low all off");'''
# upload is inserted after cp8 becomes available, not here

old='''auto cp8=[&](unsigned long long**dst,const std::vector<unsigned long long>&v,const char*w){if(v.empty())return;ck(cudaMalloc(dst,v.size()*sizeof(unsigned long long)),w);ck(cudaMemcpy(*dst,v.data(),v.size()*sizeof(unsigned long long),cudaMemcpyHostToDevice),w);};cp8(&prOB[d],G_PRERANK.owner_block_rec,"pr owner blockorder");'''
new='''auto cp8=[&](unsigned long long**dst,const std::vector<unsigned long long>&v,const char*w){if(v.empty())return;ck(cudaMalloc(dst,v.size()*sizeof(unsigned long long)),w);ck(cudaMemcpy(*dst,v.data(),v.size()*sizeof(unsigned long long),cudaMemcpyHostToDevice),w);};if(groupbatch_matchlut_mode()){cp8(&cmL[d],G_CROSS_LOW_MATCH,"cross low match");cp8(&cmH[d],G_CROSS_HIGH_MATCH,"cross high match");ck(cudaMemcpyToSymbol(D_CROSS_LOW_MATCH,&cmL[d],sizeof(cmL[d])),"cross low match ptr");ck(cudaMemcpyToSymbol(D_CROSS_HIGH_MATCH,&cmH[d],sizeof(cmH[d])),"cross high match ptr");}cp8(&prOB[d],G_PRERANK.owner_block_rec,"pr owner blockorder");'''
rep(old,new)

rep('''BatchFactorIoCfg*dcfg=dm.cfg;BatchFactorIoTask*dt=dm.io;BatchOwnerTask*dot=dm.owner;BatchCrossTask*dct=dm.cross,*dit=dm.inv;BatchP1DestTask*dpt=dm.p1;Count*arena=reinterpret_cast<Count*>(c.arena);bool wrap32=G_GROUPBATCH_WRAP32_ACTIVE;''', '''BatchFactorIoCfg*dcfg=dm.cfg;BatchFactorIoTask*dt=dm.io;BatchOwnerTask*dot=dm.owner;BatchCrossTask*dct=dm.cross,*dit=dm.inv;BatchP1DestTask*dpt=dm.p1;Count*arena=reinterpret_cast<Count*>(c.arena);bool wrap32=G_GROUPBATCH_WRAP32_ACTIVE,matchlut=groupbatch_matchlut_mode();''')

old='''if(b.cross_count[pp]){if(b.fix_low){if(wrap32)batch_high_crossll_kernel<true><<<b.cross_count[pp],threads,0,c.sMain>>>(arena,dcfg,dct+b.cross_off[pp],b.cross_count[pp]);else batch_high_crossll_kernel<false><<<b.cross_count[pp],threads,0,c.sMain>>>(arena,dcfg,dct+b.cross_off[pp],b.cross_count[pp]);}else if(pp==1)batch_low_highrr_kernel<false,true><<<b.cross_count[pp],threads,0,c.sMain>>>(arena,dcfg,dct+b.cross_off[pp],b.cross_count[pp]);else if(wrap32)batch_low_highrr_kernel<true,false><<<b.cross_count[pp],threads,0,c.sMain>>>(arena,dcfg,dct+b.cross_off[pp],b.cross_count[pp]);else batch_low_highrr_kernel<false,false><<<b.cross_count[pp],threads,0,c.sMain>>>(arena,dcfg,dct+b.cross_off[pp],b.cross_count[pp]);}'''
new='''if(b.cross_count[pp]){if(b.fix_low){if(wrap32){if(matchlut)batch_high_crossll_kernel<true,true><<<b.cross_count[pp],threads,0,c.sMain>>>(arena,dcfg,dct+b.cross_off[pp],b.cross_count[pp]);else batch_high_crossll_kernel<true,false><<<b.cross_count[pp],threads,0,c.sMain>>>(arena,dcfg,dct+b.cross_off[pp],b.cross_count[pp]);}else{if(matchlut)batch_high_crossll_kernel<false,true><<<b.cross_count[pp],threads,0,c.sMain>>>(arena,dcfg,dct+b.cross_off[pp],b.cross_count[pp]);else batch_high_crossll_kernel<false,false><<<b.cross_count[pp],threads,0,c.sMain>>>(arena,dcfg,dct+b.cross_off[pp],b.cross_count[pp]);}}else if(pp==1){if(matchlut)batch_low_highrr_kernel<false,true,true><<<b.cross_count[pp],threads,0,c.sMain>>>(arena,dcfg,dct+b.cross_off[pp],b.cross_count[pp]);else batch_low_highrr_kernel<false,true,false><<<b.cross_count[pp],threads,0,c.sMain>>>(arena,dcfg,dct+b.cross_off[pp],b.cross_count[pp]);}else if(wrap32){if(matchlut)batch_low_highrr_kernel<true,false,true><<<b.cross_count[pp],threads,0,c.sMain>>>(arena,dcfg,dct+b.cross_off[pp],b.cross_count[pp]);else batch_low_highrr_kernel<true,false,false><<<b.cross_count[pp],threads,0,c.sMain>>>(arena,dcfg,dct+b.cross_off[pp],b.cross_count[pp]);}else{if(matchlut)batch_low_highrr_kernel<false,false,true><<<b.cross_count[pp],threads,0,c.sMain>>>(arena,dcfg,dct+b.cross_off[pp],b.cross_count[pp]);else batch_low_highrr_kernel<false,false,false><<<b.cross_count[pp],threads,0,c.sMain>>>(arena,dcfg,dct+b.cross_off[pp],b.cross_count[pp]);}}'''
rep(old,new)

# final optimistic check
st2=p.stat()
if (int(st2.st_mtime),st2.st_size)!=(1788064402,264365):
    raise SystemExit(f'ABORT source changed during patch: mtime={st2.st_mtime} size={st2.st_size}')
p.write_text(s)
print('patched',p,'size',len(s))
