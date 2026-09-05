#!/usr/bin/env python3
import pathlib,sys
if len(sys.argv)!=3:raise SystemExit('usage: gen-b300-plan-target.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
old='size_t target=std::min(requested_target,min_free-reserve);int effective_target_mib=int(target>>20);'
new='''size_t target=std::min(requested_target,min_free-reserve);int effective_target_mib=int(target>>20);
    int plan_target_divisor=1;
    if(const char*e=std::getenv("GRIDFP_PLAN_TARGET_DIVISOR")){char*end=nullptr;long v=std::strtol(e,&end,10);if(!end||*end||v<1||v>16){std::cerr<<"invalid GRIDFP_PLAN_TARGET_DIVISOR="<<e<<" expected 1..16\\n";return 19;}plan_target_divisor=int(v);}
    int plan_target_mib=std::max(1,effective_target_mib/plan_target_divisor);
    if(const char*e=std::getenv("GRIDFP_PLAN_TARGET_MIB")){char*end=nullptr;long v=std::strtol(e,&end,10);if(!end||*end||v<1){std::cerr<<"invalid GRIDFP_PLAN_TARGET_MIB="<<e<<"\\n";return 19;}plan_target_mib=std::min<int>(plan_target_mib,int(v));}
    size_t plan_target=size_t(plan_target_mib)<<20;'''
if s.count(old)!=1:raise SystemExit(f'plan target declaration anchor count={s.count(old)}')
s=s.replace(old,new,1)
old='requested_scratch_mib="<<target_mib<<" effective_scratch_mib="<<effective_target_mib<<" reserve_mib="<<reserve_mib<<"\\n";'
new='requested_scratch_mib="<<target_mib<<" effective_scratch_mib="<<effective_target_mib<<" plan_target_mib="<<plan_target_mib<<" plan_target_divisor="<<plan_target_divisor<<" reserve_mib="<<reserve_mib<<"\\n";'
if s.count(old)!=1:raise SystemExit(f'plan target log anchor count={s.count(old)}')
s=s.replace(old,new,1)
old='auto t=plan_window(W,hi,lo,target);if(t.max_bytes&&t.max_bytes<=target)'
new='auto t=plan_window(W,hi,lo,plan_target);if(t.max_bytes&&t.max_bytes<=plan_target)'
if s.count(old)!=1:raise SystemExit(f'plan_window target anchor count={s.count(old)}')
s=s.replace(old,new,1)
if 'process_group(ctx[d],W,pw.wp,pw.groups[q],threads,target)' not in s:
    raise SystemExit('scratch target must remain on process_group')
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: planner_target_env=GRIDFP_PLAN_TARGET_MIB planner_divisor_env=GRIDFP_PLAN_TARGET_DIVISOR scratch_target_separate=1 process_group_uses_full_scratch_target=1')
