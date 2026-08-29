#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys

if len(sys.argv)!=3:
    raise SystemExit('usage: lower-b300-batch-row-limit.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()

anchor='''    for(size_t ri=0;ri<mods.size();++ri){Count mod=mods[ri];'''
insert='''    int b300_batch_row_limit=W;
    if(const char*e=std::getenv("B300_ROW_LIMIT")){
        char*end=nullptr;long v=std::strtol(e,&end,10);
        if(!end||*end||v<1||v>W){std::cerr<<"invalid B300_ROW_LIMIT="<<e<<" expected 1.."<<W<<"\\n";return 15;}
        b300_batch_row_limit=int(v);
    }
    std::cerr<<"B300 batch row limit: rows="<<b300_batch_row_limit<<"/"<<W<<" calibration="<<(b300_batch_row_limit<W?1:0)<<"\\n";
    for(size_t ri=0;ri<mods.size();++ri){Count mod=mods[ri];'''
if s.count(anchor)!=1:raise SystemExit(f'batch residue-loop anchor count={s.count(anchor)}')
s=s.replace(anchor,insert,1)
old='for(int row=0;row<W;++row){'
new='for(int row=0;row<b300_batch_row_limit;++row){'
if s.count(old)!=1:raise SystemExit(f'batch row-loop anchor count={s.count(old)}')
s=s.replace(old,new,1)
old='<<" row "<<row+1<<"/"<<W<<"\\n";'
new='<<" row "<<row+1<<"/"<<b300_batch_row_limit<<"\\n";'
if s.count(old)!=1:raise SystemExit(f'batch progress anchor count={s.count(old)}')
s=s.replace(old,new,1)
# Surface calibration scope on each result row so benchmark parsers cannot
# accidentally treat a partial-row residue as a full exact residue.
old='<<" scratch_target_mib="<<effective_target_mib<<" windows="<<done_windows'
new='<<" scratch_target_mib="<<effective_target_mib<<" rows="<<b300_batch_row_limit<<" calibration="<<(b300_batch_row_limit<W?1:0)<<" windows="<<done_windows'
if s.count(old)!=1:raise SystemExit(f'batch result metadata anchor count={s.count(old)}')
s=s.replace(old,new,1)
for required in ('B300_ROW_LIMIT','b300_batch_row_limit','calibration=','rows="<<b300_batch_row_limit'):
    if required not in s:raise SystemExit(f'missing batch row-limit artifact: {required}')
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'lowered {out} batch_row_limit_env=B300_ROW_LIMIT default_rows=W calibration_default=0')
