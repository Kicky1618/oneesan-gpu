from pathlib import Path

files={
'src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_factorized_fullprerank_orbit_blockfused_groupbatch_batch.cu':(1788071417,267802),
'scripts/run/b300x8.sh':(1788063021,7498),
'scripts/run/b300x8-exact.sh':(1788063021,10392),
}
for f,(mt,sz) in files.items():
    st=Path(f).stat()
    if (int(st.st_mtime),st.st_size)!=(mt,sz): raise SystemExit(f'ABORT changed {f}: {st.st_mtime} {st.st_size}')

# source cleanup
p=Path(next(iter(files))); s=p.read_text()
old='if(p1DR[d])cudaFree(p1DR[d]);if(p1DS[d])cudaFree(p1DS[d]);if(hpOB[d])cudaFree(hpOB[d]);'
new='if(p1DR[d])cudaFree(p1DR[d]);if(p1DS[d])cudaFree(p1DS[d]);if(cmL[d])cudaFree(cmL[d]);if(cmH[d])cudaFree(cmH[d]);if(hpOB[d])cudaFree(hpOB[d]);'
if s.count(old)!=1: raise SystemExit('source cleanup pattern mismatch')
s=s.replace(old,new)
p.write_text(s)

for fn in ['scripts/run/b300x8.sh','scripts/run/b300x8-exact.sh']:
    p=Path(fn); s=p.read_text()
    old='GRIDFP_GROUPBATCH_WRAP32="${GRIDFP_GROUPBATCH_WRAP32:-0}"\n'
    new=old+'GRIDFP_GROUPBATCH_MATCHLUT="${GRIDFP_GROUPBATCH_MATCHLUT:-0}"\n'
    if s.count(old)!=1: raise SystemExit(f'{fn}: var pattern mismatch')
    s=s.replace(old,new)
    old='"GRIDFP_GROUPBATCH_P1_DEST:$GRIDFP_GROUPBATCH_P1_DEST" "GRIDFP_GROUPBATCH_WRAP32:$GRIDFP_GROUPBATCH_WRAP32" "GRIDFP_GROUPBATCH_BATCHES_PER_GPU:$GRIDFP_GROUPBATCH_BATCHES_PER_GPU"'
    new='"GRIDFP_GROUPBATCH_P1_DEST:$GRIDFP_GROUPBATCH_P1_DEST" "GRIDFP_GROUPBATCH_WRAP32:$GRIDFP_GROUPBATCH_WRAP32" "GRIDFP_GROUPBATCH_MATCHLUT:$GRIDFP_GROUPBATCH_MATCHLUT" "GRIDFP_GROUPBATCH_BATCHES_PER_GPU:$GRIDFP_GROUPBATCH_BATCHES_PER_GPU"'
    if s.count(old)!=1: raise SystemExit(f'{fn}: validation list mismatch')
    s=s.replace(old,new)
    old='if (( GRIDFP_GROUPBATCH_WRAP32 > 1 )); then echo "GRIDFP_GROUPBATCH_WRAP32 must be 0 or 1" >&2; exit 2; fi\n'
    new=old+'  if (( GRIDFP_GROUPBATCH_MATCHLUT > 1 )); then echo "GRIDFP_GROUPBATCH_MATCHLUT must be 0 or 1" >&2; exit 2; fi\n'
    if s.count(old)!=1: raise SystemExit(f'{fn}: range pattern mismatch')
    s=s.replace(old,new)
    old='export GRIDFP_GROUPBATCH_DYNAMIC GRIDFP_GROUPBATCH_AFFINITY GRIDFP_GROUPBATCH_STICKY GRIDFP_GROUPBATCH_RECLAIM GRIDFP_GROUPBATCH_GRAPH GRIDFP_GROUPBATCH_P1_DEST GRIDFP_GROUPBATCH_WRAP32 GRIDFP_GROUPBATCH_BATCHES_PER_GPU'
    new='export GRIDFP_GROUPBATCH_DYNAMIC GRIDFP_GROUPBATCH_AFFINITY GRIDFP_GROUPBATCH_STICKY GRIDFP_GROUPBATCH_RECLAIM GRIDFP_GROUPBATCH_GRAPH GRIDFP_GROUPBATCH_P1_DEST GRIDFP_GROUPBATCH_WRAP32 GRIDFP_GROUPBATCH_MATCHLUT GRIDFP_GROUPBATCH_BATCHES_PER_GPU'
    if s.count(old)!=1: raise SystemExit(f'{fn}: export mismatch')
    s=s.replace(old,new)
    old='p1_dest=$GRIDFP_GROUPBATCH_P1_DEST wrap32=$GRIDFP_GROUPBATCH_WRAP32 batches_per_gpu=$GRIDFP_GROUPBATCH_BATCHES_PER_GPU'
    new='p1_dest=$GRIDFP_GROUPBATCH_P1_DEST wrap32=$GRIDFP_GROUPBATCH_WRAP32 matchlut=$GRIDFP_GROUPBATCH_MATCHLUT batches_per_gpu=$GRIDFP_GROUPBATCH_BATCHES_PER_GPU'
    if s.count(old)!=1: raise SystemExit(f'{fn}: log mismatch')
    s=s.replace(old,new)
    p.write_text(s)
print('finish patched')
