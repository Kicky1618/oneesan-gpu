#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"; W=$((N+1)); MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"
TARGET_MIB="${TARGET_MIB:-65536}"; MAX_WINDOW="${MAX_WINDOW:-14}"; ARCH="${ARCH:-native}"
REPEATS="${REPEATS:-1}"; THREADS="${THREADS:-256}"; HIGH_DROP_CHUNK="${HIGH_DROP_CHUNK:-0}"
[[ "$N" == 27 ]] || { echo "this A/B is currently specialized for n=27 / W28 fast caches" >&2; exit 2; }
command -v nvcc >/dev/null || { echo "nvcc required" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi required" >&2; exit 2; }

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_exact_main_ilp124}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"
BASE_BIN="$ONEESAN_BUILD_DIR/b300_exact_ilp1_n27"
ILP2_BIN="$ONEESAN_BUILD_DIR/b300_exact_ilp2_n27"
ILP4_BIN="$ONEESAN_BUILD_DIR/b300_exact_ilp4_n27"

# Generate the complete current production source with ILP1. The build leaves
# the last transformed CUDA file in the build directory; ILP2/4 are then applied
# to that exact source so every rank/cache/shard transform is identical.
N=27 ARCH="$ARCH" OUT="$BASE_BIN" MAIN_PULL_ILP=1 HIGH_DROP_CHUNK="$HIGH_DROP_CHUNK" PTXAS_VERBOSE=1 \
  bash "$ONEESAN_ROOT/scripts/build/b300-hbm32-batch.sh" \
  >"$LOGDIR/ilp1.build.out" 2>"$LOGDIR/ilp1.build.err"
FINAL_SRC="$(sed -nE 's/^  build_source=(.*)$/\1/p' "$LOGDIR/ilp1.build.out" | tail -n1)"
[[ -n "$FINAL_SRC" && -f "$FINAL_SRC" ]] || { echo "cannot locate generated ILP1 source from build output" >&2; exit 3; }

ILP2_SRC="$ONEESAN_BUILD_DIR/b300_exact_ilp2_postfinal_n27.cu"
ILP4_SRC="$ONEESAN_BUILD_DIR/b300_exact_ilp4_postfinal_n27.cu"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-main-pull-ilp2.py" "$FINAL_SRC" "$ILP2_SRC" >"$LOGDIR/ilp2.transform.out"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-main-pull-ilp4.py" "$FINAL_SRC" "$ILP4_SRC" >"$LOGDIR/ilp4.transform.out"

compile_variant(){
  local src="$1" bin="$2" label="$3"
  TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" -Xptxas=-v \
    -DTARGET_W=28 -DLOW_LUT_K=14 -DHIGH_LUT_K=13 "$src" -o "$bin" \
    >"$LOGDIR/${label}.build.out" 2>"$LOGDIR/${label}.build.err"
}
compile_variant "$ILP2_SRC" "$ILP2_BIN" ilp2
compile_variant "$ILP4_SRC" "$ILP4_BIN" ilp4

PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
for x in ilp1 ilp2 ilp4; do
  python3 "$PARSER" "$LOGDIR/${x}.build.err" --label "$x" --contains main_pull_kernel >>"$RESOURCE" || true
done

field(){ local key="$1" line="$2"; sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1; }
summarize_dmon(){ python3 - "$1" <<'PY'
import sys
v=[]
for line in open(sys.argv[1],encoding='utf-8',errors='replace'):
    s=line.strip()
    if not s or s.startswith('#'): continue
    a=s.split()
    if len(a)>=3:
        try:v.append(float(a[2]))
        except ValueError:pass
if len(v)>16:v=v[8:]
if not v: print('NA\tNA\t0')
else: print(f'{sum(v)/len(v):.3f}\t{max(v):.3f}\t{len(v)}')
PY
}

printf 'mode\trepeat\tresidue\twall_s\tmc_avg_pct\tmc_max_pct\tmc_samples\n' >"$RESULT"
run_case(){
  local mode="$1" bin="$2" rep="$3" so="$LOGDIR/${mode}_r${rep}.out" se="$LOGDIR/${mode}_r${rep}.err" dm="$LOGDIR/${mode}_r${rep}.dmon"
  nvidia-smi dmon -s u -d 1 >"$dm" 2>/dev/null & local dp=$!
  set +e; GRIDFP_THREADS="$THREADS" "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"; local rc=$?; set -e
  kill "$dp" 2>/dev/null || true; wait "$dp" 2>/dev/null || true
  ((rc==0)) || { echo "$mode failed rc=$rc" >&2; exit "$rc"; }
  local line residue wall avg mx ns
  line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || { echo "$mode missing residue" >&2; exit 4; }
  residue="$(field residue "$line")"; wall="$(field wall_s "$line")"; IFS=$'\t' read -r avg mx ns <<<"$(summarize_dmon "$dm")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$rep" "$residue" "${wall:-NA}" "$avg" "$mx" "$ns" >>"$RESULT"
}
for ((r=1;r<=REPEATS;++r)); do run_case ilp1 "$BASE_BIN" "$r"; run_case ilp2 "$ILP2_BIN" "$r"; run_case ilp4 "$ILP4_BIN" "$r"; done

python3 - "$RESULT" <<'PY'
import csv,statistics,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t')); by={}
for r in rows:by.setdefault(r['mode'],[]).append(r)
res={m:{r['residue'] for r in g} for m,g in by.items()}
if any(len(x)!=1 for x in res.values()): raise SystemExit(f'unstable residues: {res}')
z={m:next(iter(x)) for m,x in res.items()}
if len(set(z.values()))!=1: raise SystemExit(f'RESIDUE MISMATCH {z}')
print('residue_match=OK',next(iter(z.values())))
summary={}
for m,g in by.items():
    wall=statistics.median(float(r['wall_s']) for r in g)
    av=[float(r['mc_avg_pct']) for r in g if r['mc_avg_pct']!='NA']; mx=[float(r['mc_max_pct']) for r in g if r['mc_max_pct']!='NA']
    summary[m]=(wall,statistics.median(av) if av else float('nan'),max(mx) if mx else float('nan'))
for m,(w,a,x) in sorted(summary.items(),key=lambda kv:kv[1][0]): print(m,f'wall_s={w:.6f}',f'mc_avg_pct={a:.3f}',f'mc_max_pct={x:.3f}',f'speedup_vs_ilp1={summary["ilp1"][0]/w:.6f}')
print('BEST',min(summary,key=lambda m:summary[m][0]))
PY

echo "result=$RESULT ptxas=$RESOURCE logs=$LOGDIR" >&2
