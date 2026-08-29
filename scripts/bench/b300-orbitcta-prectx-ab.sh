#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
ARCH="${ARCH:-native}"; REPEATS="${REPEATS:-1}"
THREADS="${THREADS:-256}"; ORBIT_GY="${ORBIT_GY:-128}"
LOW_GX="${LOW_GX:-16}"; LOW_GY="${LOW_GY:-8}"
PM_ACCUM="${PM_ACCUM:-0}"; DIRECTGATHER_SPARSE64="${DIRECTGATHER_SPARSE64:-0}"
WINDOW4="${WINDOW4:-0}"; COL_ILP="${COL_ILP:-1}"
if [[ -z "${EXPECT+x}" ]]; then
  [[ "$N" == 21 && "$MOD" == 4294967291 ]] && EXPECT=998035516 || {
    echo "EXPECT is required unless N=21 MOD=4294967291" >&2; exit 2;
  }
fi
for x in PM_ACCUM DIRECTGATHER_SPARSE64 WINDOW4; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }
done
case "$COL_ILP" in 1|2|4) ;; *) echo "COL_ILP must be 1, 2, or 4" >&2; exit 2;; esac
command -v nvcc >/dev/null || { echo "nvcc required" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi required" >&2; exit 2; }

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_orbitcta_prectx_ab_n${N}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"
RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"; META="${META:-${PREFIX}_prectx.tsv}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"

# mode forward reverse
CASES=(
  "none 0 0"
  "forward 1 0"
  "reverse 0 1"
  "both 1 1"
)

field(){ local key="$1" line="$2"; sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1; }

printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
printf 'mode\towner_lines\tcontext_bytes\tfwd_nn\tfwd_nrnl\trev_nn\trev_nr\trev_nl\tresident_bytes_per_gpu\n' >"$META"
printf 'mode\tprectx_forward\tprectx_reverse\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\tforward_low_s\treverse_low_s\ttranspose_s\n' >"$RESULT"

for spec in "${CASES[@]}"; do
  read -r mode pf pr <<<"$spec"
  bin="$ONEESAN_BUILD_DIR/orbitcta_prectx_${mode}_n${N}"
  N="$N" ARCH="$ARCH" OUT="$bin" DIRECTGATHER64=1 \
    DIRECTGATHER_SPARSE64="$DIRECTGATHER_SPARSE64" \
    PRECTX_FORWARD="$pf" PRECTX_REVERSE="$pr" \
    ORBITCTA_COL_ILP="$COL_ILP" PAIR_MLP=0 CPASYNC_PAIR=0 \
    RANKFORMULA_MLP_WINDOW4="$WINDOW4" PM_ACCUM="$PM_ACCUM" PTXAS_VERBOSE=1 \
    bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta.sh" \
    >"$LOGDIR/$mode.build.out" 2>"$LOGDIR/$mode.build.err"
  python3 "$PARSER" "$LOGDIR/$mode.build.err" --label "$mode" >>"$RESOURCE" || true

  for ((rep=1; rep<=REPEATS; ++rep)); do
    so="$LOGDIR/${mode}_r${rep}.out"; se="$LOGDIR/${mode}_r${rep}.err"
    BUCKET_THREADS="$THREADS" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" \
    BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" \
    BUCKET_ORBITCTA_GRID_Y="$ORBIT_GY" \
      "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"
    line="$(grep '^residue=' "$so" | tail -n1 || true)"
    [[ -n "$line" ]] || { echo "$mode missing residue" >&2; exit 3; }
    residue="$(field residue "$line")"
    [[ "$residue" == "$EXPECT" ]] || { echo "$mode residue mismatch got=$residue expected=$EXPECT" >&2; exit 4; }
    detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$mode" "$pf" "$pr" "$rep" "$residue" "$(field wall_s "$line")" \
      "$(field forward_high_s "$detail")" "$(field reverse_high_s "$detail")" \
      "$(field forward_low_s "$detail")" "$(field reverse_low_s "$detail")" \
      "$(field transpose_s "$detail")" >>"$RESULT"
  done

  # Every device builds a full pointer-valued context table because peer row
  # addresses differ by device. Counts should be identical across owners. Use
  # the first line for per-GPU resident bytes and record line count as a sanity check.
  python3 - "$mode" "$LOGDIR/${mode}_r1.err" >>"$META" <<'PY'
import re,sys
mode,path=sys.argv[1:]
lines=[]
for line in open(path,encoding='utf-8',errors='replace'):
    if 'p10dc_prectx_high fixed_owner=' not in line: continue
    d={k:int(v) for k,v in re.findall(r'(closure_context_bytes|fwd_nn|fwd_nrnl|rev_nn|rev_nr|rev_nl)=([0-9]+)',line)}
    lines.append(d)
if not lines:
    print('\t'.join([mode,'0','0','0','0','0','0','0','0']))
else:
    d=lines[0]
    c=d.get('closure_context_bytes',0)
    counts=[d.get(k,0) for k in ('fwd_nn','fwd_nrnl','rev_nn','rev_nr','rev_nl')]
    resident=c*sum(counts)
    print('\t'.join(map(str,[mode,len(lines),c,*counts,resident])))
PY
done

python3 - "$RESULT" <<'PY'
import csv,statistics,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
by={}
for r in rows: by.setdefault(r['mode'],[]).append(r)
out=[]
for mode,g in by.items():
    z={k:statistics.median(float(r[k]) for r in g)
       for k in ('wall_s','forward_high_s','reverse_high_s','forward_low_s','reverse_low_s','transpose_s')}
    z['high']=z['forward_high_s']+z['reverse_high_s']
    out.append((z['wall_s'],mode,z))
for _,mode,z in sorted(out):
    print(mode,f"wall={z['wall_s']:.6f}",f"high={z['high']:.6f}",
          f"fh={z['forward_high_s']:.6f}",f"rh={z['reverse_high_s']:.6f}")
q={m:z for _,m,z in out}
base=q.get('none')
if base:
    for _,mode,z in sorted(out):
        print(f"speedup_vs_none {mode} wall={base['wall_s']/z['wall_s']:.6f} "
              f"high={base['high']/z['high']:.6f} "
              f"fh={base['forward_high_s']/z['forward_high_s']:.6f} "
              f"rh={base['reverse_high_s']/z['reverse_high_s']:.6f}")
print('BEST',sorted(out)[0][1],f"wall={sorted(out)[0][0]:.6f}")
PY

echo "result=$RESULT ptxas=$RESOURCE prectx=$META directgather64=1 sparse64=$DIRECTGATHER_SPARSE64 geometry=t${THREADS}-y${ORBIT_GY}" >&2
