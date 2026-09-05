#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, re, struct, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'solve'))
from path_bound import PRIMES

DIMS = [1107,1640,1428,888,420,152,42,8,1]
SCHEMA = 'oneesan-row8-raw-quotient-v2'

CRITICAL = [
    'src/cuda/b300/row8_pivots_w19.bin',
    'work/formal-probes/raw_wfa_r8.ck',
    'work/formal-probes/dual-basis/Phi_h0_integer.bin',
    'work/formal-probes/dual-basis/Phi_h1_integer.bin',
    'work/formal-probes/dual-basis/Phi_h2_extra_mod1000000007.bin',
    'src/cuda/b300/row8_runtime_mod_builder.cuh',
    'src/cuda/b300/row8_mod_matrix_runtime.cuh',
    'src/cuda/b300/row8_tensor_init.cuh',
    'formal/OneesanFormal/ExplicitGapBasis.lean',
    'formal/OneesanFormal/ProductionSimilarity.lean',
    'formal/OneesanFormal/PathCutCrossing.lean',
    'formal/OneesanFormal/ProcessedStripCut.lean',
    'formal/OneesanFormal/MateCutSemantics.lean',
    'formal/OneesanFormal/MateRowLipschitz.lean',
    'formal/OneesanFormal/Row1Frontier.lean',
    'formal/OneesanFormal/ProductionRowSemantics.lean',
    'formal/OneesanFormal/MateMarkerStack.lean',
    'src/cpp/probes/gridfp_transition_semantic_abi.cpp',
    'src/cpp/probes/gridfp_transition_exhaustive_abi.cpp',
    'src/cpp/probes/row8_cap9_overflow_wfa.cpp',
    'scripts/tools/verify_row8_cap9_overflow.py',
    'src/cpp/probes/row8_raw_fixedpoint_verify.cpp',
    'src/cpp/probes/row8_upper_support_closure.cpp',
    'src/cpp/probes/row8_h3_R_phi2_closure.cpp',
    'src/cpp/probes/row8_integer_dual_basis.cpp',
    'src/cpp/probes/row8_integer_basis_rank_mod.cpp',
    'src/cpp/probes/row8_integer_dual_closure_mod.cpp',
    'src/cpp/probes/row8_final_functional_mod.cpp',
]
EVIDENCE = [
    'work/formal-probes/gridfp_transition_semantic_abi.log',
    'work/formal-probes/gridfp_transition_exhaustive_abi.log',
    'work/formal-probes/row8_raw_fixedpoint_verify.log',
    'work/formal-probes/row8_upper_support_closure_h3h5.log',
    'work/formal-probes/row8_upper_support_closure_h6h8.log',
    'work/formal-probes/row8_h3_R_phi2_closure.log',
    'work/formal-probes/row8_h4_relation_projection.log',
    'work/formal-probes/integer_dual_basis.log',
    'work/formal-probes/row8_integer_rank_all_primes.log',
    'work/formal-probes/row8_integer_closure_all_primes.log',
    'work/formal-probes/row8_final_functional_all_primes.log',
    'work/formal-probes/oneesan_formal_build_latest.log',
    'work/formal-probes/row8_cap9_overflow_wfa.log',
    'work/formal-probes/raw_wfa_r8_cap9_overflow.ck',
]

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def meta(rel: str):
    p=ROOT/rel
    return {'path':rel,'bytes':p.stat().st_size,'sha256':sha(p)}

def checkpoint_info(path: Path):
    # C++ CkHdr: char[8], uint32 ver/r/col/reserved, uint64 n/hash
    b=path.read_bytes()[:40]
    magic,ver,r,col,reserved,n,h=struct.unpack('<8sIIIIQQ',b)
    return {'magic':magic.rstrip(b'\0').decode(),'version':ver,'r':r,'col':col,'states':n,'payload_hash_fnv64':f'{h:016x}'}

def validate_logs() -> dict:
    out={}
    abi=(ROOT/'work/formal-probes/gridfp_transition_semantic_abi.log').read_text()
    assert re.search(r'gridfp_transition_semantic_abi cases=20 bad=0 exact=1', abi)
    out['production_case_table_cpp_checks']=20
    exabi=(ROOT/'work/formal-probes/gridfp_transition_exhaustive_abi.log').read_text()
    mabi=re.search(r'gridfp_transition_exhaustive_abi maxW=(\d+) checked=(\d+) valid=(\d+) blocked=(\d+) bad=0 exact=1', exabi)
    assert mabi and int(mabi.group(1)) >= 14 and int(mabi.group(2)) >= 89680671
    out['production_case_table_exhaustive']={'max_width':int(mabi.group(1)),'cases':int(mabi.group(2))}
    p=(ROOT/'work/formal-probes/row8_raw_fixedpoint_verify.log').read_text()
    m=re.search(r'r=8 col=(\d+) states=(\d+) input_hash=([0-9a-f]+) step_hash=([0-9a-f]+) same=(\d+)',p)
    assert m and m.group(1)=='29' and m.group(2)=='3558574' and m.group(3)==m.group(4) and m.group(5)=='1'
    out['fixedpoint']={'col':29,'states':3558574,'fnv64':m.group(3)}

    expected={(3,0),(3,2),(4,0),(4,1),(4,2),(5,0),(5,1),(5,2),(6,0),(6,1),(6,2),(7,0),(7,1),(7,2),(8,0),(8,1)}
    got=set()
    for rel in ['work/formal-probes/row8_upper_support_closure_h3h5.log','work/formal-probes/row8_upper_support_closure_h6h8.log']:
        text=(ROOT/rel).read_text()
        for h,a,oh,pb,ex in re.findall(r'h=(\d+) sym=(\d+).*?outside_hit=(\d+) pair_bad=(\d+).*?exact=(\d+)',text):
            assert oh=='0' and pb=='0' and ex=='1'; got.add((int(h),int(a)))
    assert got==expected, (got^expected)
    out['upper_integer_closure_blocks']=len(got)

    h3=(ROOT/'work/formal-probes/row8_h3_R_phi2_closure.log').read_text()
    assert re.search(r'outside_bad=0 pair_bad=0 .*exact=1',h3)
    h4=(ROOT/'work/formal-probes/row8_h4_relation_projection.log').read_text()
    assert re.search(r'Arel_rank=420/420 .*bad=0 .*exact=1',h4)
    ib=(ROOT/'work/formal-probes/integer_dual_basis.log').read_text()
    assert 'h1_integer seed=849 final=1640' in ib and 'h0_integer seed=714 final=1107' in ib and 'integer_dual_basis_exact_data=1' in ib

    rank=(ROOT/'work/formal-probes/row8_integer_rank_all_primes.log').read_text().splitlines()
    ranks={p:{0:False,1:False,2:False} for p in PRIMES}
    for line in rank:
        m=re.match(r'rank h=(\d) p=(\d+) rank=(\d+)/(\d+)',line)
        if m:
            h,pv,a,b=map(int,m.groups());
            if pv in ranks and h in ranks[pv]: ranks[pv][h]=(a==b==DIMS[h])
    assert all(all(x.values()) for x in ranks.values())

    def parse_prime_exact(rel, token):
        lines=(ROOT/rel).read_text().splitlines(); seen={}
        for line in lines:
            m=re.search(r'p=(\d+).*exact=(\d+)',line)
            if m: seen[int(m.group(1))]=(m.group(2)=='1' and token in line)
        assert set(seen)==set(PRIMES) and all(seen.values())
        return len(seen)
    out['production_prime_rank_count']=len(ranks)
    out['production_prime_closure_count']=parse_prime_exact('work/formal-probes/row8_integer_closure_all_primes.log','bad=0')
    out['production_prime_final_count']=parse_prime_exact('work/formal-probes/row8_final_functional_all_primes.log','h0N=1 h1R=1')

    fb=(ROOT/'work/formal-probes/oneesan_formal_build_latest.log').read_text()
    assert 'Build completed successfully' in fb
    out['lean_build']='pass'

    caplog=(ROOT/'work/formal-probes/row8_cap9_overflow_wfa.log').read_text()
    finals={}
    for c,a,t in re.findall(r'final_check col=(\d+) accept=(\d+) tested=(\d+)', caplog):
        finals[int(c)]=(int(a),int(t))
    m=re.search(r'cap9_all_widths_no_overflow_accept=1 fixed_col=(\d+) states=(\d+) overflow_states=(\d+) exact=1', caplog)
    assert m, 'missing cap9 exact summary'
    fixed, states, overflow = map(int,m.groups())
    assert fixed==38 and states==12657343 and overflow==9098769
    assert set(finals)==set(range(1,fixed+1))
    assert all(a==0 for a,_ in finals.values())
    first_overflow=next(c for c in range(1,fixed+1) if re.search(rf'r=8 col={c} states=\d+ overflow=([1-9]\d*)', caplog))
    assert first_overflow==8
    capck=checkpoint_info(ROOT/'work/formal-probes/raw_wfa_r8_cap9_overflow.ck')
    assert capck['magic']=='R9OVERF' and capck['version']==2 and capck['r']==8 and capck['col']==fixed and capck['states']==states
    out['cap9_overflow_language']={
        'fixed_col':fixed,'states':states,'overflow_states':overflow,
        'first_overflow_col':first_overflow,'all_final_checks_zero':True,
        'checkpoint':capck,
    }
    return out

def generate():
    for x in CRITICAL+EVIDENCE:
        if not (ROOT/x).is_file(): raise SystemExit(f'missing {x}')
    evidence=validate_logs()
    ck=checkpoint_info(ROOT/'work/formal-probes/raw_wfa_r8.ck')
    assert ck['r']==8 and ck['col']==29 and ck['states']==3558574
    doc={
        'schema':SCHEMA,
        'row':8,
        'dimensions':DIMS,
        'dimension_sum':sum(DIMS),
        'production_primes':PRIMES,
        'checkpoint':ck,
        'critical_files':[meta(x) for x in CRITICAL],
        'evidence_files':[meta(x) for x in EVIDENCE],
        'evidence_summary':evidence,
        'proof_status':{
            'raw_fixedpoint_complete':True,
            'raw_quotient_transition_closed':True,
            'right_boundary_functionals_closed':True,
            'production_primes_checked':len(PRIMES),
            'production_similarity_linear_algebra_lean':True,
            'path_cut_height_bound_lean':True,
            'processed_strip_cut_bound_lean':True,
            'mate_cut_semantic_interface_lean':True,
            'marker_occurrence_stack_lean':True,
            'production_row_lipschitz_lean':True,
            'production_case_rewrite_table_lean':True,
            'production_case_table_cpp_checked':True,
            'production_case_table_cpp_exhaustive':True,
            'raw_cap8_language_complete_via_cap9_overflow':True,
            # This is deliberately false until the C++ packed-state transition semantics
            # are connected to PathCutStackWitness in the formal model.
            'implementation_semantics_bridge_complete':False,
            'exact_admissible':False,
        },
    }
    return doc

def verify(path: Path):
    d=json.loads(path.read_text())
    if d.get('schema')!=SCHEMA: raise SystemExit('bad schema')
    if d.get('dimensions')!=DIMS or d.get('dimension_sum')!=5686: raise SystemExit('bad dimensions')
    if d.get('production_primes')!=PRIMES: raise SystemExit('prime table mismatch')
    for item in d['critical_files']+d['evidence_files']:
        p=ROOT/item['path']
        if not p.is_file() or p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:
            raise SystemExit(f"fingerprint mismatch: {item['path']}")
    now=validate_logs()
    if now!=d['evidence_summary']: raise SystemExit('evidence summary mismatch')
    if checkpoint_info(ROOT/'work/formal-probes/raw_wfa_r8.ck')!=d['checkpoint']: raise SystemExit('checkpoint header mismatch')
    st=d['proof_status']
    numerical=all(st[k] for k in ['raw_fixedpoint_complete','raw_quotient_transition_closed','right_boundary_functionals_closed','production_similarity_linear_algebra_lean','path_cut_height_bound_lean','processed_strip_cut_bound_lean','mate_cut_semantic_interface_lean','marker_occurrence_stack_lean','production_row_lipschitz_lean','production_case_rewrite_table_lean','production_case_table_cpp_checked','production_case_table_cpp_exhaustive','raw_cap8_language_complete_via_cap9_overflow']) and st['production_primes_checked']==len(PRIMES)
    admissible=numerical and st.get('implementation_semantics_bridge_complete',False)
    if bool(st.get('exact_admissible'))!=admissible: raise SystemExit('exact_admissible inconsistent with proof status')
    print(f"row8 certificate verified numerical={int(numerical)} semantics_bridge={int(st.get('implementation_semantics_bridge_complete',False))} exact_admissible={int(admissible)} primes={len(PRIMES)} dims={sum(DIMS)}")
    return 0

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--generate',action='store_true'); ap.add_argument('--verify',type=Path); ap.add_argument('--out',type=Path,default=ROOT/'work/formal-probes/row8_raw_quotient_cert.json'); a=ap.parse_args()
    if a.generate:
        d=generate(); a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n'); print(a.out)
    elif a.verify: return verify(a.verify)
    else: ap.error('use --generate or --verify')
if __name__=='__main__': raise SystemExit(main())
