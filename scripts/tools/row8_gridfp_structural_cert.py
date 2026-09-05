#!/usr/bin/env python3
from __future__ import annotations
import argparse, ctypes, hashlib, json, math, re, struct, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'solve'))
from path_bound import PRIMES

SCHEMA='oneesan-row8-gridfp-structural-v2'
ROW=8
DIMS=[1107,1640,1428,888,420,152,42,8,1]
CACHE='src/cuda/b300/row8_structural_int_v1.bin'
CRITICAL=[
    'src/common/gridfp_transition.hpp',
    'src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_factorized_reverse2_row8tensor_batch.cu',
    'src/cuda/b300/row8_tensor_init.cuh',
    'src/cuda/b300/row8_structural_tensor_init.cuh',
    CACHE,
    'scripts/tools/row8_gridfp_structural_cert.py',
    'formal/OneesanFormal/ProductionDecisionBound.lean',
]

def sha(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def meta(rel:str):
    p=ROOT/rel
    return {'path':rel,'bytes':p.stat().st_size,'sha256':sha(p)}

class SH(ctypes.Structure):
    _fields_=[('magic',ctypes.c_char*8),('version',ctypes.c_uint32),('r',ctypes.c_uint32),('dims',ctypes.c_uint32*9),('total_nz',ctypes.c_uint64),('fnv_hash',ctypes.c_uint64)]
class VH(ctypes.Structure):
    _fields_=[('tag',ctypes.c_uint32),('sym',ctypes.c_uint32),('h',ctypes.c_uint32),('nnz',ctypes.c_uint32)]
class BH(ctypes.Structure):
    _fields_=[('sym',ctypes.c_uint32),('h',ctypes.c_uint32),('h2',ctypes.c_uint32),('rows',ctypes.c_uint32),('cols',ctypes.c_uint32),('nnz',ctypes.c_uint32)]

def structural_norms():
    b=(ROOT/CACHE).read_bytes(); off=0
    h=SH.from_buffer_copy(b[:ctypes.sizeof(SH)]);off=ctypes.sizeof(SH)
    if bytes(h.magic).rstrip(b'\0')!=b'R8STR01' or h.version!=1 or h.r!=8 or list(h.dims)!=DIMS:
        raise SystemExit('bad structural cache header')
    alpha_max=beta_max=0
    for _ in range(5):
        v=VH.from_buffer_copy(b[off:off+ctypes.sizeof(VH)]);off+=ctypes.sizeof(VH)
        vals=[]
        for _ in range(v.nnz):
            off+=2
            a=struct.unpack_from('<b',b,off)[0];off+=1;vals.append(abs(a))
        m=max(vals,default=0)
        if v.tag==1:alpha_max=max(alpha_max,m)
        elif v.tag==2:beta_max=max(beta_max,m)
        else:raise SystemExit('bad structural vector tag')
    max_row=max_col=0; nz=0
    for _ in range(25):
        q=BH.from_buffer_copy(b[off:off+ctypes.sizeof(BH)]);off+=ctypes.sizeof(BH)
        if q.rows!=DIMS[q.h] or q.cols!=DIMS[q.h2]: raise SystemExit('bad structural block dims')
        rp=list(struct.unpack_from('<'+'I'*(q.rows+1),b,off));off+=4*(q.rows+1)
        ci=list(struct.unpack_from('<'+'H'*q.nnz,b,off));off+=2*q.nnz
        cv=list(struct.unpack_from('<'+'b'*q.nnz,b,off));off+=q.nnz
        rs=[sum(abs(cv[e]) for e in range(rp[i],rp[i+1])) for i in range(q.rows)]
        cs=[0]*q.cols
        for j,a in zip(ci,cv):cs[j]+=abs(a)
        max_row=max(max_row,max(rs,default=0));max_col=max(max_col,max(cs,default=0));nz+=q.nnz
    if off!=len(b) or nz!=h.total_nz: raise SystemExit('bad structural cache body')
    return {'alpha_max':alpha_max,'beta_max':beta_max,'max_abs_row_sum':max_row,'max_abs_col_sum':max_col,'nonzeros':nz}

def bounds(width:int):
    if width<2: raise SystemExit('width must be >=2')
    nrm=structural_norms()
    hi=width-width//2; lo=width//2
    prefix=nrm['alpha_max']*(nrm['max_abs_col_sum']**max(0,hi-1))
    suffix=nrm['beta_max']*(nrm['max_abs_row_sum']**max(0,lo-1))
    structural=max(DIMS)*prefix*suffix
    edges=ROW*(2*width-1)
    # The production baseline is deterministic once its horizontal include/exclude
    # decisions are fixed.  There are only r(w-1) such binary choices through row r;
    # blocked cells merely force a choice and cannot increase the trace count.  See
    # ProductionDecisionBound.fiber_card_le.
    decision_bits=ROW*(width-1)
    baseline=1<<decision_bits
    difference=baseline+structural
    return nrm, {'prefix_abs':prefix,'suffix_abs':suffix,'structural_abs':structural,'processed_edges':edges,'baseline_decision_bits':decision_bits,'baseline_abs':baseline,'difference_abs':difference}

def required_primes(width:int):
    _,bd=bounds(width); P=1;k=0
    while P<=bd['difference_abs']:
        P*=PRIMES[k];k+=1
    return PRIMES[:k],P,bd

def parse_log(path:Path):
    text=path.read_text(); rows=[]
    pat=r'row8_cert_compare n=(\d+) width=(\d+) modulus=(\d+) gpus=(\d+) main_states=(\d+) mismatch=(\d+) exact=(\d+) wall_s=([0-9.eE+-]+)'
    for m in re.finditer(pat,text):
        n,w,p,g,states,mis,ex,wall=m.groups(); rows.append({'n':int(n),'width':int(w),'modulus':int(p),'gpus':int(g),'main_states':int(states),'mismatch':int(mis),'exact':int(ex),'wall_s':float(wall)})
    if not rows: raise SystemExit('no row8_cert_compare records')
    return rows

def load_provenance(path:Path, width:int):
    d=json.loads(path.read_text())
    if d.get('format')!='ONEESAN_BUILD_PROVENANCE_V2': raise SystemExit('bad comparison build provenance format')
    if d.get('source')!='src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_factorized_reverse2_row8tensor_batch.cu': raise SystemExit('wrong comparison build source')
    if f'-DTARGET_W={width}' not in d.get('compile_args',[]): raise SystemExit('comparison provenance TARGET_W mismatch')
    if not re.fullmatch(r'[0-9a-f]{64}', d.get('binary_sha256','')): raise SystemExit('bad comparison binary hash')
    # Verify every dependency/auxiliary dependency against the current source tree.
    for x in d.get('dependencies',[])+d.get('auxiliary_dependencies',[]):
        rp=x.get('path'); q=ROOT/rp
        size=x.get('size',x.get('bytes'))
        if not rp or not q.is_file() or (size is not None and q.stat().st_size!=size) or sha(q)!=x.get('sha256'):
            raise SystemExit('comparison provenance dependency mismatch: '+str(rp))
    # If the binary is still present, require its bytes to match too.  The snapshot
    # remains clean-clone verifiable after the build artifact is removed.
    bf=d.get('binary_file')
    candidates=[]
    if bf:
        candidates=[path.parent/bf, ROOT/'work/formal-probes'/bf, ROOT/'build'/bf]
    for q in candidates:
        if q.is_file():
            if q.stat().st_size!=d.get('binary_size') or sha(q)!=d.get('binary_sha256'):
                raise SystemExit('comparison binary fingerprint mismatch: '+str(q))
            break
    return d

def compact_provenance(d:dict):
    return {k:d[k] for k in [
        'format','source','git_commit','compiler_path','compiler_version','compile_args',
        'binary_file','binary_size','binary_sha256','dependencies','auxiliary_dependencies','provenance_sha256'
    ] if k in d}

def verify_provenance_snapshot(d:dict, width:int):
    if d.get('format')!='ONEESAN_BUILD_PROVENANCE_V2' or d.get('source')!='src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_factorized_reverse2_row8tensor_batch.cu':
        raise SystemExit('bad embedded comparison provenance')
    if f'-DTARGET_W={width}' not in d.get('compile_args',[]): raise SystemExit('embedded comparison TARGET_W mismatch')
    if not re.fullmatch(r'[0-9a-f]{64}', d.get('binary_sha256','')): raise SystemExit('bad embedded comparison binary hash')
    for x in d.get('dependencies',[])+d.get('auxiliary_dependencies',[]):
        rp=x.get('path'); q=ROOT/rp; size=x.get('size',x.get('bytes'))
        if not rp or not q.is_file() or (size is not None and q.stat().st_size!=size) or sha(q)!=x.get('sha256'):
            raise SystemExit('embedded comparison provenance dependency mismatch: '+str(rp))

def generate(log:Path, provenance:Path):
    rows=parse_log(log); n=rows[0]['n'];w=rows[0]['width'];states=rows[0]['main_states']
    prov=load_provenance(provenance,w)
    if w!=n+1: raise SystemExit('n/width mismatch')
    if any((r['n'],r['width'],r['main_states'])!=(n,w,states) for r in rows):raise SystemExit('mixed comparison log')
    if any(r['mismatch']!=0 or r['exact']!=1 for r in rows):raise SystemExit('nonzero comparison mismatch')
    seen=[r['modulus'] for r in rows]
    if len(seen)!=len(set(seen)):raise SystemExit('duplicate comparison prime')
    req,P,bd=required_primes(w)
    if seen!=req:raise SystemExit(f'comparison primes must be the first {len(req)} production primes; got {seen}')
    actual=math.prod(seen)
    if actual!=P or actual<=bd['difference_abs']:raise SystemExit('CRT product does not exceed deterministic difference bound')
    norms=structural_norms()
    doc={
      'schema':SCHEMA,'row':ROW,'n':n,'width':w,'main_states':states,
      'comparison_primes':seen,'prime_count':len(seen),'prime_product_bits':actual.bit_length(),
      'processed_strip_edges':bd['processed_edges'],
      'bounds':{k:(str(v) if isinstance(v,int) else v) for k,v in bd.items()},
      'bound_bits':{k:int(v).bit_length() for k,v in bd.items() if isinstance(v,int)},
      'structural_norms':norms,
      'integer_vector_equal':True,
      'crt_argument':'The production row-8 Grid-FP baseline is a deterministic execution once its r(w-1) horizontal include/exclude bits are fixed; blocked cells only force a decision and cannot add traces. ProductionDecisionBound.fiber_card_le formally bounds every target coefficient by 2^(r(w-1)) without assuming any path semantics. The structural integer automaton is independently bounded from its cache operator infinity norms. Their difference is divisible by the listed prime product and has absolute value strictly below that product, hence is zero over Z.',
      'critical_files':[meta(x) for x in CRITICAL],
      'evidence_log':{'path':str(log.relative_to(ROOT)) if log.is_relative_to(ROOT) else str(log),'bytes':log.stat().st_size,'sha256':sha(log)},
      'comparison_build_provenance':compact_provenance(prov),
      'comparison_provenance_file_sha256':sha(provenance),
      'records':rows,
    }
    return doc

def verify(path:Path):
    d=json.loads(path.read_text())
    if d.get('schema')!=SCHEMA or not d.get('integer_vector_equal'):raise SystemExit('bad certificate schema/status')
    for x in d['critical_files']:
        p=ROOT/x['path']
        if not p.is_file() or p.stat().st_size!=x['bytes'] or sha(p)!=x['sha256']:raise SystemExit('fingerprint mismatch: '+x['path'])
    lp=Path(d['evidence_log']['path']); lp=lp if lp.is_absolute() else ROOT/lp
    if not lp.is_file() or lp.stat().st_size!=d['evidence_log']['bytes'] or sha(lp)!=d['evidence_log']['sha256']:raise SystemExit('evidence log fingerprint mismatch')
    verify_provenance_snapshot(d.get('comparison_build_provenance',{}), int(d['width']))
    rows=parse_log(lp); req,P,bd=required_primes(int(d['width'])); actual=math.prod([r['modulus'] for r in rows])
    if any(r['mismatch']!=0 or r['exact']!=1 for r in rows): raise SystemExit('evidence comparison mismatch')
    if [r['modulus'] for r in rows]!=req or actual!=P or actual<=bd['difference_abs']: raise SystemExit('evidence CRT coverage mismatch')
    expected={
      'row':ROW,'n':rows[0]['n'],'width':rows[0]['width'],'main_states':rows[0]['main_states'],
      'comparison_primes':req,'prime_count':len(req),'prime_product_bits':actual.bit_length(),
      'processed_strip_edges':bd['processed_edges'],
      'bounds':{k:str(v) for k,v in bd.items()},
      'bound_bits':{k:int(v).bit_length() for k,v in bd.items()},
      'structural_norms':structural_norms(),'integer_vector_equal':True,'records':rows}
    for k,v in expected.items():
        if d.get(k)!=v: raise SystemExit('certificate mismatch: '+k)
    print(f"row8 Grid-FP/structural certificate verified n={d['n']} width={d['width']} states={d['main_states']} primes={d['prime_count']} product_bits={d['prime_product_bits']} processed_edges={d['processed_strip_edges']} decision_bits={d['bounds'].get('baseline_decision_bits')} difference_bits={d['bound_bits'].get('difference_abs')} integer_equal=1")

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--log',type=Path);ap.add_argument('--provenance',type=Path);ap.add_argument('--generate',action='store_true');ap.add_argument('--verify',type=Path);ap.add_argument('--out',type=Path);ap.add_argument('--show-required',type=int,metavar='WIDTH');a=ap.parse_args()
    if a.show_required is not None:
        ps,P,bd=required_primes(a.show_required);print(json.dumps({'width':a.show_required,'processed_edges':bd['processed_edges'],'difference_bound_bits':bd['difference_abs'].bit_length(),'primes':ps,'prime_product_bits':P.bit_length(),'structural_norms':structural_norms()},indent=2));return
    if a.generate:
        if not a.log or not a.provenance:ap.error('--generate requires --log and --provenance')
        d=generate(a.log,a.provenance);out=a.out or ROOT/f"work/formal-probes/row8_gridfp_structural_w{d['width']}.json";out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n');print(out);return
    if a.verify:verify(a.verify);return
    ap.error('use --show-required, --generate, or --verify')
if __name__=='__main__':main()
