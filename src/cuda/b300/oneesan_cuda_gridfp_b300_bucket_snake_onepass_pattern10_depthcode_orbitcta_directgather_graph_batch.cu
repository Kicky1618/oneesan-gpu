#ifndef BUCKET_SNAKE_REVERSE_FUSED
#define BUCKET_SNAKE_REVERSE_FUSED 1
#endif

#include "../gridfp/ramstream32_bucket_orbit_closure_pattern10_depthcode_orbitcta_delta_direct_affine_rankformula_nometa4_abstract_graph.cuh"
#include "../gridfp/ramstream32_reverse_build_release.hpp"
#if P10DC_RANKFORMULA_DIRECTGATHER64
// sparse64 is an extension of the directgather64 ABI/runtime helpers, not a
// replacement header. Keep the base definitions visible before the sparse
// table/runtime extension is instantiated.
#include "../gridfp/ramstream32_bucket_low_rankformula_directgather64.cuh"
#if P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64
#include "../gridfp/ramstream32_bucket_low_rankformula_directgather_sparse64.cuh"
#endif
#else
#include "../gridfp/ramstream32_bucket_low_rankformula_nometa_directmap_depthmajor.cuh"
#endif

#define BSN_REVERSE_FUSED_TABLES_TYPE ReverseBucketZeroTables
#define build_reverse_bucket_atomic build_reverse_bucket_atomic_release_inputs
#if P10DC_RANKFORMULA_DIRECTGATHER64
#if P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64
using P10DCOrbitSelectedFusedDeviceTables = BucketFusedDirectHighRowsRankFormulaNometa4DirectGatherSparse64Tables;
#else
using P10DCOrbitSelectedFusedDeviceTables = BucketFusedDirectHighRowsRankFormulaNometa4DirectGather64Tables;
#endif
#else
using P10DCOrbitSelectedFusedDeviceTables = BucketFusedDirectHighRowsRankFormulaNometa4DirectMapDepthMajorTables;
#endif
#if P10DC_RANKFORMULA_PRECTX_FORWARD || P10DC_RANKFORMULA_PRECTX_REVERSE
#define BucketFusedDeviceTables BucketFusedPrecomputedHighCtxTables<P10DCOrbitSelectedFusedDeviceTables>
// build_reverse_split54(..., true) releases reverse.atomic.high_orbit before
// the generic memory preflight executes.  The exact resident reverse stream is
// the split54 payload retained by rattach, which is already in scope where this
// macro expands.  Count that table rather than the now-empty legacy vector.
#define BSN_GRAPH_BATCH_EXTRA_METADATA_BYTES(borbit,reverse) \
    (sizeof(P10DCHighClosurePreCtx) * \
     (size_t(P10DC_RANKFORMULA_PRECTX_FORWARD) * ((borbit).high_nn.size() + (borbit).high_nrnl.size()) + \
      size_t(P10DC_RANKFORMULA_PRECTX_REVERSE) * \
          (rattach.split.high.nn.size() + rattach.split.high.nr.size() + rattach.split.high.nl.size())))
#else
#define BucketFusedDeviceTables P10DCOrbitSelectedFusedDeviceTables
#endif
#define BucketForwardOrbitClosureAttachHost BucketForwardPattern10DepthCodeHost
#define BucketReverseOrbitClosureAttachHost BucketReversePattern10DepthCodeHost
#define BucketForwardOrbitClosureAttachDeviceTables BucketForwardPattern10DepthCodeDeviceTables
#define BucketReverseOrbitClosureAttachDeviceTables BucketReversePattern10DepthCodeDeviceTables
#define build_bucket_forward_orbit_closure_attach build_bucket_forward_pattern10_depthcode_placeholder
#define build_bucket_reverse_orbit_closure_attach_checked build_bucket_reverse_pattern10_depthcode_zero_checked
#define BucketOnePassGraphs BucketPattern10DepthCodeOrbitCtaDirectAffineRankFormulaNometa4AbstractGraphs
#define bucket_onepass_graph_sync_devices bucket_pattern10_depthcode_orbitcta_rankformula_nometa4_abstract_graph_sync_devices

#include "oneesan_cuda_gridfp_b300_bucket_snake_onepass_graph_batch.cu"
