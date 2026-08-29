#ifndef BUCKET_SNAKE_REVERSE_FUSED
#define BUCKET_SNAKE_REVERSE_FUSED 1
#endif
#include "../gridfp/ramstream32_bucket_orbit_closure_pattern10_depthcode_warpstriped_delta_direct_affine_rankformula_nometa4_abstract_graph.cuh"
#include "../gridfp/ramstream32_reverse_build_release.hpp"
#if P10DC_RANKFORMULA_NOMETA_DIRECTMAP
#if P10DC_RANKFORMULA_DIRECTGATHER64
#include "../gridfp/ramstream32_bucket_low_rankformula_directgather64.cuh"
#if P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64
#include "../gridfp/ramstream32_bucket_low_rankformula_directgather_sparse64.cuh"
#endif
#if P10DC_RANKFORMULA_DIRECTGATHER_SORTED
#include "../gridfp/ramstream32_bucket_low_rankformula_directgather64_sorted.cuh"
#endif
#elif P10DC_RANKFORMULA_DIRECTGATHER_SORTED
#include "../gridfp/ramstream32_bucket_low_rankformula_nometa_directmap_sorted.cuh"
#elif P10DC_RANKFORMULA_DIRECTGATHER_DEPTHMAJOR
#include "../gridfp/ramstream32_bucket_low_rankformula_nometa_directmap_depthmajor.cuh"
#else
#include "../gridfp/ramstream32_bucket_low_rankformula_nometa_directmap.cuh"
#endif
#endif

#define BSN_REVERSE_FUSED_TABLES_TYPE ReverseBucketZeroTables
#define build_reverse_bucket_atomic build_reverse_bucket_atomic_release_inputs
#if P10DC_RANKFORMULA_NOMETA_DIRECTMAP
#if P10DC_RANKFORMULA_DIRECTGATHER64
#if P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64
#if P10DC_RANKFORMULA_DIRECTGATHER_SORTED
using P10DCSelectedFusedDeviceTables = BucketFusedDirectHighRowsRankFormulaNometa4DirectGatherSparse64SortedTables;
#else
using P10DCSelectedFusedDeviceTables = BucketFusedDirectHighRowsRankFormulaNometa4DirectGatherSparse64Tables;
#endif
#else
#if P10DC_RANKFORMULA_DIRECTGATHER_SORTED
using P10DCSelectedFusedDeviceTables = BucketFusedDirectHighRowsRankFormulaNometa4DirectGather64SortedTables;
#else
using P10DCSelectedFusedDeviceTables = BucketFusedDirectHighRowsRankFormulaNometa4DirectGather64Tables;
#endif
#endif
#elif P10DC_RANKFORMULA_DIRECTGATHER_SORTED
using P10DCSelectedFusedDeviceTables = BucketFusedDirectHighRowsRankFormulaNometa4DirectMapSortedTables;
#elif P10DC_RANKFORMULA_DIRECTGATHER_DEPTHMAJOR
using P10DCSelectedFusedDeviceTables = BucketFusedDirectHighRowsRankFormulaNometa4DirectMapDepthMajorTables;
#else
using P10DCSelectedFusedDeviceTables = BucketFusedDirectHighRowsRankFormulaNometa4DirectMapTables;
#endif
#else
using P10DCSelectedFusedDeviceTables = BucketFusedDirectHighRowsRankFormulaNometa4Tables;
#endif
#if P10DC_RANKFORMULA_PRECTX_FORWARD || P10DC_RANKFORMULA_PRECTX_REVERSE
#if P10DC_RANKFORMULA_PRECTX_COMPACT
#define BucketFusedDeviceTables BucketFusedCompactPrecomputedHighCtxTables<P10DCSelectedFusedDeviceTables>
#define P10DC_WARP_PRECTX_BYTES sizeof(P10DCHighClosureCompactPreCtx)
#else
#define BucketFusedDeviceTables BucketFusedPrecomputedHighCtxTables<P10DCSelectedFusedDeviceTables>
#define P10DC_WARP_PRECTX_BYTES sizeof(P10DCHighClosurePreCtx)
#endif
// Reverse pattern10-depthcode attach replaces the legacy atomic HIGH orbit
// vector with resident split54 NN/NR/NL streams. Count those exact streams for
// the HBM preflight; reverse.atomic.high_orbit may already have been released.
#define BSN_GRAPH_BATCH_EXTRA_METADATA_BYTES(borbit,reverse) \
    (P10DC_WARP_PRECTX_BYTES * \
     (size_t(P10DC_RANKFORMULA_PRECTX_FORWARD) * ((borbit).high_nn.size() + (borbit).high_nrnl.size()) + \
      size_t(P10DC_RANKFORMULA_PRECTX_REVERSE) * \
          (rattach.split.high.nn.size() + rattach.split.high.nr.size() + rattach.split.high.nl.size())))
#else
#define BucketFusedDeviceTables P10DCSelectedFusedDeviceTables
#endif
#define BucketForwardOrbitClosureAttachHost BucketForwardPattern10DepthCodeHost
#define BucketReverseOrbitClosureAttachHost BucketReversePattern10DepthCodeHost
#define BucketForwardOrbitClosureAttachDeviceTables BucketForwardPattern10DepthCodeDeviceTables
#define BucketReverseOrbitClosureAttachDeviceTables BucketReversePattern10DepthCodeDeviceTables
#define build_bucket_forward_orbit_closure_attach build_bucket_forward_pattern10_depthcode_placeholder
#define build_bucket_reverse_orbit_closure_attach_checked build_bucket_reverse_pattern10_depthcode_zero_checked
#define BucketOnePassGraphs BucketPattern10DepthCodeWarpStripedDeltaDirectAffineRankFormulaNometa4AbstractGraphs
#define bucket_onepass_graph_sync_devices bucket_pattern10_depthcode_warpstriped_delta_direct_affine_rankformula_nometa4_abstract_graph_sync_devices

#include "oneesan_cuda_gridfp_b300_bucket_snake_onepass_graph_batch.cu"
