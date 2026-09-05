#pragma once

#include "ramstream32_bucket_low_rankformula_nometa_directmap.cuh"

#if !P10DC_RANKFORMULA_NOMETA_DIRECTMAP
#error "depth-major directgather requires DIRECTMAP"
#endif
#if !P10DC_RANKFORMULA_DIRECTGATHER
#error "depth-major directgather requires DIRECTGATHER"
#endif
#if !P10DC_RANKFORMULA_DIRECTGATHER_DEPTHMAJOR
#error "depth-major directgather header requires DEPTHMAJOR=1"
#endif

// The base direct-map table builder now emits DIRECTGATHER descriptors directly
// in [height][depth][rank] order when DEPTHMAJOR=1.  Keep this compatibility
// alias for the B300 wiring, but do not allocate a second rank-major table or
// launch a device transpose.  Consecutive warp lanes therefore read consecutive
// uint4 descriptors (16-byte lane stride instead of 13*16=208 bytes).
using BucketFusedDirectHighRowsRankFormulaNometa4DirectMapDepthMajorTables =
    BucketFusedDirectHighRowsRankFormulaNometa4DirectMapTables;
