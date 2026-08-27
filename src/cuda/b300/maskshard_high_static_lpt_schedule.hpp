#pragma once

#ifdef MASKSHARD_HIGH_CAP_LPT_CAPTURE_AFFINITY
// v0.77 layers capture affinity on the frozen v0.76 scheduler. Rename only the
// cap-LPT wrapper types/functions while importing v0.76, then provide a derived
// state that can replace equal-load GPU identities without changing LPT loads.
#undef MASKSHARD_HIGH_CAP_LPT_CAPTURE_AFFINITY
#define MaskShardHighCapLptState MaskShardHighCapLptStateV076
#define MaskShardHighCapLptJobsProxy MaskShardHighCapLptJobsProxyV076
#define MaskShardHighCapLptSchedule MaskShardHighCapLptScheduleV076
#define maskshard_build_high_cap_lpt_schedule maskshard_build_high_cap_lpt_schedule_v076
#define maskshard_prepare_highclosure_rowdepth_compact_cap_lpt \
        maskshard_prepare_highclosure_rowdepth_compact_cap_lpt_v076
#define maskshard_set_row_depth_fblock_io_row_cap_lpt \
        maskshard_set_row_depth_fblock_io_row_cap_lpt_v076
#include "maskshard_high_static_lpt_schedule_v076.hpp"
#undef MaskShardHighCapLptState
#undef MaskShardHighCapLptJobsProxy
#undef MaskShardHighCapLptSchedule
#undef maskshard_build_high_cap_lpt_schedule
#undef maskshard_prepare_highclosure_rowdepth_compact_cap_lpt
#undef maskshard_set_row_depth_fblock_io_row_cap_lpt
[[maybe_unused]] static constexpr auto G_MS_HIGH_CAP_LPT_V076_ROW_SETTER_KEEP =
    &maskshard_set_row_depth_fblock_io_row_cap_lpt_v076;
#define MASKSHARD_HIGH_CAP_LPT_CAPTURE_AFFINITY 1
#ifdef maskshard_build_high_static_lpt_schedule
#undef maskshard_build_high_static_lpt_schedule
#endif
#ifdef maskshard_prepare_highclosure_rowdepth_compact
#undef maskshard_prepare_highclosure_rowdepth_compact
#endif
#ifdef maskshard_set_row_depth_fblock_io_row
#undef maskshard_set_row_depth_fblock_io_row
#endif

#ifdef MASKSHARD_HIGH_CAP_LPT_LOCALITY_GUARD
// v0.78 freezes the v0.77 policy in the same way: import it under V077 names,
// then derive the exact peer-I/O Pareto guard without changing v0.77 binaries.
#define MaskShardHighCapLptState MaskShardHighCapLptStateV077
#define MaskShardHighCapLptJobsProxy MaskShardHighCapLptJobsProxyV077
#define MaskShardHighCapLptSchedule MaskShardHighCapLptScheduleV077
#define maskshard_build_high_cap_lpt_schedule maskshard_build_high_cap_lpt_schedule_v077
#define maskshard_prepare_highclosure_rowdepth_compact_cap_lpt \
        maskshard_prepare_highclosure_rowdepth_compact_cap_lpt_v077
#define maskshard_set_row_depth_fblock_io_row_cap_lpt \
        maskshard_set_row_depth_fblock_io_row_cap_lpt_v077
#include "maskshard_high_cap_lpt_capture_affinity.hpp"
#undef MaskShardHighCapLptState
#undef MaskShardHighCapLptJobsProxy
#undef MaskShardHighCapLptSchedule
#undef maskshard_build_high_cap_lpt_schedule
#undef maskshard_prepare_highclosure_rowdepth_compact_cap_lpt
#undef maskshard_set_row_depth_fblock_io_row_cap_lpt
[[maybe_unused]] static constexpr auto G_MS_HIGH_CAP_LPT_V077_ROW_SETTER_KEEP =
    &maskshard_set_row_depth_fblock_io_row_cap_lpt_v077;
#ifdef maskshard_build_high_static_lpt_schedule
#undef maskshard_build_high_static_lpt_schedule
#endif
#ifdef maskshard_prepare_highclosure_rowdepth_compact
#undef maskshard_prepare_highclosure_rowdepth_compact
#endif
#ifdef maskshard_set_row_depth_fblock_io_row
#undef maskshard_set_row_depth_fblock_io_row
#endif

#ifdef MASKSHARD_HIGH_GRAPH_MASK_SEQUENCE
// v0.79 freezes the complete v0.78 scheduler, then adds only the execution-side
// resident mask sequence. The selected per-cap job assignment is unchanged.
#define MaskShardHighCapLptState MaskShardHighCapLptStateV078
#define MaskShardHighCapLptJobsProxy MaskShardHighCapLptJobsProxyV078
#define MaskShardHighCapLptSchedule MaskShardHighCapLptScheduleV078
#define maskshard_build_high_cap_lpt_schedule maskshard_build_high_cap_lpt_schedule_v078
#define maskshard_prepare_highclosure_rowdepth_compact_cap_lpt \
        maskshard_prepare_highclosure_rowdepth_compact_cap_lpt_v078
#define maskshard_set_row_depth_fblock_io_row_cap_lpt \
        maskshard_set_row_depth_fblock_io_row_cap_lpt_v078
#include "maskshard_high_cap_lpt_locality_guard.hpp"
#undef MaskShardHighCapLptState
#undef MaskShardHighCapLptJobsProxy
#undef MaskShardHighCapLptSchedule
#undef maskshard_build_high_cap_lpt_schedule
#undef maskshard_prepare_highclosure_rowdepth_compact_cap_lpt
#undef maskshard_set_row_depth_fblock_io_row_cap_lpt
[[maybe_unused]] static constexpr auto G_MS_HIGH_CAP_LPT_V078_ROW_SETTER_KEEP =
    &maskshard_set_row_depth_fblock_io_row_cap_lpt_v078;
#ifdef maskshard_build_high_static_lpt_schedule
#undef maskshard_build_high_static_lpt_schedule
#endif
#ifdef maskshard_prepare_highclosure_rowdepth_compact
#undef maskshard_prepare_highclosure_rowdepth_compact
#endif
#ifdef maskshard_set_row_depth_fblock_io_row
#undef maskshard_set_row_depth_fblock_io_row
#endif
#include "maskshard_high_graph_mask_sequence.cuh"
#include "maskshard_high_graph_mask_sequence_schedule.hpp"
#else
#include "maskshard_high_cap_lpt_locality_guard.hpp"
#endif

#else
#include "maskshard_high_cap_lpt_capture_affinity.hpp"
#endif

#else
#include "maskshard_high_static_lpt_schedule_v076.hpp"
#endif
