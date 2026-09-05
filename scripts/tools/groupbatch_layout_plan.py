#!/usr/bin/env python3
"""Exact state-weighted lane distribution of the groupbatch scratch schedule.

This models the sorted whole-window factor partition, not GPU execution time.
Use the solver's GRIDFP_PLAN_ONLY output for the full device memory budget.
"""
import argparse
import json

from frontier_batch_capacity import capacity


def layout_plan(n, scratch_mib, max_lanes):
    counts = capacity(n, scratch_mib, 0)
    result = dict(n=n, scratch_mib=scratch_mib, max_lanes=max_lanes, windows=[])
    target = scratch_mib * 2**20

    def align(states):
        return (states + 63) // 64 * 64

    for window in counts['windows']:
        shapes = sorted(window['shapes'],
                        key=lambda s: 2*s['main_states']+s['blocked_states'], reverse=True)
        groups = [s for s in shapes for _ in range(s['masks'])]
        states_by_lanes = {lanes: 0 for lanes in (1, 2, 4, 8, 16, 32)}
        batches = 0
        peak = 0
        position = 0
        while position < len(groups):
            batch = []
            original_bytes = 0
            while position < len(groups):
                shape = groups[position]
                need = 4*(align(shape['main_states'])+align(shape['blocked_states']))
                if need > target:
                    raise ValueError('single group exceeds scratch budget')
                if batch and original_bytes+need > target:
                    break
                batch.append(shape)
                original_bytes += need
                position += 1
            used = 0
            i = 0
            while i < len(batch):
                last = i+1
                while last < len(batch) and batch[last]['occupied'] == batch[i]['occupied']:
                    last += 1
                lanes = min(max_lanes, 1 << ((last-i).bit_length()-1))
                shape = batch[i]
                states_by_lanes[lanes] += lanes*shape['states']
                used += 4*(align(lanes*shape['main_states'])+
                           align(lanes*shape['blocked_states']))
                i += lanes
            assert used <= original_bytes <= target
            peak = max(peak, used)
            batches += 1
        total = counts['main_states']+counts['blocked_states']
        assert sum(states_by_lanes.values()) == total
        result['windows'].append(dict(fixed_low=window['fixed_low'], batches=batches,
            peak_scratch_bytes=peak, states=total, states_by_lanes=states_by_lanes,
            interleaved_state_fraction=1-states_by_lanes[1]/total))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, choices=range(3, 28), default=27)
    parser.add_argument('--scratch-mib', type=int, default=10240)
    parser.add_argument('--lanes', type=int, choices=(1, 2, 4, 8, 16, 32), default=16)
    args = parser.parse_args()
    if args.scratch_mib < 1:
        parser.error('scratch-mib must be positive')
    print(json.dumps(layout_plan(args.n, args.scratch_mib, args.lanes), indent=2))


if __name__ == '__main__':
    main()
