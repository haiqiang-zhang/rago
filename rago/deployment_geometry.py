"""Pure deployment geometry shared by analytical and measured searches.

There are no service curves, memory estimates, or performance rankings here.
Callers supply the supported chip counts for each stage.
"""
from __future__ import annotations

import itertools

from .config import PhysicalMapping, enumerate_concrete_device_mappings


def collocation_combinations(
    stages, *, filter_full_disaggregation=True,
    filter_retrieval_at_beginning_or_end=False, disaggregate_decode=True,
):
    """Enumerate contiguous partitions using RAGO's existing restrictions."""
    stages = list(stages)
    if len(stages) < 2:
        return []
    out = []
    for cuts in itertools.product((1, 0), repeat=max(0, len(stages) - 1)):
        groups = [[]]
        for index, stage in enumerate(stages):
            groups[-1].append(stage)
            if index < len(cuts) and cuts[index]:
                groups.append([])
        if filter_full_disaggregation and len(groups) == len(stages):
            continue
        if disaggregate_decode and groups[-1] != ["decode"]:
            continue
        if filter_retrieval_at_beginning_or_end and any(
            len(group) > 1 and (group[0] == "retrieval" or group[-1] == "retrieval")
            for group in groups
        ):
            continue
        out.append(groups)
    return out


def placement_strategies(
    stages, *, placement_policy, min_num_chips, max_num_chips,
    max_num_chips_per_stage, allow_decode_collocation=True,
):
    """Apply the same partition bounds used by RAG-Stack's RAGO sweep."""
    if placement_policy == "disaggregated":
        return [[[stage] for stage in stages]]
    if placement_policy != "collocated":
        raise ValueError(f"Unknown mapping policy: {placement_policy}")
    if max_num_chips == 1:
        return [[list(stages)]]
    return [
        groups for groups in collocation_combinations(
            stages, filter_retrieval_at_beginning_or_end=True,
            disaggregate_decode=not allow_decode_collocation,
        )
        if len(groups) <= max_num_chips
        and len(groups) * max_num_chips_per_stage >= min_num_chips
    ]


def enumerate_hardware_mappings(
    *, stages, allowed_num_chips, min_num_chips, max_num_chips,
    max_num_chips_per_stage, num_chips_per_server, placement_policy,
    collocation_strategy, available_devices, device_links=None, max_layouts=64,
    independent_stage_widths=True,
):
    """Enumerate legal stage allocations before assigning physical devices.

    A shared group's capacity is the maximum of its members' requests, not
    their sum. Different groups use disjoint devices. Unified prefill/decode
    roles are one engine and therefore request the same width. The legacy
    uniform-width mode exists only to reproduce older experiment geometry.
    """
    groups = (
        [[stage] for stage in stages]
        if placement_policy == "disaggregated" else collocation_strategy
    )
    if placement_policy not in {"collocated", "disaggregated"}:
        raise ValueError(f"Unknown mapping policy: {placement_policy}")
    powers = {1 << i for i in range(int(max_num_chips_per_stage).bit_length())}
    choices = []
    for group in groups:
        gpu_stages = [stage for stage in group if stage != "retrieval"]
        if not gpu_stages:
            choices.append([{}])
            continue
        if independent_stage_widths:
            stage_choices = [sorted(n for n in allowed_num_chips[stage]
                                    if 0 < n <= max_num_chips_per_stage
                                    and (placement_policy != "collocated" or n in powers))
                             for stage in gpu_stages]
            allocations = []
            for widths in itertools.product(*stage_choices):
                allocation = dict(zip(gpu_stages, widths))
                if any(allocation[pre] != allocation[dec]
                       for pre, dec in unified_role_pairs(gpu_stages)):
                    continue
                allocations.append(allocation)
            choices.append(allocations)
            continue
        allowed = set(allowed_num_chips[gpu_stages[0]])
        for stage in gpu_stages[1:]:
            allowed.intersection_update(allowed_num_chips[stage])
        if placement_policy == "collocated":
            allowed.intersection_update(powers)
        choices.append([{stage: n for stage in gpu_stages}
                        for n in sorted(allowed) if 0 < n <= max_num_chips_per_stage])

    result = []
    for allocations in itertools.product(*choices):
        total = sum(max(allocation.values(), default=0) for allocation in allocations)
        if not min_num_chips <= total <= max_num_chips:
            continue
        mapping = PhysicalMapping(
            num_chips={
                **PhysicalMapping().num_chips,
                **{stage: width for allocation in allocations for stage, width in allocation.items()},
                "e2e": total,
            },
            num_retrieval_servers=max(1, total // num_chips_per_server),
            placement_policy=placement_policy,
            collocation_strategy=[list(group) for group in groups],
        )
        result.extend(enumerate_concrete_device_mappings(
            mapping, available_devices=available_devices,
            device_links=device_links, max_layouts=max_layouts,
        ))
    return result


def unified_role_pairs(stages):
    """RAGO role names belonging to one engine when they share a group."""
    stages = set(stages)
    for pre in sorted(stages):
        if pre == "prefill":
            dec = "decode"
        elif pre.endswith("_prefill"):
            dec = pre[:-len("prefill")] + "decode"
        else:
            continue
        if dec in stages:
            yield pre, dec
