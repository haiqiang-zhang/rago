import copy
import dataclasses
import hashlib
import itertools
import json
from typing import Any, Iterable

# Data classes of retrieval / inference policies


@dataclasses.dataclass(frozen=True)
class RetrievalPolicy:
    # Whether to invoke retrieval at all
    run_retrieval: bool = False

    # Retrieval pattern can be either "once" or "interval" (every n tokens)
    retrieval_pattern: str | None = None
    retrieval_interval: int | None = None

    # Query expansion model
    run_query_expansion: bool = False
    query_expansion_seq_len_inference_prefill: int | None = None
    query_expansion_seq_len_prefill_with_template: int | None = (
        None  # prefill = query expansion instruction template + question
    )
    query_expansion_dec_steps: int | None = None

    # Per-user fanout introduced by the query rewriter: one rewriter call
    # produces this many sub-queries, each of which hits ``encode`` and
    # ``retrieval`` independently. ``None``/1 means no fanout (1 rewrite
    # per user request, the historical default). Mirrors how
    # ``passage_reranker_topk`` multiplies workload at the reranker stage.
    query_expansion_fanout: int | None = None

    # Passage reranker model
    run_passage_reranker: bool = False
    passage_reranker_seq_len_inference_prefill: int | None = None
    passage_reranker_topk: int | None = None


@dataclasses.dataclass(frozen=True)
class EncodeDBPolicy:
    # Whether to invoke encode db at all
    run_encode_db: bool = False
    # Chunk length is the seq_len_inference_prefill for the encoder model
    encode_seq_len_inference_prefill: int | None = None
    # number of vectors in the database for encode db
    encode_db_nvec: int | None = None


class PhysicalMapping:
    def __init__(
        self,
        num_retrieval_servers: int = 0,
        num_chips: dict[str, int] = {
            "query_expansion_prefill": 0,
            "query_expansion_decode": 0,
            "encode": 0,
            "passage_reranker": 0,
            "passage_filter": 0,
            "passage_compressor_prefill": 0,
            "passage_compressor_decode": 0,
            "prefill": 0,
            "decode": 0,
            "e2e": 0,
        },
        placement_policy: str = "disaggregated",  # 'collocated' or 'disaggregated'
        collocation_strategy: (
            list[list[str]] | None
        ) = None,  # [['retrieval'], ['prefill'], ['decode']]
        available_devices: list[str] | None = None,
        resource_group_devices: list[list[str]] | None = None,
        stage_devices: dict[str, list[str]] | None = None,
        device_layout_id: str | None = None,
    ):
        self.num_retrieval_servers = num_retrieval_servers
        self.num_chips = num_chips
        self.placement_policy = placement_policy
        self.collocation_strategy = collocation_strategy
        # Concrete physical placement.  RAGO owns this part of a candidate:
        # rag_stack may qualify the stage names and add engine semantics, but a
        # candidate is never priced/selected as a chip-count-only abstraction.
        self.available_devices = list(available_devices or [])
        self.resource_group_devices = [
            list(devices) for devices in (resource_group_devices or [])
        ]
        self.stage_devices = {
            str(stage): list(devices)
            for stage, devices in (stage_devices or {}).items()
        }
        self.device_layout_id = device_layout_id

    # Print each of the attribute
    def __str__(self):
        return (
            f"PhysicalMapping(num_retrieval_servers={self.num_retrieval_servers},"
            + f" num_chips={self.num_chips}, placement_policy={self.placement_policy},"
            + f" collocation_strategy={self.collocation_strategy})"
        )

    @property
    def has_concrete_devices(self) -> bool:
        return bool(self.stage_devices or self.resource_group_devices)


def _link_value(
    device_links: dict[Any, Any] | None,
    left: str,
    right: str,
) -> Any:
    """Stable topology-class value for an unordered device pair."""
    if left == right:
        return ("self",)
    links = device_links or {}
    key = tuple(sorted((str(left), str(right))))
    value = links.get(key)
    if value is None:
        value = links.get("|".join(key), links.get(frozenset(key), ("default",)))
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, dict):
        return tuple(sorted(value.items()))
    return value


def _layout_signature(
    group_devices: list[list[str]],
    device_links: dict[Any, Any] | None,
) -> tuple:
    """Topology-equivalence signature, preserving rank and group order.

    Device labels themselves are deliberately absent.  Two layouts related by
    renaming otherwise-equivalent GPUs collapse to one candidate, while a
    same-pair and cross-pair TP group (or a different TP/PP rank order) remain
    distinct candidates.

    ``stage_devices`` need not be a second signature axis: RAGO defines the
    within-group placement policy canonically below.  Full-width stages use the
    whole ordered group and smaller riders consume consecutive round-robin
    slices in collocation-strategy order.  Thus stage occupancy is a
    deterministic function of the ordered group ranks and stage chip counts;
    layouts with the same signature are topology-isomorphic for every stage
    and for DES contention.  Alternative rider stacking is not a deployment
    dimension (and the measured resolver implements the same balancing rule).
    """
    flat = [device for group in group_devices for device in group]
    boundaries = tuple(len(group) for group in group_devices)
    matrix = tuple(
        _link_value(device_links, flat[i], flat[j])
        for i in range(len(flat))
        for j in range(i + 1, len(flat))
    )
    return boundaries, matrix


def _validate_fixed_devices(mapping: PhysicalMapping) -> PhysicalMapping:
    groups = mapping.collocation_strategy or []
    if mapping.resource_group_devices and len(mapping.resource_group_devices) != len(groups):
        raise ValueError(
            "resource_group_devices must align 1:1 with collocation_strategy"
        )
    available = set(mapping.available_devices)
    for stage, devices in mapping.stage_devices.items():
        expected = int(mapping.num_chips.get(stage, len(devices)) or len(devices))
        if stage != "retrieval" and len(devices) != expected:
            raise ValueError(
                f"stage_devices[{stage!r}] has {len(devices)} devices, expected {expected}"
            )
        if available and not set(devices).issubset(available):
            raise ValueError(
                f"stage_devices[{stage!r}] contains devices outside available_devices"
            )
    return mapping


def enumerate_concrete_device_mappings(
    mapping: PhysicalMapping,
    *,
    available_devices: Iterable[str] | None = None,
    device_links: dict[Any, Any] | None = None,
    max_layouts: int = 64,
) -> list[PhysicalMapping]:
    """Expand one chip-count mapping into topology-distinct physical mappings.

    With a homogeneous fabric there is exactly one deterministic consecutive
    assignment.  With a declared pair topology, rank permutations are
    enumerated and deduplicated by their ordered link-class matrix.  This keeps
    a 4-GPU 2-pair box small (usually 1--4 layouts per resource mapping) while
    retaining the same-pair/cross-pair and TP/PP-order cases that change cost.

    Fixed mappings carrying ``stage_devices`` are validated and returned
    unchanged; this is the replay path for measured resolved layouts.
    """
    if mapping.has_concrete_devices:
        return [_validate_fixed_devices(mapping)]

    devices = [str(device) for device in (available_devices or mapping.available_devices)]
    groups = [list(group) for group in (mapping.collocation_strategy or [])]
    if not groups:
        raise ValueError("Concrete device materialization requires collocation_strategy")

    group_sizes: list[int] = []
    for group in groups:
        gpu_stages = [stage for stage in group if stage != "retrieval"]
        group_sizes.append(
            max((int(mapping.num_chips.get(stage, 0) or 0) for stage in gpu_stages), default=0)
        )
    total = sum(group_sizes)
    if total > len(devices):
        raise ValueError(
            f"Physical mapping needs {total} devices, only {len(devices)} available: {devices}"
        )
    if total and not devices:
        raise ValueError("GPU mapping requires a non-empty available_devices inventory")

    # No heterogeneous link classes means rank/device permutations are
    # cost-equivalent; keep the deterministic declared order only.
    if not device_links or total > 8:
        orders = [tuple(devices[:total])]
    else:
        orders = itertools.permutations(devices, total)

    layouts: list[PhysicalMapping] = []
    seen: set[tuple] = set()
    for order in orders:
        cursor = 0
        group_devices: list[list[str]] = []
        for size in group_sizes:
            group_devices.append(list(order[cursor:cursor + size]))
            cursor += size
        signature = _layout_signature(group_devices, device_links)
        if signature in seen:
            continue
        seen.add(signature)

        concrete = copy.deepcopy(mapping)
        concrete.available_devices = list(devices)
        concrete.resource_group_devices = group_devices
        concrete.stage_devices = {}
        for group, assigned in zip(groups, group_devices):
            # Canonical pooled-group scheduling contract.  This intentionally
            # mirrors rag_stack.search_space.placement.derive_layout: small
            # co-resident engines rotate across ranks instead of all piling on
            # rank 0, while TP/full-width engines occupy the complete group.
            # Since this policy is deterministic, only topology-distinct group
            # rank orders need enumeration on the target four-card box.
            rider_offset = 0
            for stage in group:
                if stage == "retrieval":
                    continue
                n = int(mapping.num_chips.get(stage, len(assigned)) or len(assigned))
                if n >= len(assigned):
                    picked = list(assigned)
                else:
                    picked = [
                        assigned[(rider_offset + index) % len(assigned)]
                        for index in range(n)
                    ]
                    rider_offset = (rider_offset + n) % len(assigned)
                concrete.stage_devices[stage] = picked
        digest = hashlib.sha256(json.dumps(
            {
                "groups": groups,
                "group_devices": group_devices,
                "stage_devices": concrete.stage_devices,
                "num_chips": {
                    str(stage): int(value or 0)
                    for stage, value in concrete.num_chips.items()
                },
                "num_retrieval_servers": int(concrete.num_retrieval_servers or 0),
                "placement_policy": concrete.placement_policy,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()[:12]
        concrete.device_layout_id = f"layout_{digest}"
        layouts.append(concrete)
        if len(layouts) >= max(1, int(max_layouts)):
            break

    if not layouts:
        concrete = copy.deepcopy(mapping)
        concrete.available_devices = list(devices)
        concrete.resource_group_devices = [[] for _ in groups]
        concrete.device_layout_id = "layout_0000"
        layouts.append(concrete)
    return layouts
