"""Single source of truth for gate metadata used by the API routes.

Each gate has a small bundle of route-layer attributes (predecessor node
for `aupdate_state`, expected/resuming statuses, the state fields it
reads and writes, etc.) that previously lived in 6 parallel dicts spread
across `gates.py` and `pipeline.py`. Adding a new gate had a half-dozen
touchpoints, and keeping the dicts aligned was a manual chore.

The Gate dataclass below collapses those into one record per gate.
Lookup tables (`GATES_BY_NODE`, `GATES_BY_INT_ID`) are derived once at
import time so callers can use whichever key they have.

`int_id` is None for `gate_chunks` because that gate lives outside the
int-keyed `0/1/2` namespace; it has its own string-keyed route. Other
fields that only make sense for the int-keyed gates (`item_field`,
`review_field`) are also None for `gate_chunks`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Gate:
    node_name: str              # graph node name, e.g. "gate_0"
    state_key: str              # gates_enabled key, e.g. "entities"
    predecessor_node: str       # immediate predecessor for aupdate_state(as_node=...)
    expected_status: str        # PipelineStatus when paused at this gate
    resuming_status: str        # status flipped on the queue row when /submit accepts
    int_id: int | None          # 0/1/2 for legacy int-keyed routes; None for gate_chunks
    item_field: str | None      # state field holding review items (int-keyed gates only)
    review_field: str | None    # state field receiving the raw analyst payload


# Ordered list — display order matches the pipeline's gate sequence and
# drives the int-id derivation for any future tooling that needs it.
GATES: tuple[Gate, ...] = (
    Gate(
        node_name="gate_0",
        state_key="entities",
        predecessor_node="extract_entities",
        expected_status="gate_0",
        resuming_status="resuming_from_gate_0",
        int_id=0,
        item_field="entities",
        review_field="gate0_reviews",
    ),
    Gate(
        node_name="gate_chunks",
        state_key="chunks",
        predecessor_node="chunk_behaviors",
        expected_status="gate_chunks",
        resuming_status="resuming_from_gate_chunks",
        int_id=None,
        item_field=None,
        review_field=None,
    ),
    Gate(
        node_name="gate_1",
        state_key="procedures",
        predecessor_node="draft_procedures",
        expected_status="gate_1",
        resuming_status="resuming_from_gate_1",
        int_id=1,
        item_field="drafts",
        review_field="gate1_reviews",
    ),
    Gate(
        node_name="gate_2",
        state_key="bundle",
        predecessor_node="normalize",
        expected_status="gate_2",
        resuming_status="resuming_from_gate_2",
        int_id=2,
        item_field="relationship_preview",
        review_field="gate2_reviews",
    ),
)

GATES_BY_NODE: dict[str, Gate] = {g.node_name: g for g in GATES}
GATES_BY_INT_ID: dict[int, Gate] = {g.int_id: g for g in GATES if g.int_id is not None}
