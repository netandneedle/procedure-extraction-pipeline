"""AI senior-analyst reviewer for the pipeline's human-review gates.

A stateful reviewer: reads the source once, then walks each enabled gate
carrying its own reasoning forward. Resets per source — cross-source learning
is the feedback flywheel's job, not this one's.

Two modes per gate, configured via `Source.gate_modes` / `PipelineState`:
  assist — recommends; a human decides and submits.
  auto   — decides and the pipeline advances unattended.

Entry point is `run_reviewer`.
"""

from app.services.reviewer.runner import (
    apply_grounding,
    drop_invalid_entity_types,
    run_reviewer,
)

__all__ = ["apply_grounding", "drop_invalid_entity_types", "run_reviewer"]
