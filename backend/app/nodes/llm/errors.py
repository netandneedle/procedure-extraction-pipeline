"""Structured exceptions raised by the LLM layer.

These are distinct from anthropic.APIError (transport / rate-limit concerns).
They surface when the model returns a successful response whose *content*
fails our validation — usually a schema mismatch after tool_use.

The adapter converts Pydantic ValidationError into LLMValidationError so
callers (pipeline nodes) don't have to import pydantic.ValidationError
just to catch it.
"""

from __future__ import annotations

from typing import Any


class LLMValidationError(Exception):
    """Raised when a tool_use response fails Pydantic validation after retries.

    Attributes:
        tool_name: The tool that produced the bad output.
        errors: List of Pydantic error dicts (loc, msg, type). May be long;
            callers should truncate before logging.
        raw_output: The original tool_input dict from the last attempt.
            Kept for post-mortem in an llm_outputs table. Do NOT log in full
            — may contain user content from source documents.
        attempts: Number of LLM calls made before giving up.
    """

    def __init__(
        self,
        tool_name: str,
        errors: list[dict[str, Any]],
        raw_output: dict[str, Any],
        attempts: int,
    ) -> None:
        self.tool_name = tool_name
        self.errors = errors
        self.raw_output = raw_output
        self.attempts = attempts

        # Summarize first few errors for the exception message. Full list
        # is on the instance for observability.
        preview = errors[:3]
        super().__init__(
            f"{tool_name}: tool_use validation failed after {attempts} attempt(s); "
            f"first errors: {preview}"
        )
