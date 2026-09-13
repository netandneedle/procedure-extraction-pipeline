"""LangGraph PostgreSQL checkpointer configuration.

Persists pipeline state to PostgreSQL, enabling:
- Pause at human gates (interrupt + resume)
- Crash recovery (state survives restarts)
- Multi-day waits (analyst reviews on their schedule)
"""

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.config import settings


_checkpointer: AsyncPostgresSaver | None = None
_checkpointer_cm = None  # holds the context manager to keep it alive


async def get_checkpointer() -> AsyncPostgresSaver:
    """Create and return the PostgreSQL checkpointer.

    Uses async context manager API (langgraph-checkpoint-postgres >= 2.0).
    The context manager is kept alive for the app's lifetime; cleanup
    happens in close_checkpointer().
    """
    global _checkpointer, _checkpointer_cm
    if _checkpointer is not None:
        return _checkpointer

    conn_string = settings.database_url.replace("+asyncpg", "")
    _checkpointer_cm = AsyncPostgresSaver.from_conn_string(conn_string)
    _checkpointer = await _checkpointer_cm.__aenter__()
    await _checkpointer.setup()
    return _checkpointer


async def close_checkpointer() -> None:
    """Close the checkpointer connection pool."""
    global _checkpointer, _checkpointer_cm
    if _checkpointer_cm is not None:
        await _checkpointer_cm.__aexit__(None, None, None)
        _checkpointer = None
        _checkpointer_cm = None
