"""FastAPI dependency injection for database sessions, graph, and checkpointer.

Usage in routes:
    from app.api.dependencies import get_db, get_graph

    @router.post("/")
    async def create(db: AsyncSession = Depends(get_db)):
        ...

    @router.post("/run")
    async def run(graph = Depends(get_graph)):
        ...
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.base import async_session


# =============================================================================
# Database session
# =============================================================================

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield a database session, auto-close on request end."""
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()


# =============================================================================
# Compiled LangGraph + checkpointer
#
# These are set at startup by the lifespan handler in main.py.
# Routes access them via Depends(get_graph) / Depends(get_checkpointer).
# =============================================================================

_compiled_graph: Any = None
_checkpointer: Any = None


def set_graph(graph: Any) -> None:
    """Called once at startup to store the compiled graph."""
    global _compiled_graph
    _compiled_graph = graph


def set_checkpointer(checkpointer: Any) -> None:
    """Called once at startup to store the checkpointer."""
    global _checkpointer
    _checkpointer = checkpointer


def get_graph() -> Any:
    """Dependency: return the compiled LangGraph pipeline."""
    if _compiled_graph is None:
        raise RuntimeError("Pipeline graph not initialized. Is the app starting up?")
    return _compiled_graph


def get_checkpointer() -> Any:
    """Dependency: return the LangGraph PostgreSQL checkpointer."""
    if _checkpointer is None:
        raise RuntimeError("Checkpointer not initialized. Is the app starting up?")
    return _checkpointer
