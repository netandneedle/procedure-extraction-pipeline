"""SQLAlchemy declarative base and engine setup.

Single async engine shared across the app lifetime. The LLM cache layer
used to require a separate sync engine because ``call_llm`` was sync;
that's been migrated to the async client + async session, so only the
async engine remains.
"""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass


# Async engine — shared across the app lifetime.
# The database_url already uses asyncpg (postgresql+asyncpg://...).
engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_size=5,
    max_overflow=10,
)

# Session factory — injected into routes via dependency.
async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)
