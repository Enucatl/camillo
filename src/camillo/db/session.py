from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from camillo.settings import settings

engine = create_async_engine(settings.database_url, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession]:
    """Yield one async database session for a FastAPI request."""
    async with AsyncSessionLocal() as session:
        yield session
