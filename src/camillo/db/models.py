from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from camillo.db.base import Base
from camillo.settings import settings


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp for database defaults."""
    return datetime.now(UTC)


class Memory(Base):
    """Stored cognitive memory.

    Memory rows are namespace-scoped because the service is intended to serve
    multiple repositories or agents without cross-contaminating recall.
    """

    __tablename__ = "memories"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    namespace: Mapped[str] = mapped_column(String(255), index=True)
    session_id: Mapped[str | None] = mapped_column(String(255), index=True, nullable=True)
    raw_content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(settings.embedding_dim))
    type: Mapped[str] = mapped_column(String(50), default="episodic")
    status: Mapped[str] = mapped_column(String(50), default="active")
    base_importance: Mapped[float] = mapped_column(Float, default=0.5)
    access_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    last_accessed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)

    outgoing_edges: Mapped[list[HebbianEdge]] = relationship(
        back_populates="source",
        foreign_keys="HebbianEdge.source_id",
        cascade="all, delete-orphan",
    )
    incoming_edges: Mapped[list[HebbianEdge]] = relationship(
        back_populates="target",
        foreign_keys="HebbianEdge.target_id",
        cascade="all, delete-orphan",
    )


class HebbianEdge(Base):
    """Undirected associative edge between two memories.

    Edges are stored with a canonical source/target ordering in the store so the
    same memory pair cannot acquire two independent weights.
    """

    __tablename__ = "hebbian_edges"
    __table_args__ = (UniqueConstraint("source_id", "target_id", name="uq_hebbian_edges_pair"),)

    source_id: Mapped[UUID] = mapped_column(
        ForeignKey("memories.id", ondelete="CASCADE"), primary_key=True
    )
    target_id: Mapped[UUID] = mapped_column(
        ForeignKey("memories.id", ondelete="CASCADE"), primary_key=True
    )
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    last_co_accessed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    source: Mapped[Memory] = relationship(foreign_keys=[source_id], back_populates="outgoing_edges")
    target: Mapped[Memory] = relationship(foreign_keys=[target_id], back_populates="incoming_edges")
