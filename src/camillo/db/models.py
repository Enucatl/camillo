from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
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
    confidence: Mapped[float] = mapped_column(Float, default=0.8)
    source: Mapped[str | None] = mapped_column(Text, nullable=True)
    superseded_by: Mapped[UUID | None] = mapped_column(ForeignKey("memories.id"), nullable=True)
    deprecated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
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
    outgoing_relations: Mapped[list[MemoryRelation]] = relationship(
        back_populates="source",
        foreign_keys="MemoryRelation.source_id",
        cascade="all, delete-orphan",
    )
    incoming_relations: Mapped[list[MemoryRelation]] = relationship(
        back_populates="target",
        foreign_keys="MemoryRelation.target_id",
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


class MemoryRelation(Base):
    """Semantic or lifecycle relationship between two memories.

    Relations are separate from Hebbian edges so associative strength and belief
    reconciliation can evolve without conflating their meanings.
    """

    __tablename__ = "memory_relations"
    __table_args__ = (
        UniqueConstraint(
            "source_id",
            "target_id",
            "relation_type",
            name="uq_memory_relations_source_target_type",
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    source_id: Mapped[UUID] = mapped_column(ForeignKey("memories.id", ondelete="CASCADE"))
    target_id: Mapped[UUID] = mapped_column(ForeignKey("memories.id", ondelete="CASCADE"))
    relation_type: Mapped[str] = mapped_column(String(50))
    confidence: Mapped[float] = mapped_column(Float, default=0.8)
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    source: Mapped[Memory] = relationship(
        foreign_keys=[source_id],
        back_populates="outgoing_relations",
    )
    target: Mapped[Memory] = relationship(
        foreign_keys=[target_id],
        back_populates="incoming_relations",
    )


class DreamRun(Base):
    """Audit record for one dreaming/consolidation attempt.

    Dreaming is a background promotion process, so each run records source and
    created memory IDs without taking ownership of transaction commits.
    """

    __tablename__ = "dream_runs"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    namespace: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    seed_memory_ids: Mapped[list[UUID]] = mapped_column(
        ARRAY(PG_UUID(as_uuid=True)),
        nullable=False,
        default=list,
    )
    source_memory_ids: Mapped[list[UUID]] = mapped_column(
        ARRAY(PG_UUID(as_uuid=True)),
        nullable=False,
        default=list,
    )
    created_memory_ids: Mapped[list[UUID]] = mapped_column(
        ARRAY(PG_UUID(as_uuid=True)),
        nullable=False,
        default=list,
    )
    clusters_considered: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    clusters_dreamed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    memories_created: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
