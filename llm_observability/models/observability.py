from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field


class SessionMeta(BaseModel):
    pid: int
    cwd: str


class Session(BaseModel):
    id: str
    name: str
    started_at: float
    ended_at: Optional[float] = Field(default=None)
    meta: SessionMeta


class Callsite(BaseModel):
    file: Optional[str] = Field(default=None)
    line: Optional[int] = Field(default=None)
    function: Optional[str] = Field(default=None)


class Event(BaseModel):
    provider: str
    api: str
    request: Any
    response: Optional[Any] = None
    error: Optional[str] = Field(default=None)
    started_at: float
    ended_at: float
    duration_ms: float
    callsite: Optional[Callsite] = Field(default=None)


class ObservedEvent(Event):
    session_id: str


class ObservabilityExport(BaseModel):
    sessions: List[Session]
    events: List[ObservedEvent]
    generated_at: float
    version: int = 1
