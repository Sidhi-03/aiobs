from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class BaseOpenAIRequest(BaseModel):
    """Base class for OpenAI request capture models."""

    model: Optional[str] = Field(default=None)

    def redacted(self) -> "BaseOpenAIRequest":
        """Return a copy safe for logging (override in subclasses)."""
        return self


class BaseOpenAIResponse(BaseModel):
    """Base class for OpenAI response capture models."""

    id: Optional[str] = Field(default=None)
    model: Optional[str] = Field(default=None)
    usage: Optional[Dict[str, Any]] = Field(default=None)

    def redacted(self) -> "BaseOpenAIResponse":
        """Return a copy safe for logging (override in subclasses)."""
        return self

