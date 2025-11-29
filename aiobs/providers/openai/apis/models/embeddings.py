from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from .base import BaseOpenAIRequest, BaseOpenAIResponse


class EmbeddingsRequest(BaseOpenAIRequest):
    """Request model for OpenAI embeddings.create API."""

    input: Optional[Union[str, List[str], List[int], List[List[int]]]] = Field(default=None)
    encoding_format: Optional[str] = Field(default=None)
    dimensions: Optional[int] = Field(default=None)
    user: Optional[str] = Field(default=None)
    other: Dict[str, Any] = Field(default_factory=dict)


class EmbeddingData(BaseModel):
    """Single embedding object in the response."""

    index: int
    embedding: List[float] = Field(default_factory=list)
    object: str = "embedding"


class EmbeddingsResponse(BaseOpenAIResponse):
    """Response model for OpenAI embeddings.create API."""

    object: Optional[str] = Field(default=None)
    data: Optional[List[EmbeddingData]] = Field(default=None)
    embedding_dimensions: Optional[int] = Field(default=None)

