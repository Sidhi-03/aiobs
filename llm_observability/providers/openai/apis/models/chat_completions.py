from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from .base import BaseOpenAIRequest, BaseOpenAIResponse


class Message(BaseModel):
    role: str
    content: Any


class ChatCompletionsRequest(BaseOpenAIRequest):
    messages: Optional[List[Message]] = Field(default=None)
    temperature: Optional[float] = Field(default=None)
    max_tokens: Optional[int] = Field(default=None)
    other: Dict[str, Any] = Field(default_factory=dict)


class ChatCompletionsResponse(BaseOpenAIResponse):
    text: Optional[str] = Field(default=None)

