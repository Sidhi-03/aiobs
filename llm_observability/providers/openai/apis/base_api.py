from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional


class BaseOpenAIAPIModule(ABC):
    """Abstract interface for an OpenAI API module."""

    name: str = "openai-api"

    @classmethod
    def is_available(cls) -> bool:
        return True

    @abstractmethod
    def install(self, collector: Any) -> Optional[Callable[[], None]]:
        """Install instrumentation and return optional unpatch function."""
        raise NotImplementedError

