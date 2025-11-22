from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional, Any


class BaseProvider(ABC):
    """Abstract base class for provider instrumentation.

    Subclasses install monkeypatches or hooks to capture request/response
    details and call `collector._record_event(...)` with normalized payloads.
    """

    name: str = "provider"

    @classmethod
    def is_available(cls) -> bool:
        """Return True if the provider can be instrumented (deps present)."""
        return True

    @abstractmethod
    def install(self, collector: Any) -> Optional[Callable[[], None]]:
        """Apply instrumentation and return an optional unpatch function."""
        raise NotImplementedError

