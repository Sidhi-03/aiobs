"""Minimal LLM observability SDK.

Usage (global singleton):

    from llm_observability import observer
    observer.observe()  # enable instrumentation
    # ... make LLM calls ...
    observer.end()      # end current session
    observer.flush()    # write a single JSON file to disk

Extensible provider model with OpenAI support out of the box.
"""

from .collector import Collector
from .providers.base import BaseProvider

# Global collector singleton, intentionally simple API
observer = Collector()

__all__ = ["observer", "Collector", "BaseProvider"]
