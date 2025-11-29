from __future__ import annotations

from typing import Any, Callable, List, Optional

from ..base import BaseProvider
from .apis.base_api import BaseOpenAIAPIModule
from .apis.chat_completions import ChatCompletionsAPI
from .apis.embeddings import EmbeddingsAPI


class OpenAIProvider(BaseProvider):
    name = "openai"

    def __init__(self) -> None:
        self._modules: List[BaseOpenAIAPIModule] = []

    @classmethod
    def is_available(cls) -> bool:
        # Available if any sub-module is available
        try:
            return ChatCompletionsAPI.is_available() or EmbeddingsAPI.is_available()
        except Exception:
            return False

    def install(self, collector: Any) -> Optional[Callable[[], None]]:
        unpatchers: List[Callable[[], None]] = []

        # Build module list (extensible: append additional APIs here)
        modules: List[BaseOpenAIAPIModule] = []
        if ChatCompletionsAPI.is_available():
            modules.append(ChatCompletionsAPI())
        if EmbeddingsAPI.is_available():
            modules.append(EmbeddingsAPI())

        for mod in modules:
            try:
                up = mod.install(collector)
                if up:
                    unpatchers.append(up)
                self._modules.append(mod)
            except Exception:
                continue

        def unpatch_all() -> None:
            for up in reversed(unpatchers):
                try:
                    up()
                except Exception:
                    pass

        return unpatch_all if unpatchers else None
