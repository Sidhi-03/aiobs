from __future__ import annotations

import inspect
import os
import time
from typing import Any, Callable, Dict, List, Optional

from .base_api import BaseOpenAIAPIModule
from .models import (
    EmbeddingsRequest,
    EmbeddingsResponse,
    EmbeddingData,
)
from ....models import Event as ObsEvent, Callsite as ObsCallsite


class EmbeddingsAPI(BaseOpenAIAPIModule):
    name = "embeddings"

    @classmethod
    def is_available(cls) -> bool:
        try:
            from openai.resources.embeddings import Embeddings as _  # noqa: F401
            return True
        except Exception:
            return False

    def install(self, collector: Any) -> Optional[Callable[[], None]]:
        try:
            from openai.resources.embeddings import Embeddings as _Embeddings  # type: ignore
        except Exception:
            return None

        if not hasattr(_Embeddings, "create"):
            return None

        original_create = _Embeddings.create

        def _extract_request(args: tuple, kwargs: dict) -> EmbeddingsRequest:
            req: Dict[str, Any] = {}
            if args:
                req["_args"] = args
            req.update(kwargs or {})

            # Extract input - could be string, list of strings, or token arrays
            input_data = req.get("input")
            # Truncate input preview for large inputs
            if isinstance(input_data, list) and len(input_data) > 3:
                input_preview = input_data[:3]
            else:
                input_preview = input_data

            return EmbeddingsRequest(
                model=req.get("model"),
                input=input_preview,
                encoding_format=req.get("encoding_format"),
                dimensions=req.get("dimensions"),
                user=req.get("user"),
                other={k: v for k, v in req.items() if k not in {
                    "model", "input", "encoding_format", "dimensions", "user"
                }},
            )

        def _extract_response(resp: Any) -> EmbeddingsResponse:
            rid = getattr(resp, "id", None)
            rmodel = getattr(resp, "model", None)
            obj_type = getattr(resp, "object", None)

            # Extract embedding data
            data: Optional[List[EmbeddingData]] = None
            embedding_dims: Optional[int] = None
            try:
                raw_data = getattr(resp, "data", None)
                if raw_data:
                    data = []
                    for item in raw_data:
                        emb = getattr(item, "embedding", [])
                        if embedding_dims is None and emb:
                            embedding_dims = len(emb)
                        data.append(EmbeddingData(
                            index=getattr(item, "index", 0),
                            embedding=emb,
                            object=getattr(item, "object", "embedding"),
                        ))
            except Exception:
                pass

            # Extract usage
            usage = getattr(resp, "usage", None)
            u = None
            try:
                if usage is not None:
                    if hasattr(usage, "model_dump"):
                        u = usage.model_dump()
                    elif hasattr(usage, "dict"):
                        u = usage.dict()
                    elif isinstance(usage, dict):
                        u = usage
            except Exception:
                pass

            return EmbeddingsResponse(
                id=rid,
                model=rmodel,
                object=obj_type,
                data=data,
                usage=u,
                embedding_dimensions=embedding_dims,
            )

        def _get_callsite() -> Optional[ObsCallsite]:
            try:
                frames = inspect.stack()[2:]
                for fi in frames:
                    fname = os.path.abspath(fi.filename)
                    if (
                        f"{os.sep}aiobs{os.sep}" in fname
                        or f"{os.sep}site-packages{os.sep}openai{os.sep}" in fname
                        or fname.endswith("openai/__init__.py")
                    ):
                        continue
                    try:
                        rel = os.path.relpath(fname, start=os.getcwd())
                    except Exception:
                        rel = fname
                    return ObsCallsite(file=rel, line=fi.lineno, function=fi.function)
            except Exception:
                pass
            return None

        def wrapped(self, *args, **kwargs):  # type: ignore[no-redef]
            import uuid
            started = time.time()
            request_info = _extract_request(args, kwargs)
            callsite = _get_callsite()
            error: Optional[str] = None
            response_info: Optional[EmbeddingsResponse] = None

            # Capture parent span context for trace tree linking
            parent_span_id = collector.get_current_span_id()
            span_id = str(uuid.uuid4())

            try:
                resp = original_create(self, *args, **kwargs)
                response_info = _extract_response(resp)
                return resp
            except Exception as e:  # noqa: BLE001
                error = f"{type(e).__name__}: {e}"
                raise
            finally:
                ended = time.time()
                event = ObsEvent(
                    provider="openai",
                    api="embeddings.create",
                    callsite=callsite,
                    request=request_info,
                    response=response_info,
                    error=error,
                    started_at=started,
                    ended_at=ended,
                    duration_ms=round((ended - started) * 1000, 3),
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                )
                collector._record_event(event)

        setattr(_Embeddings, "create", wrapped)  # type: ignore[attr-defined]

        def unpatch() -> None:
            try:
                setattr(_Embeddings, "create", original_create)  # type: ignore[attr-defined]
            except Exception:
                pass

        return unpatch

