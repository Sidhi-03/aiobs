import json
import os
from types import SimpleNamespace
import pytest

pytest.importorskip("openai")

from aiobs import observer


def test_openai_chat_completions_instrumentation(monkeypatch, tmp_path):
    # Prepare a fake create implementation to avoid network calls
    from openai.resources.chat.completions import Completions

    def fake_create(self, *args, **kwargs):  # noqa: ARG001
        message = SimpleNamespace(content="hello world")
        choice = SimpleNamespace(message=message)
        usage = {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
        return SimpleNamespace(id="fake-id", model="gpt-test", choices=[choice], usage=usage)

    # Monkeypatch BEFORE observe() so the provider wraps our fake
    monkeypatch.setattr(Completions, "create", fake_create, raising=True)

    # Start observer (installs provider instrumentation)
    observer.observe("openai-instrumentation")

    # Make a call using OpenAI client (will hit the wrapped fake)
    from openai import OpenAI

    client = OpenAI(api_key="sk-test")
    _ = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])

    # Flush and verify event captured
    out_path = tmp_path / "obs.json"
    observer.end()
    written = observer.flush(str(out_path))
    assert os.path.exists(written)

    data = json.loads(out_path.read_text())
    events = data.get("events", [])
    assert events, "No events captured by OpenAI instrumentation"
    ev = events[0]
    assert ev["provider"] == "openai"
    assert ev["api"] == "chat.completions.create"
    assert ev["response"]["text"] == "hello world"
    assert ev["request"]["model"] == "gpt-4o-mini"
    assert ev["span_id"] is not None  # Verify span tracking is working
    # Note: callsite may be None in test environments due to frame filtering


def test_openai_embeddings_instrumentation(monkeypatch, tmp_path):
    # Prepare a fake create implementation to avoid network calls
    from openai.resources.embeddings import Embeddings

    def fake_create(self, *args, **kwargs):  # noqa: ARG001
        embedding_data = SimpleNamespace(
            index=0,
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
            object="embedding"
        )
        usage = {"prompt_tokens": 5, "total_tokens": 5}
        return SimpleNamespace(
            id=None,
            model="text-embedding-3-small",
            object="list",
            data=[embedding_data],
            usage=usage
        )

    # Monkeypatch BEFORE observe() so the provider wraps our fake
    monkeypatch.setattr(Embeddings, "create", fake_create, raising=True)

    # Start observer (installs provider instrumentation)
    observer.observe("openai-embeddings-test")

    # Make a call using OpenAI client (will hit the wrapped fake)
    from openai import OpenAI

    client = OpenAI(api_key="sk-test")
    _ = client.embeddings.create(model="text-embedding-3-small", input="Hello world")

    # Flush and verify event captured
    out_path = tmp_path / "obs_embeddings.json"
    observer.end()
    written = observer.flush(str(out_path))
    assert os.path.exists(written)

    data = json.loads(out_path.read_text())
    events = data.get("events", [])
    assert events, "No events captured by OpenAI embeddings instrumentation"
    ev = events[0]
    assert ev["provider"] == "openai"
    assert ev["api"] == "embeddings.create"
    assert ev["request"]["model"] == "text-embedding-3-small"
    assert ev["request"]["input"] == "Hello world"
    assert ev["response"]["model"] == "text-embedding-3-small"
    assert ev["response"]["data"] is not None
    assert len(ev["response"]["data"]) == 1
    assert ev["response"]["data"][0]["embedding"] == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert ev["response"]["embedding_dimensions"] == 5
    assert ev["span_id"] is not None  # Verify span tracking is working
