import os
import sys


# Ensure repo root is importable (so tests can import llm_observability)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest


@pytest.fixture(autouse=True)
def reset_observer_state():
    # Fresh collector state for each test
    from llm_observability import observer

    observer.reset()
    try:
        yield
    finally:
        observer.reset()
