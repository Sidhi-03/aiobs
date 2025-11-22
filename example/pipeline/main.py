import os
import sys
from typing import Optional

# Make repo root importable for local package (llm_observability)
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from llm_observability import observer  # noqa: E402

# Support running as a module or a script
try:  # package-relative imports (when run with -m example.pipeline.main)
    from .client import make_client, default_model  # type: ignore
    from .tasks.research import research  # type: ignore
    from .tasks.summarize import summarize  # type: ignore
    from .tasks.critique import critique  # type: ignore
except Exception:  # fallback for direct script run: python example/pipeline/main.py
    _example_dir = os.path.dirname(os.path.dirname(__file__))
    if _example_dir not in sys.path:
        sys.path.insert(0, _example_dir)
    from pipeline.client import make_client, default_model  # type: ignore
    from pipeline.tasks.research import research  # type: ignore
    from pipeline.tasks.summarize import summarize  # type: ignore
    from pipeline.tasks.critique import critique  # type: ignore


def main(query: Optional[str] = None) -> None:
    client = make_client()
    model = default_model()

    # Start a single observability session for the whole pipeline
    observer.observe(session_name="pipeline-example")

    try:
        q = query or "In one sentence, explain what an API is."
        print(f"Query: {q}\n")

        notes = research(q, client, model)
        print("Notes:")
        for n in notes:
            print(f"- {n}")
        print()

        draft = summarize(notes, client, model)
        print("Draft:\n" + draft + "\n")

        improved = critique(draft, client, model)
        print("Improved:\n" + improved + "\n")
    finally:
        observer.end()
        out = observer.flush()
        print(f"Observability written to: {out}")


if __name__ == "__main__":
    # Accept an optional query as a CLI arg
    arg_query = sys.argv[1] if len(sys.argv) > 1 else None
    main(arg_query)
