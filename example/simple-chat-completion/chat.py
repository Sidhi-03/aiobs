import os
import sys
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

# Ensure the package in repo root is importable when running this example directly
_repo_root = os.path.dirname(os.path.dirname(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
from aiobs import observer  # noqa: E402


def main() -> None:
    # Load env from nearest .env (searching upward)
    load_dotenv(find_dotenv(usecwd=True))

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Missing OPENAI_API_KEY in environment (.env)")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Fixed question to keep things minimal.
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "In one sentence, explain what an API is."},
    ]

    # Start observability, make the LLM call, then end and flush
    observer.observe(
        session_name="simple-chat-completion",
        labels={
            "environment": "development",
            "example": "simple_chat",
            "provider": "openai",
        }
    )

    completion = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=messages,
        temperature=0.7,
        max_tokens=100,
    )

    print(completion.choices[0].message.content.strip())

    observer.end()
    observer.flush()


if __name__ == "__main__":
    main()
