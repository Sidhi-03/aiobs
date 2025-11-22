import os
import sys
from typing import Optional

from dotenv import load_dotenv, find_dotenv
from openai import OpenAI


def make_client() -> OpenAI:
    # Load env from nearest .env (search upwards from CWD)
    load_dotenv(find_dotenv(usecwd=True))

    api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Missing OPENAI_API_KEY in environment (.env)")
        sys.exit(1)

    return OpenAI(api_key=api_key)


def default_model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")

