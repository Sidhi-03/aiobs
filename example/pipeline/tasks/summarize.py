from typing import List


def summarize(notes: List[str], client, model: str) -> str:
    """Summarize research notes into a short answer."""
    notes_text = "\n".join(f"- {n}" for n in notes)
    messages = [
        {"role": "system", "content": "You are a concise, accurate technical writer."},
        {
            "role": "user",
            "content": (
                "Given these notes, produce a crisp 4-6 sentence summary with no bullet points.\n\n"
                f"Notes:\n{notes_text}"
            ),
        },
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=250,
    )
    return (resp.choices[0].message.content or "").strip()

