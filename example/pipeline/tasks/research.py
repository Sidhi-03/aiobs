from typing import List


def research(query: str, client, model: str) -> List[str]:
    """Gather concise notes to answer a query."""
    messages = [
        {"role": "system", "content": "You generate brief, factual research notes."},
        {
            "role": "user",
            "content": (
                "Collect 5-7 concise bullet points of facts and references (no links) "
                f"about: {query}. Keep each under 20 words."
            ),
        },
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=300,
    )
    text = resp.choices[0].message.content or ""
    # Very light parsing: split by lines starting with a dash or bullet
    notes = [
        line.strip(" -•\t")
        for line in text.splitlines()
        if line.strip().startswith(("-", "•")) or line.strip()
    ]
    # Keep it to ~7 lines max to feed into the next step
    return [n for n in notes if n][:7]

