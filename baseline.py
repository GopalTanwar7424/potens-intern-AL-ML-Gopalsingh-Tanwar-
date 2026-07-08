"""
STRETCH GOAL: baseline single-prompt classifier.

Same ten examples, but no tools, no multi-turn reasoning, no evidence
gathering -- one prompt in, one JSON out. Used to show, side by side,
what the tool-calling agent buys you over a plain classifier (e.g. it
can't cite a similar past ticket or a KB article, and has no
mechanism to escalate low-confidence cases to a human).

    python baseline.py
"""
import json
import os
from schemas import CATEGORIES, PRIORITIES

HERE = os.path.dirname(os.path.abspath(__file__))
INPUTS_PATH = os.path.join(HERE, "examples", "inputs.json")

BASELINE_SYSTEM_PROMPT = f"""Classify the ticket into one category from {CATEGORIES}
and one priority from {PRIORITIES}. Respond ONLY with JSON: {{"category": ..., "priority": ..., "why": ...}}"""


def _get_provider():
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        return "anthropic", anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    if os.environ.get("GROQ_API_KEY"):
        import groq
        return "groq", groq.Groq(api_key=os.environ["GROQ_API_KEY"])
    return None, None


def classify_baseline_claude(client, text):
    resp = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=300,
        system=BASELINE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text}],
    )
    raw = "".join(b.text for b in resp.content if b.type == "text").strip().strip("`")
    if raw.startswith("json"):
        raw = raw[4:].strip()
    return json.loads(raw)


def classify_baseline_groq(client, text):
    resp = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        max_tokens=300,
        messages=[
            {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    raw = (resp.choices[0].message.content or "").strip().strip("`")
    if raw.startswith("json"):
        raw = raw[4:].strip()
    return json.loads(raw)


def classify_baseline_mock(text):
    """Offline stand-in: same keyword-hint table as mock_llm, but with
    NO tool evidence at all -- purely string matching on the raw text.
    This intentionally represents the *weakest* possible baseline."""
    from mock_llm import _CATEGORY_HINTS
    lower = text.lower()
    scores = {cat: sum(1 for kw in kws if kw in lower) for cat, kws in _CATEGORY_HINTS.items()}
    category = max(scores, key=scores.get) if max(scores.values()) > 0 else "General Inquiry"
    urgent = any(w in lower for w in ["hacked", "breach", "cancel", "third time", "furious", "unauthorized"])
    priority = "P0" if urgent else ("P1" if scores.get(category, 0) > 0 else "P2")
    return {"category": category, "priority": priority,
            "why": "Pure keyword match on raw text, no evidence gathering, no tools."}


def main():
    with open(INPUTS_PATH) as f:
        examples = json.load(f)

    provider, client = _get_provider()
    engine = provider or "mock"

    results = []
    correct = 0
    print(f"{'ID':<8}{'Baseline Category':<24}{'Expected':<24}Match")
    print("-" * 70)
    for ex in examples:
        if provider == "anthropic":
            pred = classify_baseline_claude(client, ex["text"])
        elif provider == "groq":
            pred = classify_baseline_groq(client, ex["text"])
        else:
            pred = classify_baseline_mock(ex["text"])
        match = pred["category"] == ex.get("expected_category")
        correct += match
        results.append({"ticket_id": ex["ticket_id"], "text": ex["text"],
                         "expected": ex.get("expected_category"), "baseline": pred, "engine": engine})
        print(f"{ex['ticket_id']:<8}{pred['category']:<24}{ex.get('expected_category',''):<24}{match}")

    print("-" * 70)
    print(f"Baseline accuracy vs hand-labeled expectation: {correct}/{len(examples)}  (engine={engine})")

    out_path = os.path.join(HERE, "examples", "baseline_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()