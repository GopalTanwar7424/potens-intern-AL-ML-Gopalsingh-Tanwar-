"""
Triage Agent -- real tool calling via the Anthropic Messages API.

The model is given tool *definitions* only. It must decide, turn by
turn, which tool (if any) to call, with what arguments, based on the
actual ticket text -- there is no keyword/if-else shortcut that
determines the category directly. The full transcript (assistant
text, tool_use blocks, tool_result blocks) is preserved as the
reasoning trace.

Usage:
    export ANTHROPIC_API_KEY=sk-...      # real Claude tool calling, OR
    export GROQ_API_KEY=gsk_...          # real Groq tool calling (free tier)
    python agent.py "My card was charged twice this month, please fix it"

If neither key is set, falls back to mock_llm.py so the project
remains runnable offline for demo purposes (this fallback is loudly
labeled -- see README "Mock mode" section). If GROQ_API_KEY is set,
Groq's OpenAI-compatible tool-calling API is used. If both keys are
set, Anthropic is preferred.
"""
from __future__ import annotations
import json
import os
import sys
import uuid
from typing import Optional

from schemas import TriageOutput, ToolCallRecord, CATEGORIES, PRIORITIES, NEXT_TOOLS
from tools import TOOL_REGISTRY

MODEL = "claude-sonnet-4-5"
GROQ_MODEL = "openai/gpt-oss-120b"
MAX_TURNS = 6

SYSTEM_PROMPT = f"""You are a support-ticket triage agent.

Classify the incoming text into exactly one category from this fixed list:
{json.dumps(CATEGORIES)}

Assign exactly one priority from: {json.dumps(PRIORITIES)}
- P0: urgent / active harm (security breach, data loss, money at risk, customer explicitly threatening to churn immediately)
- P1: important, blocking the user, needs attention within a business day
- P2: normal queue, no urgency (questions, minor issues, feature requests)

You have tools available: kb_lookup, similar_ticket_search, draft_acknowledgment,
and request_human_review. Use them to gather real evidence before deciding --
for example, look up whether this matches a known issue, or check how similar
past tickets were resolved and at what priority. Call request_human_review
if, after using the other tools, you are still genuinely unsure about the
category or priority.

You do not have to call every tool. Call the ones that actually help you
decide. Think step by step in your text output between tool calls -- that
text becomes the visible reasoning trace, so narrate your reasoning honestly
(what you're checking and why), don't just call tools silently.

When you are done, respond with ONLY a JSON object (no markdown fences, no
other text) with exactly these keys:
{{
  "category": one of {CATEGORIES},
  "priority": one of {PRIORITIES},
  "next_tool": one of {NEXT_TOOLS} (the recommended next action for a human/system, "none" if fully handled),
  "why": "one paragraph explaining the decision, referencing the evidence you gathered",
  "confidence": a number between 0 and 1
}}
"""

TOOLS_SPEC = [
    {
        "name": "kb_lookup",
        "description": "Search the internal knowledge base for known issues or documented solutions relevant to the ticket text.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "search text"}},
            "required": ["query"],
        },
    },
    {
        "name": "similar_ticket_search",
        "description": "Search past resolved tickets for similar incidents, returning their category/priority/resolution as evidence.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "the ticket text to compare against"}},
            "required": ["query"],
        },
    },
    {
        "name": "draft_acknowledgment",
        "description": "Draft a customer-facing acknowledgment message for a given category/priority.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string"},
                "priority": {"type": "string"},
                "customer_name": {"type": "string"},
            },
            "required": ["category", "priority"],
        },
    },
    {
        "name": "request_human_review",
        "description": "Escalate the ticket to a human reviewer when confidence is low, with a reason.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "priority": {"type": "string"},
            },
            "required": ["reason"],
        },
    },
]


# Groq (and OpenAI-compatible APIs generally) want tools wrapped as
# {"type": "function", "function": {...}} rather than Anthropic's flatter
# {"name", "description", "input_schema"} shape. Same schemas, different envelope.
GROQ_TOOLS_SPEC = [
    {"type": "function", "function": {"name": t["name"], "description": t["description"],
                                       "parameters": t["input_schema"]}}
    for t in TOOLS_SPEC
]


def _get_provider():
    """Pick a provider based on which key is set. Anthropic wins if both are present."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic  # imported lazily so the mock path needs no dependency
        return "anthropic", anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    if os.environ.get("GROQ_API_KEY"):
        import groq  # imported lazily so the mock path needs no dependency
        return "groq", groq.Groq(api_key=os.environ["GROQ_API_KEY"])
    return None, None


def _recover_json_from_groq_error(e) -> Optional[dict]:
    """Best-effort recovery when Groq rejects a malformed pseudo tool call
    (e.g. a call to a fake tool named 'json'). The model's real decision is
    usually still sitting in error.body['error']['failed_generation']."""
    body = getattr(e, "body", None)
    if not isinstance(body, dict):
        return None
    failed_gen = body.get("error", {}).get("failed_generation")
    if not failed_gen:
        return None
    try:
        parsed = json.loads(failed_gen)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        if isinstance(parsed.get("arguments"), dict):
            return parsed["arguments"]
        if "category" in parsed:
            return parsed
    return None


def _trace_entry_to_text(t: dict) -> str:
    """Render one trace entry as a human-readable reasoning-trace line,
    regardless of which type it is (reasoning / tool_call / parse_error)."""
    if t["type"] == "reasoning":
        return t["text"]
    if t["type"] == "tool_call":
        return f"[tool:{t['tool']}] input={t['input']} -> output={t['output']}"
    if t["type"] == "parse_error":
        return f"[parse_error, turn {t.get('turn')}] model did not return valid JSON: {t.get('raw', '')}"
    return f"[{t.get('type', 'unknown')}] {t}"


def run_agent(ticket_text: str, metadata: Optional[dict] = None, ticket_id: Optional[str] = None) -> dict:
    metadata = metadata or {}
    ticket_id = ticket_id or f"TCK-{uuid.uuid4().hex[:8]}"

    provider, client = _get_provider()
    if client is None:
        # No API key available -- fall back to the offline mock engine so
        # the project still runs end to end. See README "Mock mode".
        from mock_llm import run_mock_agent
        return run_mock_agent(ticket_text, metadata, ticket_id)

    if provider == "groq":
        return _run_agent_groq(client, ticket_text, metadata, ticket_id)
    return _run_agent_anthropic(client, ticket_text, metadata, ticket_id)


def _run_agent_anthropic(client, ticket_text: str, metadata: dict, ticket_id: str) -> dict:
    user_content = f"Ticket text: {ticket_text}\nMetadata: {json.dumps(metadata)}"
    messages = [{"role": "user", "content": user_content}]
    trace = []          # human-readable reasoning trace
    tool_calls = []      # ToolCallRecord list

    final_json = None
    for turn in range(MAX_TURNS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            tools=TOOLS_SPEC,
            messages=messages,
        )

        assistant_blocks = []
        tool_use_blocks = []
        for block in response.content:
            if block.type == "text" and block.text.strip():
                trace.append({"type": "reasoning", "turn": turn, "text": block.text.strip()})
                assistant_blocks.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                tool_use_blocks.append(block)
                assistant_blocks.append({"type": "tool_use", "id": block.id,
                                          "name": block.name, "input": block.input})

        messages.append({"role": "assistant", "content": assistant_blocks})

        if response.stop_reason == "tool_use":
            tool_result_blocks = []
            for block in tool_use_blocks:
                fn = TOOL_REGISTRY.get(block.name)
                result = fn(**block.input) if fn else {"error": f"unknown tool {block.name}"}
                trace.append({"type": "tool_call", "turn": turn, "tool": block.name,
                               "input": block.input, "output": result})
                tool_calls.append(ToolCallRecord(tool_name=block.name, tool_input=block.input,
                                                  tool_output=result))
                tool_result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })
            messages.append({"role": "user", "content": tool_result_blocks})
            continue

        # No more tool calls -- try to parse final JSON from the text
        text_out = "".join(b["text"] for b in assistant_blocks if b["type"] == "text").strip()
        text_out = text_out.strip("`")
        if text_out.startswith("json"):
            text_out = text_out[4:].strip()
        try:
            final_json = json.loads(text_out)
        except json.JSONDecodeError:
            trace.append({"type": "parse_error", "turn": turn, "raw": text_out})
        break

    if final_json is None:
        # Model never produced valid JSON within MAX_TURNS -- escalate.
        final_json = {
            "category": "General Inquiry",
            "priority": "P1",
            "next_tool": "request_human_review",
            "why": "Agent did not converge on a structured decision within the turn budget; escalating for human review.",
            "confidence": 0.2,
        }

    escalated = final_json.get("next_tool") == "request_human_review" or final_json.get("confidence", 1) < 0.5

    output = TriageOutput(
        ticket_id=ticket_id,
        input_text=ticket_text,
        metadata=metadata,
        category=final_json.get("category", "General Inquiry"),
        priority=final_json.get("priority", "P2"),
        next_tool=final_json.get("next_tool", "none"),
        reasoning=[_trace_entry_to_text(t) for t in trace],
        why=final_json.get("why", ""),
        confidence=float(final_json.get("confidence", 0.5)),
        tool_calls=[tc.__dict__ for tc in tool_calls],
        escalated=escalated,
        engine="claude",
    )
    return {"output": output.to_dict(), "full_trace": trace}


def _run_agent_groq(client, ticket_text: str, metadata: dict, ticket_id: str) -> dict:
    """Same reasoning loop as _run_agent_anthropic, adapted to Groq's
    OpenAI-compatible chat.completions + tool_calls shape."""
    user_content = f"Ticket text: {ticket_text}\nMetadata: {json.dumps(metadata)}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    trace = []
    tool_calls = []

    final_json = None
    for turn in range(MAX_TURNS):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                max_tokens=1500,
                tools=GROQ_TOOLS_SPEC,
                messages=messages,
            )
        except Exception as e:
            # gpt-oss-120b occasionally tries to "call" a pseudo-tool named
            # "json" to return its final answer instead of plain text, which
            # Groq's API rejects since "json" isn't a declared tool. The
            # actual decision is still present in the error body -- recover
            # it instead of failing the whole run.
            recovered = _recover_json_from_groq_error(e)
            if recovered:
                trace.append({"type": "reasoning", "turn": turn,
                               "text": "[recovered decision from a malformed tool call]"})
                final_json = recovered
                break
            raise

        msg = response.choices[0].message
        text_out = (msg.content or "").strip()

        if text_out:
            trace.append({"type": "reasoning", "turn": turn, "text": text_out})

        # Append the assistant turn exactly as returned so Groq can see its
        # own prior tool_calls on the next turn.
        assistant_entry = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        messages.append(assistant_entry)

        if msg.tool_calls:
            for tc in msg.tool_calls:
                fn = TOOL_REGISTRY.get(tc.function.name)
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    args = {}
                result = fn(**args) if fn else {"error": f"unknown tool {tc.function.name}"}
                trace.append({"type": "tool_call", "turn": turn, "tool": tc.function.name,
                               "input": args, "output": result})
                tool_calls.append(ToolCallRecord(tool_name=tc.function.name, tool_input=args,
                                                  tool_output=result))
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                })
            continue

        # No more tool calls -- try to parse final JSON from the text
        clean = text_out.strip("`")
        if clean.startswith("json"):
            clean = clean[4:].strip()
        try:
            final_json = json.loads(clean)
        except json.JSONDecodeError:
            trace.append({"type": "parse_error", "turn": turn, "raw": clean})
        break

    if final_json is None:
        final_json = {
            "category": "General Inquiry",
            "priority": "P1",
            "next_tool": "request_human_review",
            "why": "Agent did not converge on a structured decision within the turn budget; escalating for human review.",
            "confidence": 0.2,
        }

    escalated = final_json.get("next_tool") == "request_human_review" or final_json.get("confidence", 1) < 0.5

    output = TriageOutput(
        ticket_id=ticket_id,
        input_text=ticket_text,
        metadata=metadata,
        category=final_json.get("category", "General Inquiry"),
        priority=final_json.get("priority", "P2"),
        next_tool=final_json.get("next_tool", "none"),
        reasoning=[_trace_entry_to_text(t) for t in trace],
        why=final_json.get("why", ""),
        confidence=float(final_json.get("confidence", 0.5)),
        tool_calls=[tc.__dict__ for tc in tool_calls],
        escalated=escalated,
        engine="groq",
    )
    return {"output": output.to_dict(), "full_trace": trace}


if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else "My app crashes every time I try to upload a receipt photo."
    result = run_agent(text)
    print(json.dumps(result, indent=2, default=str))