"""
OFFLINE DEMO FALLBACK -- NOT the real agent.

agent.py's real path sends the ticket to Claude with genuine tool-use:
the model decides which of the 3-4 tools to call and how to interpret
their output, then produces the structured decision. That requires
ANTHROPIC_API_KEY and network access.

This module exists only so the project can still be *run and inspected*
in environments without API access (e.g. an offline sandbox). It still
calls the exact same real tool functions from tools.py and builds the
same trace/output shape, but the "brain" deciding category/priority
here is a transparent scoring heuristic over the tools' own similarity
scores -- not the LLM. Treat any /examples output with
`"engine": "mock"` as a stand-in for what the real Claude-driven run
would produce; re-run `python run_examples.py` with an API key set to
get the genuine LLM-tool-calling trace.
"""
from __future__ import annotations
from typing import Optional

from schemas import TriageOutput, ToolCallRecord
from tools import kb_lookup, similar_ticket_search, draft_acknowledgment, request_human_review

_CATEGORY_HINTS = {
    "Billing": ["charge", "refund", "invoice", "payment", "subscription", "billed", "price", "card"],
    "Technical Bug": ["crash", "error", "bug", "broken", "not working", "fail", "freeze", "sync"],
    "Account & Security": ["login", "password", "account", "hacked", "suspicious", "breach", "2fa", "access"],
    "Feature Request": ["would be great", "please add", "feature", "suggestion", "wish", "could you add"],
    "Complaint / Escalation": ["cancel", "unacceptable", "furious", "third time", "again", "terrible", "worst"],
    "General Inquiry": ["how do i", "how can i", "what is", "where is", "question"],
}


def run_mock_agent(ticket_text: str, metadata: Optional[dict], ticket_id: str) -> dict:
    metadata = metadata or {}
    trace = []
    tool_calls = []

    def log_tool(name, inp, out):
        trace.append({"type": "tool_call", "tool": name, "input": inp, "output": out})
        tool_calls.append(ToolCallRecord(tool_name=name, tool_input=inp, tool_output=out))

    trace.append({"type": "reasoning", "text": "[MOCK ENGINE] Step 1: search knowledge base for a matching known issue."})
    kb = kb_lookup(ticket_text)
    log_tool("kb_lookup", {"query": ticket_text}, kb)

    trace.append({"type": "reasoning", "text": "[MOCK ENGINE] Step 2: search past tickets for similar historical incidents."})
    sim = similar_ticket_search(ticket_text)
    log_tool("similar_ticket_search", {"query": ticket_text}, sim)

    best_kb = kb["matches"][0] if kb["matches"] else None
    best_sim = sim["matches"][0] if sim["matches"] else None

    lower = ticket_text.lower()
    hint_scores = {cat: sum(1 for kw in kws if kw in lower) for cat, kws in _CATEGORY_HINTS.items()}
    hinted_cat = max(hint_scores, key=hint_scores.get) if max(hint_scores.values()) > 0 else None

    # Decide category: prefer strong similar-ticket evidence, else KB, else keyword hint, else General Inquiry
    candidates = []
    if best_sim and best_sim["score"] > 0.35:
        candidates.append((best_sim["score"] + 0.05, best_sim["category"]))
    if best_kb and best_kb["score"] > 0.3:
        candidates.append((best_kb["score"], best_kb["category"]))
    if hinted_cat:
        candidates.append((0.3, hinted_cat))

    if candidates:
        candidates.sort(reverse=True)
        confidence, category = candidates[0]
        confidence = min(0.95, max(0.3, confidence))
    else:
        category, confidence = "General Inquiry", 0.35

    trace.append({"type": "reasoning", "text": (
        f"[MOCK ENGINE] Step 3: best similar ticket match={best_sim}, best KB match={best_kb}, "
        f"keyword hint={hinted_cat}. Selecting category='{category}' with confidence={confidence:.2f}."
    )})

    # Priority: escalate hard signals regardless of category
    urgent_signals = ["hacked", "breach", "unauthorized", "unrecognized login", "suspicious", "cancel", "third time", "furious"]
    if any(sig in lower for sig in urgent_signals) or category == "Complaint / Escalation":
        priority = "P0"
    elif best_sim and best_sim["priority"] in ("P0", "P1"):
        priority = best_sim["priority"]
    elif category in ("Technical Bug", "Billing", "Account & Security"):
        priority = "P1"
    else:
        priority = "P2"

    trace.append({"type": "reasoning", "text": f"[MOCK ENGINE] Step 4: priority signals evaluated -> priority='{priority}'."})

    escalated = confidence < 0.5
    if escalated:
        reason = f"Low confidence ({confidence:.2f}) classifying this ticket; category/priority uncertain."
        esc = request_human_review(reason=reason, priority=priority)
        log_tool("request_human_review", {"reason": reason, "priority": priority}, esc)
        next_tool = "request_human_review"
        why = (f"Confidence was below threshold after checking KB and past tickets "
               f"(best similar ticket score={best_sim['score'] if best_sim else 0}), so this was escalated to a human reviewer.")
    else:
        ack = draft_acknowledgment(category=category, priority=priority,
                                    customer_name=metadata.get("customer_name"))
        log_tool("draft_acknowledgment", {"category": category, "priority": priority,
                                           "customer_name": metadata.get("customer_name")}, ack)
        next_tool = "draft_acknowledgment"
        evidence = []
        if best_sim and best_sim["score"] > 0.35:
            evidence.append(f"similar past ticket {best_sim['id']} (resolved as {best_sim['category']}/{best_sim['priority']})")
        if best_kb and best_kb["score"] > 0.3:
            evidence.append(f"knowledge-base article {best_kb['id']}")
        evidence_str = " and ".join(evidence) if evidence else "keyword signal in the ticket text"
        why = (f"Classified as '{category}' at priority '{priority}' based on {evidence_str}. "
               f"An acknowledgment was drafted and priority was set from the observed urgency signals.")

    output = TriageOutput(
        ticket_id=ticket_id,
        input_text=ticket_text,
        metadata=metadata,
        category=category,
        priority=priority,
        next_tool=next_tool,
        reasoning=[t["text"] if t["type"] == "reasoning" else f"[tool:{t['tool']}] input={t['input']} -> output={t['output']}"
                   for t in trace],
        why=why,
        confidence=round(confidence, 2),
        tool_calls=[tc.__dict__ for tc in tool_calls],
        escalated=escalated,
        engine="mock",
    )
    return {"output": output.to_dict(), "full_trace": trace}
