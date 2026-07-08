"""
Real, callable tools for the Triage Agent.

These are genuine Python functions with their own logic and small
datasets. The LLM decides *whether* and *when* to call them and what
arguments to pass -- it never gets to hardcode the classification
itself via these tools; the tools only supply supporting evidence
(knowledge-base hits, past-ticket similarity, a drafted reply, or a
human-escalation hand-off) that the agent must reason over.
"""
from __future__ import annotations
import difflib
import time
import uuid
from typing import Optional


# ---------------------------------------------------------------------------
# Tool 1: kb_lookup
# A small internal knowledge base the agent can search for known issues /
# documented solutions. Real text-similarity search (difflib), not a
# lookup table the agent can just echo back as the final answer.
# ---------------------------------------------------------------------------

_KB_ARTICLES = [
    {
        "id": "KB-101",
        "title": "How to update payment method",
        "category": "Billing",
        "body": "Customers can update their card in Settings > Billing. "
                 "Failed charges auto-retry 3 times over 5 days.",
    },
    {
        "id": "KB-102",
        "title": "Refund policy",
        "category": "Billing",
        "body": "Refunds are issued for duplicate charges or cancellations "
                 "within 14 days. Processing takes 5-7 business days.",
    },
    {
        "id": "KB-201",
        "title": "App crashes on file upload",
        "category": "Technical Bug",
        "body": "Known issue with files over 25MB on the mobile app version "
                 "< 4.2. Fixed in 4.3. Workaround: upload via web.",
    },
    {
        "id": "KB-202",
        "title": "Data not syncing across devices",
        "category": "Technical Bug",
        "body": "Sync delays can occur if the account has >50k records. "
                 "Force sync via Settings > Sync > Force Refresh.",
    },
    {
        "id": "KB-301",
        "title": "Password reset flow",
        "category": "Account & Security",
        "body": "Reset link is emailed and expires in 30 minutes. If not "
                 "received, check spam or request a manual reset from support.",
    },
    {
        "id": "KB-302",
        "title": "Suspicious login alerts",
        "category": "Account & Security",
        "body": "Unrecognized-device alerts should be escalated immediately. "
                 "Do not close the ticket without confirming the user's identity.",
    },
    {
        "id": "KB-401",
        "title": "Feature request intake process",
        "category": "Feature Request",
        "body": "Feature requests are logged in the product backlog and "
                 "reviewed monthly. No individual ETA is promised.",
    },
]


def kb_lookup(query: str, top_k: int = 2) -> dict:
    """Search the internal knowledge base for the most relevant articles."""
    scored = []
    for art in _KB_ARTICLES:
        text = f"{art['title']} {art['body']}"
        score = difflib.SequenceMatcher(None, query.lower(), text.lower()).ratio()
        scored.append((score, art))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]
    return {
        "query": query,
        "matches": [
            {"id": a["id"], "title": a["title"], "category": a["category"],
             "body": a["body"], "score": round(s, 3)}
            for s, a in top
        ],
    }


# ---------------------------------------------------------------------------
# Tool 2: similar_ticket_search
# Searches a small archive of past resolved tickets for similar past
# incidents and how they were resolved. Real similarity scoring, real
# historical outcomes -- gives the agent evidence, not the answer itself.
# ---------------------------------------------------------------------------

_PAST_TICKETS = [
    {
        "id": "T-1001", "text": "I was charged twice for my subscription this month",
        "category": "Billing", "priority": "P1",
        "resolution": "Confirmed duplicate charge, refunded within 3 days.",
    },
    {
        "id": "T-1002", "text": "The app keeps crashing every time I try to upload a photo",
        "category": "Technical Bug", "priority": "P1",
        "resolution": "Identified as known upload bug, advised web workaround, escalated to eng.",
    },
    {
        "id": "T-1003", "text": "I can't log in, it says my password is wrong but I'm sure it's right",
        "category": "Account & Security", "priority": "P1",
        "resolution": "Sent manual password reset link, confirmed identity via email.",
    },
    {
        "id": "T-1004", "text": "Someone logged into my account from a country I've never been to",
        "category": "Account & Security", "priority": "P0",
        "resolution": "Forced logout of all sessions, reset password, enabled 2FA, escalated to security team.",
    },
    {
        "id": "T-1005", "text": "It would be great if you added dark mode",
        "category": "Feature Request", "priority": "P2",
        "resolution": "Logged in backlog, no immediate action.",
    },
    {
        "id": "T-1006", "text": "This is the third time I've had this billing issue, I'm cancelling",
        "category": "Complaint / Escalation", "priority": "P0",
        "resolution": "Escalated to retention specialist same day, issued goodwill credit.",
    },
    {
        "id": "T-1007", "text": "How do I export my data to a CSV file?",
        "category": "General Inquiry", "priority": "P2",
        "resolution": "Pointed to Settings > Export, resolved in one reply.",
    },
]


def similar_ticket_search(query: str, top_k: int = 3) -> dict:
    """Find past resolved tickets most similar to the current input."""
    scored = []
    for t in _PAST_TICKETS:
        score = difflib.SequenceMatcher(None, query.lower(), t["text"].lower()).ratio()
        scored.append((score, t))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]
    return {
        "query": query,
        "matches": [
            {"id": t["id"], "text": t["text"], "category": t["category"],
             "priority": t["priority"], "resolution": t["resolution"],
             "score": round(s, 3)}
            for s, t in top
        ],
    }


# ---------------------------------------------------------------------------
# Tool 3: draft_acknowledgment
# Produces a real, ready-to-send customer acknowledgment based on the
# category/priority the agent has decided on.
# ---------------------------------------------------------------------------

_TEMPLATES = {
    "P0": "Hi{name}, thank you for flagging this -- we understand this is "
          "urgent and it's been escalated immediately to our {team} team. "
          "You'll hear back within the hour.",
    "P1": "Hi{name}, thanks for reaching out. We've logged this as a {category} "
          "issue and it's in our priority queue -- expect a response within "
          "one business day.",
    "P2": "Hi{name}, thanks for your message! We've logged this {category} "
          "item and will follow up as soon as we can.",
}

_TEAM_BY_CATEGORY = {
    "Billing": "billing",
    "Technical Bug": "engineering",
    "Account & Security": "security",
    "Feature Request": "product",
    "Complaint / Escalation": "retention",
    "General Inquiry": "support",
}


def draft_acknowledgment(category: str, priority: str, customer_name: Optional[str] = None) -> dict:
    """Draft a customer-facing acknowledgment message."""
    name_part = f" {customer_name}" if customer_name else ""
    template = _TEMPLATES.get(priority, _TEMPLATES["P2"])
    team = _TEAM_BY_CATEGORY.get(category, "support")
    message = template.format(name=name_part, team=team, category=category)
    return {"category": category, "priority": priority, "draft": message}


# ---------------------------------------------------------------------------
# Tool 4 (stretch): request_human_review
# Low-confidence escalation path -- hands the ticket to a human queue.
# ---------------------------------------------------------------------------

_HUMAN_QUEUES = {
    "P0": "tier2-oncall",
    "P1": "tier2-support",
    "P2": "tier1-support",
}


def request_human_review(reason: str, priority: str = "P1") -> dict:
    """Escalate to a human reviewer when the agent's confidence is low."""
    return {
        "escalated": True,
        "review_id": f"HR-{uuid.uuid4().hex[:8]}",
        "queue": _HUMAN_QUEUES.get(priority, "tier1-support"),
        "reason": reason,
        "requested_at": time.time(),
    }


TOOL_REGISTRY = {
    "kb_lookup": kb_lookup,
    "similar_ticket_search": similar_ticket_search,
    "draft_acknowledgment": draft_acknowledgment,
    "request_human_review": request_human_review,
}
