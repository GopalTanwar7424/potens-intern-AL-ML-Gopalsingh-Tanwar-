"""
Shared data structures for the Triage Agent.

Kept as plain dataclasses (no pydantic dependency) so the project
runs with zero extra installs beyond the anthropic SDK.
"""
from dataclasses import dataclass, field, asdict
from typing import Any, Optional
import time

# ---- Fixed taxonomy -------------------------------------------------------

CATEGORIES = [
    "Billing",              # payments, invoices, refunds, subscription charges
    "Technical Bug",        # something is broken / erroring / crashing
    "Account & Security",   # login, access, breach, suspicious activity
    "Feature Request",      # asking for new functionality
    "Complaint / Escalation",  # unhappy customer, threatening to churn, repeat issue
    "General Inquiry",      # how-to questions, information requests
]

# P0 = drop everything (security/data/outage/money-at-risk, active harm)
# P1 = important, needs same/next business day attention (blocking a user)
# P2 = normal queue, no urgency (questions, feature requests, minor issues)
PRIORITIES = ["P0", "P1", "P2"]

NEXT_TOOLS = [
    "kb_lookup",
    "similar_ticket_search",
    "draft_acknowledgment",
    "request_human_review",
    "none",
]


@dataclass
class ToolCallRecord:
    """One tool invocation the agent made, with real input/output."""
    tool_name: str
    tool_input: dict
    tool_output: Any
    timestamp: float = field(default_factory=time.time)


@dataclass
class TriageOutput:
    ticket_id: str
    input_text: str
    metadata: dict
    category: str
    priority: str
    next_tool: str
    reasoning: list          # ordered list of natural-language reasoning steps
    why: str                 # single-paragraph "why" summary, always populated
    confidence: float        # 0-1
    tool_calls: list         # list[ToolCallRecord] (serialized)
    escalated: bool = False
    engine: str = "claude"   # "claude" or "groq" (real tool calling) or "mock" (offline demo)

    def to_dict(self):
        d = asdict(self)
        return d
