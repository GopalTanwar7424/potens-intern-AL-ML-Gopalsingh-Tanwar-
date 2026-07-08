"""
STRETCH GOAL: Streamlit UI that visualizes a triage run's reasoning
trace as a tree -- ticket -> reasoning/tool-call steps in order ->
final decision.

    pip install streamlit
    streamlit run streamlit_app.py
"""
import json
import os
import streamlit as st

from agent import run_agent

HERE = os.path.dirname(os.path.abspath(__file__))
COMBINED_PATH = os.path.join(HERE, "examples", "combined_results.json")

st.set_page_config(page_title="Triage Agent - Reasoning Tree", layout="wide")
st.title("🧭 Triage Agent — Reasoning Trace Viewer")

mode = st.sidebar.radio("Mode", ["Browse saved examples", "Run a new ticket"])


def render_tree(result):
    out = result["output"]
    trace = result["full_trace"]

    st.subheader(f"🎫 {out['ticket_id']}")
    st.write(f"**Input:** {out['input_text']}")

    cols = st.columns(4)
    cols[0].metric("Category", out["category"])
    cols[1].metric("Priority", out["priority"])
    cols[2].metric("Confidence", f"{out['confidence']:.2f}")
    cols[3].metric("Engine", out["engine"])

    st.markdown("---")
    st.markdown("**Reasoning tree**")
    st.markdown(f"```\n🎫 Ticket: {out['ticket_id']}\n```")
    for i, step in enumerate(trace):
        indent = "    " * 1
        if step["type"] == "reasoning":
            st.markdown(f"{indent}├── 💭 **Reasoning:** {step['text']}")
        elif step["type"] == "tool_call":
            with st.expander(f"{indent}├── 🔧 Tool call: {step['tool']}"):
                st.json({"input": step["input"], "output": step["output"]})
        else:
            st.markdown(f"{indent}├── ⚠️ {step.get('type')}: {step}")
    st.markdown(f"    └── ✅ **Decision:** {out['category']} / {out['priority']} → `{out['next_tool']}`")

    st.markdown("---")
    st.markdown(f"**Why:** {out['why']}")
    if out.get("escalated"):
        st.warning("This ticket was escalated to a human reviewer (low confidence).")


if mode == "Browse saved examples":
    if not os.path.exists(COMBINED_PATH):
        st.info("No saved examples yet. Run `python run_examples.py` first.")
    else:
        with open(COMBINED_PATH) as f:
            combined = json.load(f)
        ids = [r["output"]["ticket_id"] for r in combined]
        choice = st.sidebar.selectbox("Pick a ticket", ids)
        result = next(r for r in combined if r["output"]["ticket_id"] == choice)
        render_tree(result)
else:
    text = st.text_area("Ticket text", "My app keeps crashing when I try to upload a file.")
    name = st.text_input("Customer name (optional)")
    if st.button("Run triage agent"):
        with st.spinner("Running agent..."):
            result = run_agent(text, metadata={"customer_name": name} if name else {})
        render_tree(result)
