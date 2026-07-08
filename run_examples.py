"""
Runs the triage agent over every ticket in examples/inputs.json and
writes one output JSON per ticket into examples/outputs/, plus a
combined_results.json and a printed summary table.

    python run_examples.py
"""
import json
import os
from agent import run_agent

HERE = os.path.dirname(os.path.abspath(__file__))
INPUTS_PATH = os.path.join(HERE, "examples", "inputs.json")
OUTPUTS_DIR = os.path.join(HERE, "examples", "outputs")


def main():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(INPUTS_PATH) as f:
        examples = json.load(f)

    combined = []
    print(f"{'ID':<8}{'Category':<24}{'Prio':<6}{'NextTool':<22}{'Conf':<6}{'Esc':<5}Engine")
    print("-" * 90)
    for ex in examples:
        result = run_agent(ex["text"], metadata=ex.get("metadata", {}), ticket_id=ex["ticket_id"])
        out = result["output"]
        out["expected_category"] = ex.get("expected_category")
        combined.append(result)

        out_path = os.path.join(OUTPUTS_DIR, f"{ex['ticket_id']}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        print(f"{out['ticket_id']:<8}{out['category']:<24}{out['priority']:<6}"
              f"{out['next_tool']:<22}{out['confidence']:<6}{str(out['escalated']):<5}{out['engine']}")

    with open(os.path.join(HERE, "examples", "combined_results.json"), "w") as f:
        json.dump(combined, f, indent=2, default=str)

    correct = sum(1 for r in combined if r["output"]["category"] == r["output"].get("expected_category"))
    print("-" * 90)
    print(f"Category match vs hand-labeled expectation: {correct}/{len(combined)}")


if __name__ == "__main__":
    main()
