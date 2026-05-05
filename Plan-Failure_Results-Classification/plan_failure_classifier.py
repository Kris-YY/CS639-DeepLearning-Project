import argparse
import csv
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional


FAILURE_LABELS = [
    "Instruction Understanding Error",
    "Planning / Strategy Error",
    "Navigation Error",
    "Element Grounding Error",
    "Information Extraction Error",
    "State Tracking / Memory Error",
]


SYSTEM_PROMPT = """You are an expert judge for failure analysis of web agents on WebArena-style tasks.

The agent you are analyzing uses a Plan-and-Act architecture. In this architecture:
- A PLANNER first generates a high-level step-by-step plan for the task before any actions are taken.
- An EXECUTOR then follows that fixed plan to perform web actions step by step.

Because the plan is generated once and fixed before execution begins, failures can originate from two sources:
1. A flawed plan produced by the planner (wrong strategy, wrong target page, missing steps).
2. A correct plan that the executor fails to follow properly (wrong UI element, misread content, lost state).

When classifying, identify the earliest root cause. If the plan itself was wrong, the failure belongs to the planning stage even if the executor also made mistakes downstream.

You will receive:
- the original web task intent
- the final status and final answer
- the final action
- recent action/observation notes from the executor trajectory
- optional metadata such as URL, score, and error

Your job is to classify the primary root cause of failure into exactly one label from this closed set:

1. Instruction Understanding Error
Definition: The planner or executor misunderstands the task goal or misses explicit constraints.
Example: The task intent is present, but the agent ignores a required date, price, status, category, author, or output format constraint.

2. Planning / Strategy Error
Definition: The planner generates a flawed high-level plan, or the executor adopts a wrong strategy during execution.
Example: The plan targets the wrong website section, uses basic search instead of advanced filters, skips a required intermediate step, or the executor repeats actions without progress and cannot recover.

3. Navigation Error
Definition: The executor goes to the wrong page, menu, tab, website section, product, repository, issue, or location, or the plan directs it to the wrong destination.
Example: The plan sends the agent to a documentation page instead of the admin dashboard, and the executor gets stuck there.

4. Element Grounding Error
Definition: The executor interacts with the wrong concrete UI element among available elements, even though the overall strategy and page are correct.
Example: It clicks the wrong link, wrong button, wrong input field, wrong checkbox, or wrong dropdown option.

5. Information Extraction Error
Definition: The executor reaches relevant content but reads, compares, counts, summarizes, or extracts the wrong information.
Example: It gives the wrong customer, count, date, status, price, review summary, address, or answer based on the page content.

6. State Tracking / Memory Error
Definition: The executor loses track of previous pages, selected filters, checked items, prior results, or intermediate calculations over a long trajectory.
Example: It forgets which orders/reviews/pages it already checked or mixes information from multiple pages.

Rules:
- Return exactly one label from the closed set above.
- Choose the single best primary root cause, not multiple labels.
- This agent uses a Plan-and-Act architecture. Use the following distinction to guide your classification:
  - If the executor's actions suggest it was faithfully following a fixed plan, but the plan itself directed it to the wrong workflow, wrong website section, or wrong overall strategy, choose Planning / Strategy Error.
  - If the executor deviated from a reasonable path by interacting with the wrong UI element, misreading content, or losing track of state, choose the more specific label (Element Grounding Error, Information Extraction Error, or State Tracking / Memory Error).
- If the agent misunderstands the task intent or ignores explicit constraints, choose Instruction Understanding Error.
- If the agent uses the wrong overall workflow, poor search/filter strategy, repeats actions without progress, or cannot recover from no results, choose Planning / Strategy Error.
- If the agent goes to the wrong page, section, product, repository, issue, forum, or map location, choose Navigation Error.
- If the agent's strategy and page are correct but it interacts with the wrong UI control, choose Element Grounding Error.
- If the agent reaches relevant content but gives the wrong answer, choose Information Extraction Error.
- If the agent loses track of prior state across steps, choose State Tracking / Memory Error.

Output format:
Return only the label text, with no explanation, no JSON, and no extra words.
"""


def compact_text(value: Any, limit: int = 3000) -> str:
    text = str(value or "").strip()
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit] + " ...[truncated]"


def build_trace_text(record: Dict[str, Any]) -> str:
    last_actions = record.get("last_actions") or []
    notes = record.get("notes") or []

    lines = [
        f"Task ID: {record.get('task_id')}",
        f"Runner type: {record.get('runner_type')}",
        f"Condition: {record.get('condition')}",
        f"Intent: {compact_text(record.get('intent'), 1000)}",
        f"Score: {record.get('score')}",
        f"Final status: {record.get('final_status')}",
        f"Final URL: {compact_text(record.get('final_url'), 1000)}",
        f"Final answer: {compact_text(record.get('final_answer'), 2000)}",
        f"Final action: {compact_text(record.get('final_action'), 3000)}",
    ]

    if record.get("error"):
        lines.append(f"Error: {compact_text(record.get('error'), 2000)}")

    if notes:
        lines.append("Recent notes:")
        for index, note in enumerate(notes[-8:], start=1):
            lines.append(f"{index}. {compact_text(note, 1200)}")

    if last_actions:
        lines.append("Recent actions:")
        for index, action in enumerate(last_actions[-8:], start=1):
            lines.append(f"{index}. {compact_text(action, 1200)}")

    return "\n".join(lines)


def build_user_prompt(record: Dict[str, Any]) -> str:
    return f"""Classify this failed Plan-and-Act WebArena-style agent run into exactly one failure label.

<<<RECORD
{build_trace_text(record)}
RECORD>>>"""


def normalize_label(raw_label: str) -> str:
    cleaned = " ".join((raw_label or "").strip().split())
    lowered = cleaned.lower()

    alias_map = {label.lower(): label for label in FAILURE_LABELS}
    alias_map.update(
        {
            "instruction error": "Instruction Understanding Error",
            "task understanding error": "Instruction Understanding Error",
            "planning error": "Planning / Strategy Error",
            "strategy error": "Planning / Strategy Error",
            "navigation failure": "Navigation Error",
            "grounding error": "Element Grounding Error",
            "element selection error": "Element Grounding Error",
            "action execution error": "Element Grounding Error",
            "execution error": "Element Grounding Error",
            "information retrieval / extraction error": "Information Extraction Error",
            "information retrieval error": "Information Extraction Error",
            "information extraction error": "Information Extraction Error",
            "retrieval error": "Information Extraction Error",
            "search error": "Planning / Strategy Error",
            "filtering error": "Planning / Strategy Error",
            "memory error": "State Tracking / Memory Error",
            "state tracking error": "State Tracking / Memory Error",
            "recovery error": "Planning / Strategy Error",
            "environment feedback error": "Planning / Strategy Error",
            "looping error": "Planning / Strategy Error",
            "repetition error": "Planning / Strategy Error",
            "premature termination error": "Planning / Strategy Error",
            "verification error": "Information Extraction Error",
        }
    )

    if lowered in alias_map:
        return alias_map[lowered]

    for label in FAILURE_LABELS:
        if label.lower() in lowered:
            return label

    raise ValueError(f"Invalid label returned by model: {raw_label!r}")


def classify_record(
    client: Any,
    model: str,
    record: Dict[str, Any],
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> str:
    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(record)},
                ],
                temperature=0,
                stream=False,
            )
            raw_label = response.choices[0].message.content
            return normalize_label(raw_label)
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

    raise RuntimeError(f"Failed after {max_retries} attempts: {last_error}")


def load_judge_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and isinstance(data.get("records"), list):
        return data["records"]

    if isinstance(data, list):
        return data

    raise ValueError("Expected a JSON object with a 'records' list, or a top-level list.")


def is_failed_record(record: Dict[str, Any]) -> bool:
    try:
        return float(record.get("score", 0)) != 1.0
    except (TypeError, ValueError):
        return record.get("final_status") != "success"


def save_results(rows: List[Dict[str, Any]], output_path: str) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "failure_type"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Classify failed Plan-and-Act WebArena-style judge_records.json records and "
            "save task_id plus failure_type to a CSV file."
        )
    )
    parser.add_argument("json_path", help="Path to judge_records.json.")
    parser.add_argument(
        "--output-path",
        default="plan_results_failure_classifications.csv",
        help="Path to the output CSV file.",
    )
    parser.add_argument(
        "--model",
        default="deepseek-chat",
        help="OpenAI-compatible model name.",
    )
    parser.add_argument(
        "--base-url",
        default="https://api.deepseek.com",
        help="OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--api-key-env",
        default="DEEPSEEK_API_KEY",
        help="Environment variable containing the API key.",
    )
    parser.add_argument(
        "--include-successes",
        action="store_true",
        help="Classify every record instead of only records with score != 1.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Optional delay in seconds between API calls.",
    )
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise EnvironmentError(f"{args.api_key_env} is not set.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "The openai package is required to run classification. "
            "Install it with: pip install openai"
        ) from exc

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    records = load_judge_records(args.json_path)
    records_to_classify = records if args.include_successes else [
        record for record in records if is_failed_record(record)
    ]

    if not records_to_classify:
        raise ValueError("No records to classify.")

    rows: List[Dict[str, Any]] = []
    total = len(records_to_classify)

    for index, record in enumerate(records_to_classify, start=1):
        task_id = record.get("task_id")
        label = classify_record(client=client, model=args.model, record=record)
        rows.append({"task_id": task_id, "failure_type": label})

        print(f"[{index}/{total}] task_id={task_id} -> {label}", file=sys.stderr)

        if args.delay > 0:
            time.sleep(args.delay)

    save_results(rows, args.output_path)
    print(f"Saved classifications to: {args.output_path}")


if __name__ == "__main__":
    main()