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

You will receive:
- the original web task intent
- the final status and final answer
- the final action
- recent action/observation notes from the agent trajectory
- optional metadata such as URL, score, and error

Your job is to classify the primary root cause of failure into exactly one label from this closed set:

1. Instruction Understanding Error
Definition: The agent misunderstands the task goal or misses explicit constraints.
Example: The task is present, but the agent says no task was provided; or it ignores a required date, price, status, category, author, or output format.

2. Planning / Strategy Error
Definition: The agent understands the task but chooses the wrong high-level workflow or strategy.
Example: It uses basic search instead of advanced filters, searches in the wrong domain, or skips a required intermediate step.

3. Navigation Error
Definition: The agent goes to the wrong page, menu, tab, website section, product, repository, issue, or location.
Example: It opens a screen protector product while trying to find a Nintendo Switch card holder.

4. Element Grounding Error
Definition: The agent chooses the wrong concrete UI element among available elements.
Example: It clicks the wrong link, wrong button, wrong input field, wrong checkbox, or wrong dropdown option.

5. Information Extraction Error
Definition: The agent reaches relevant content but reads, compares, counts, summarizes, or extracts the wrong information.
Example: It gives the wrong customer, count, date, status, price, review summary, address, or answer based on the page.

6. State Tracking / Memory Error
Definition: The agent loses track of previous pages, selected filters, checked items, prior results, or intermediate calculations over a long trajectory.
Example: It forgets which orders/reviews/pages it already checked or mixes information from multiple pages.

Rules:
- Return exactly one label from the closed set above.
- Choose the single best primary root cause, not multiple labels.
- Prefer the earliest root cause that best explains the downstream failure.
- If the agent says no task was provided even though the intent is present, choose Instruction Understanding Error.
- If the agent uses the wrong overall workflow, poor search/filter strategy, repeats actions without progress, or cannot recover from no results, choose Planning / Strategy Error.
- If the agent goes to the wrong page, section, product, repository, issue, forum, or map location, choose Navigation Error.
- If the agent's strategy is reasonable but it interacts with the wrong UI control or cannot ground the intended UI element, choose Element Grounding Error.
- If the agent reaches relevant content but gives the wrong answer, choose Information Extraction Error.

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
    return f"""Classify this failed WebArena-style agent run into exactly one failure label.

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
            "Classify failed WebArena-style judge_records.json records and "
            "save task_id plus failure_type to a CSV file."
        )
    )
    parser.add_argument("json_path", help="Path to judge_records.json.")
    parser.add_argument(
        "--output-path",
        default="failure_classifications_judge_records.csv",
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
