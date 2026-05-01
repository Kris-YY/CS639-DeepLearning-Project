#!/usr/bin/env python3
"""Extract judge-ready JSON records from a WebArena/Plan-and-Act result directory."""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import html
import json
import re
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract compact JSON records for LLM judging from a result directory."
    )
    parser.add_argument("--result-dir", required=True, help="Result directory to extract from.")
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repo root used to resolve relative task config paths.",
    )
    parser.add_argument(
        "--condition",
        default="",
        help="Optional override for the condition label, e.g. fixed_plan or baseline.",
    )
    parser.add_argument(
        "--last-actions",
        type=int,
        default=5,
        help="Number of trailing actions to include per record.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output path. Defaults to <result-dir>/judge_records.json.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_task_config_path(repo_root: Path, config_base_dir: str | None, task_id: int) -> Path | None:
    if not config_base_dir:
        return None

    base = Path(config_base_dir)
    candidates: list[Path] = []
    if base.is_absolute():
        candidates.append(base)
    else:
        candidates.append(repo_root / base)
        candidates.append(repo_root / "VisualAgentBench" / "VAB-WebArena-Lite" / base)

    for candidate in candidates:
        task_path = candidate / f"{task_id}.json"
        if task_path.exists():
            return task_path
    return None


def infer_runner_type(run_config: dict[str, Any]) -> str:
    if run_config.get("cot_planner_model") and run_config.get("planner_ip"):
        return "plan_and_act_replanning"
    if run_config.get("cot_actor_model"):
        return "plan_and_act_static"
    if "model" in run_config:
        return "baseline"
    return "unknown"


def infer_condition(runner_type: str, override: str) -> str:
    if override:
        return override
    if runner_type == "plan_and_act_static":
        return "fixed_plan"
    if runner_type == "plan_and_act_replanning":
        return "replanning"
    if runner_type == "baseline":
        return "baseline"
    return "unknown"


def extract_action_text(entry: Any) -> str | None:
    if isinstance(entry, dict):
        action = entry.get("action_str")
        return action.strip() if isinstance(action, str) and action.strip() else None
    if isinstance(entry, str):
        match = re.search(r"\[Start of Action\]\s*(.*?)\s*\[End of Action\]", entry, re.S)
        if match:
            return match.group(1).strip()
        stripped = entry.strip()
        return stripped or None
    return None


def extract_notes_from_action(action_text: str) -> list[str]:
    notes: list[str] = []
    for line in action_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# Note:"):
            notes.append(stripped)
    return notes


def extract_exit_message(action_text: str | None) -> str | None:
    if not action_text:
        return None
    match = re.search(r"exit\(message=(\".*?\"|'.*?')\)", action_text, re.S)
    if not match:
        return None
    try:
        value = ast.literal_eval(match.group(1))
    except Exception:
        return match.group(1).strip("\"'")
    return str(value)


def parse_actions_file(path: Path, trailing_count: int) -> dict[str, Any]:
    data = load_json(path)
    raw_actions = data.get("actions", [])
    action_texts = [text for text in (extract_action_text(entry) for entry in raw_actions) if text]
    notes: list[str] = []
    for action_text in action_texts:
        notes.extend(extract_notes_from_action(action_text))

    final_action = action_texts[-1] if action_texts else None
    final_answer = extract_exit_message(final_action)

    status_hint = "incomplete"
    if final_answer is not None:
        if final_answer.startswith("Early stop:"):
            status_hint = "early_stop"
        elif final_answer.strip():
            status_hint = "answered"
        else:
            status_hint = "stopped_no_answer"

    return {
        "score": data.get("score"),
        "step_count": len(action_texts),
        "last_actions": action_texts[-trailing_count:],
        "notes": notes,
        "final_action": final_action,
        "final_answer": final_answer,
        "status_hint": status_hint,
    }


def parse_error_file(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}

    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"\[Config file\]:\s+(?P<config>.+?/(?P<task_id>\d+)\.json)\n(?P<body>.*?)(?=(?:\n\[Config file\]:)|\Z)",
        re.S,
    )

    errors: dict[int, str] = {}
    for match in pattern.finditer(text):
        task_id = int(match.group("task_id"))
        body = match.group("body").strip()
        errors[task_id] = body or None
    return errors


def parse_render_file(path: Path) -> dict[str, str | None]:
    if not path.exists():
        return {"final_url": None, "final_page_text": None}

    text = path.read_text(encoding="utf-8")
    url_matches = re.findall(
        r"<h3 class='url'><a href=([^>]+)>URL:\s*(.*?)</a></h3>",
        text,
        re.S,
    )
    state_matches = re.findall(
        r"<div class='state_obv'><pre>(.*?)</pre><div>",
        text,
        re.S,
    )

    final_url = None
    if url_matches:
        final_url = html.unescape(url_matches[-1][1].strip())

    final_page_text = None
    if state_matches:
        final_page_text = html.unescape(state_matches[-1].strip())

    return {"final_url": final_url, "final_page_text": final_page_text}


def discover_task_ids(result_dir: Path, error_map: dict[int, str]) -> list[int]:
    task_ids: set[int] = set(error_map.keys())

    actions_dir = result_dir / "actions"
    if actions_dir.exists():
        for path in actions_dir.glob("*.json"):
            if path.stem.isdigit():
                task_ids.add(int(path.stem))

    for path in result_dir.glob("render_*.html"):
        match = re.fullmatch(r"render_(\d+)", path.stem)
        if match:
            task_ids.add(int(match.group(1)))

    return sorted(task_ids)


def model_name_from_config(run_config: dict[str, Any]) -> str | None:
    return (
        run_config.get("cot_actor_model")
        or run_config.get("model")
        or run_config.get("cot_planner_model")
    )


def model_endpoint_from_config(run_config: dict[str, Any]) -> str | None:
    return run_config.get("actor_ip") or run_config.get("planner_ip")


def build_record(
    task_id: int,
    repo_root: Path,
    result_dir: Path,
    run_config: dict[str, Any],
    runner_type: str,
    condition: str,
    trailing_count: int,
    error_map: dict[int, str],
) -> dict[str, Any]:
    task_config_path = resolve_task_config_path(
        repo_root,
        run_config.get("test_config_base_dir"),
        task_id,
    )

    intent = None
    if task_config_path:
        try:
            intent = load_json(task_config_path).get("intent")
        except Exception:
            intent = None

    actions_path = result_dir / "actions" / f"{task_id}.json"
    render_path = result_dir / f"render_{task_id}.html"
    error_text = error_map.get(task_id)

    actions_info = {
        "score": None,
        "step_count": 0,
        "last_actions": [],
        "notes": [],
        "final_action": None,
        "final_answer": None,
        "status_hint": "no_record",
    }
    if actions_path.exists():
        actions_info = parse_actions_file(actions_path, trailing_count)

    render_info = parse_render_file(render_path)

    final_status = actions_info["status_hint"]
    if error_text:
        final_status = "crash"
    elif final_status == "incomplete" and actions_info["step_count"] == 0 and render_path.exists():
        final_status = "no_record"

    return {
        "task_id": task_id,
        "intent": intent,
        "condition": condition,
        "runner_type": runner_type,
        "final_status": final_status,
        "final_answer": actions_info["final_answer"],
        "final_action": actions_info["final_action"],
        "final_url": render_info["final_url"],
        "final_page_text": render_info["final_page_text"],
        "step_count": actions_info["step_count"],
        "last_actions": actions_info["last_actions"],
        "notes": actions_info["notes"],
        "error": error_text,
        "score": actions_info["score"],
        "model_name": model_name_from_config(run_config),
        "model_endpoint": model_endpoint_from_config(run_config),
        "action_set_tag": run_config.get("action_set_tag"),
        "observation_type": run_config.get("observation_type"),
        "render_path": str(render_path.resolve()) if render_path.exists() else None,
        "actions_path": str(actions_path.resolve()) if actions_path.exists() else None,
        "task_config_path": str(task_config_path.resolve()) if task_config_path else None,
        "raw_reference": {
            "result_dir_config_path": str((result_dir / "config.json").resolve())
            if (result_dir / "config.json").exists()
            else None,
            "error_file_path": str((result_dir / "error.txt").resolve())
            if (result_dir / "error.txt").exists()
            else None,
        },
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    result_dir = Path(args.result_dir).resolve()
    output_path = (
        Path(args.output).resolve()
        if args.output
        else result_dir / "judge_records.json"
    )

    run_config_path = result_dir / "config.json"
    if not run_config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {result_dir}")
    run_config = load_json(run_config_path)

    runner_type = infer_runner_type(run_config)
    condition = infer_condition(runner_type, args.condition)
    error_map = parse_error_file(result_dir / "error.txt")
    task_ids = discover_task_ids(result_dir, error_map)

    records = [
        build_record(
            task_id=task_id,
            repo_root=repo_root,
            result_dir=result_dir,
            run_config=run_config,
            runner_type=runner_type,
            condition=condition,
            trailing_count=args.last_actions,
            error_map=error_map,
        )
        for task_id in task_ids
    ]

    bundle = {
        "schema_version": SCHEMA_VERSION,
        "source_result_dir": str(result_dir),
        "repo_root": str(repo_root),
        "runner_type": runner_type,
        "condition": condition,
        "extracted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "records": records,
    }

    output_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
