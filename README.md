# DeepSeek Smoke Test Runbook

This repository combines:

- `plan-and-act/`
- `VisualAgentBench/VAB-WebArena-Lite/`

This README documents:

1. How to set up this combined workspace from a fresh clone
2. How to run the current WebArena-Lite smoke tests from this repo
3. How DeepSeek is wired into the existing OpenAI-compatible code paths
4. Where baseline and Plan-and-Act experiment outputs should be stored

For the upstream project docs, see:

- [`plan-and-act/README.md`](plan-and-act/README.md)
- [`VisualAgentBench/VAB-WebArena-Lite/README.md`](VisualAgentBench/VAB-WebArena-Lite/README.md)

## Workspace Layout

The expected local layout is:

```text
Nick-Plan/
├── baseline_eval/
├── plan-and-act/
├── plan_eval/
├── scripts/
└── VisualAgentBench/
    └── VAB-WebArena-Lite/
```

The WebArena-Lite runtime already contains the expected symlink:

- `VisualAgentBench/VAB-WebArena-Lite/plan_and_act -> ../../plan-and-act`

## Fresh Setup From A Clone

These steps assume you are starting from this repo root.

### 1. Clone the VisualWebArena runtime inside `VAB-WebArena-Lite`

```bash
cd VisualAgentBench/VAB-WebArena-Lite

git clone https://github.com/web-arena-x/visualwebarena.git visualwebarena
git -C visualwebarena reset --hard ad57aae4dad71531504726900b80db02e0526158
bash replace.sh
```

### 2. Create a Python environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install
pip install -e .
```

### 3. Install the extra packages this checkout relies on in practice

```bash
python -m pip install lxml dashscope anthropic
```

### 4. Export the WebArena website URLs

Set these to your actual hosted WebArena environment:

```bash
export MPLCONFIGDIR=/tmp/mpl
export DATASET=webarena
export SHOPPING="http://18.220.65.163:7770"
export SHOPPING_ADMIN="http://18.220.65.163:7780/admin"
export REDDIT="http://18.220.65.163:9999"
export GITLAB="http://18.220.65.163:8023"
export MAP="http://18.220.65.163:3000"
export WIKIPEDIA="http://18.220.65.163:8888"
export HOMEPAGE="http://18.220.65.163:4399"
```

### 5. Export DeepSeek credentials

This workspace uses two slightly different credential paths:

- `plan-and-act` uses `DEEPSEEK_API_KEY` when `--actor_ip` or `--planner_ip` points at a DeepSeek URL
- the WebArena-Lite baseline stack still authenticates through the OpenAI-compatible client path, so it reads `OPENAI_API_KEY`

Set both keys so the current code works end to end:

```bash
export DEEPSEEK_API_KEY=sk-65b2ca3bcb1d4da69341b8e0d948f380
export OPENAI_API_KEY="$DEEPSEEK_API_KEY"
```

Optional shell-wide default for the OpenAI-compatible base URL:

```bash
export OPENAI_API_URL="https://api.deepseek.com/beta"
```

For baseline runs, `run.py` now also sets `OPENAI_API_URL` from `--planner_ip`, so the baseline model and evaluator are explicitly routed to the same DeepSeek endpoint even if `OPENAI_API_URL` was not pre-exported in the shell.


### 6. Generate task configs and login cookies

```bash
python scripts/generate_test_data.py
bash prepare.sh
```

At that point, the runtime should be ready for WebArena-Lite smoke tests.

## Where To Run Commands

Run evaluation commands from the WebArena-Lite runtime root:

```bash
cd /path/to/Nick-Plan/VisualAgentBench/VAB-WebArena-Lite
source venv/bin/activate
```

Canonical experiment output roots:

- baseline runs: `baseline_eval/`
- Plan-and-Act runs: `plan_eval/`

If you are in `VisualAgentBench/VAB-WebArena-Lite`, define:

```bash
REPO_ROOT="$(cd ../.. && pwd)"
```

## First Recommended Smoke Test

This is the smallest useful replanning smoke test for this repo:

```bash
python plan_and_act/run_plan_and_act_with_replanning.py \
  --test_start_idx 0 \
  --test_end_idx 1 \
  --test_config_base_dir config_files/wa/test_webarena_lite \
  --cot_planner_model deepseek-chat \
  --cot_actor_model deepseek-chat \
  --planner_ip https://api.deepseek.com/beta \
  --actor_ip https://api.deepseek.com/beta \
  --result_dir "$REPO_ROOT/plan_eval/deepseek_lite_smoke_1" \
  --action_set_tag webrl_id \
  --observation_type webrl
```

If you use a different DeepSeek model ID on your account, replace `deepseek-chat` with that model name in both flags.

## Baseline Smoke Test

For the direct baseline runner, store outputs under `baseline_eval/`:

```bash
python run.py \
  --instruction_path agent/prompts/jsons/p_cot_id_actree_3s.json \
  --test_start_idx 0 \
  --test_end_idx 1 \
  --result_dir "$REPO_ROOT/baseline_eval/deepseek_baseline_smoke_1" \
  --test_config_base_dir config_files/wa/test_webarena_lite \
  --provider openai \
  --eval_llm_model deepseek-chat \
  --model deepseek-chat \
  --mode chat \
  --planner_ip https://api.deepseek.com/beta \
  --max_tokens 2048 \
  --viewport_width 1280 \
  --viewport_height 720 \
  --action_set_tag id_accessibility_tree \
  --observation_type accessibility_tree \
  --max_steps 25 \
  --current_viewport_only
```

For an alternate baseline that keeps the WebRL interface but uses an executor-style prompt without an explicit plan, run:

```bash
python run.py \
  --instruction_path agent/prompts/jsons/p_webrl_executor_style_no_plan_chat.json \
  --test_start_idx 0 \
  --test_end_idx 1 \
  --result_dir "$REPO_ROOT/baseline_eval/deepseek_webrl_executor_style_smoke_1" \
  --test_config_base_dir config_files/wa/test_webarena_lite \
  --provider openai \
  --eval_llm_model deepseek-chat \
  --model deepseek-chat \
  --mode chat \
  --planner_ip https://api.deepseek.com/beta \
  --max_obs_length 0 \
  --max_tokens 2048 \
  --viewport_width 1280 \
  --viewport_height 720 \
  --action_set_tag webrl_id \
  --observation_type webrl \
  --max_steps 25 \
  --current_viewport_only
```

## Static Smoke Test

The non-replanning runner in this repo can use a per-task precomputed plan directory.
Recommended convention:

- store frozen plans under `plan_eval/plans/`
- use one file per task id
- example: `plan_eval/plans/0.json`, `plan_eval/plans/1.json`

Each file should contain a single `Plan` object:

```json
{
  "reasoning": "...",
  "plan": "..."
}
```

```bash
python plan_and_act/run_plan_and_act.py \
  --test_start_idx 0 \
  --test_end_idx 1 \
  --test_config_base_dir config_files/wa/test_webarena_lite \
  --precomputed_cot_plans_path "$REPO_ROOT/plan_eval/plans" \
  --cot_actor_model deepseek-chat \
  --eval_llm_model deepseek-chat \
  --actor_ip https://api.deepseek.com/beta \
  --result_dir "$REPO_ROOT/plan_eval/deepseek_static_lite_smoke_1" \
  --action_set_tag webrl_id \
  --observation_type webrl \
  --viewport_width 1280 \
  --viewport_height 720 \
  --max_tokens 2048 \
  --max_steps 25 \
  --current_viewport_only
```

If a task plan file is missing, the runner will generate it online and save it as `<task_id>.json` in that directory.
For a strictly frozen-plan experiment, generate the plan files once and then reuse them unchanged across later runs.

## Where Results Go

Useful output locations after a run:

```bash
ls "$REPO_ROOT/plan_eval/deepseek_lite_smoke_1"
ls "$REPO_ROOT/baseline_eval/deepseek_baseline_smoke_1"
ls log_files
```

The most useful artifacts are:

- `plan_eval/<run_name>/actions/<task_id>.json`
- `plan_eval/<run_name>/render_<task_id>.html`
- `baseline_eval/<run_name>/actions/<task_id>.json`
- `baseline_eval/<run_name>/render_<task_id>.html`
- `<run_dir>/judge_records.json` after extraction
- `log_files/log_*.log`

To extract judge-ready records from either output root:

```bash
python3 "$REPO_ROOT/scripts/extract_judge_records.py" \
  --result-dir "$REPO_ROOT/plan_eval/deepseek_lite_smoke_1"

python3 "$REPO_ROOT/scripts/extract_judge_records.py" \
  --result-dir "$REPO_ROOT/baseline_eval/deepseek_baseline_smoke_1" \
  --condition baseline
```

## Interpreting `judge_records.json`

The extracted file is a compact summary for downstream judging and comparison.
Each bundle has:

- top-level run metadata such as `runner_type`, `condition`, `source_result_dir`, and `extracted_at`
- a `records` array with one entry per discovered task id

Each record keeps the minimum state needed for later analysis:

- `score`: benchmark result for that task
- `final_status`: extractor-derived status, not a separate benchmark label
- `final_answer`: parsed from `exit(message="...")` when present
- `final_action`: the last extracted action text
- `step_count`, `last_actions`, and `notes`: compact trajectory summary
- `error`: the corresponding block from `error.txt` if the task crashed
- `render_path`, `actions_path`, `task_config_path`: pointers back to the full artifacts

Status semantics in the current extractor:

- `answered`: the final action was an `exit(message=...)` with a non-empty answer
- `incomplete`: there was no final extracted answer
- `crash`: the task had an entry in `error.txt`
- `no_record`: only used when a render exists but no action record was extracted

Score semantics in these runs:

- `1.0`: pass
- `0.0`: evaluated fail
- `-0.1`: unfinished or crashed before a normal evaluation score was produced

### Current `test 0-99` Comparison

The two extracted files are:

- `baseline_eval/test 0-99/judge_records.json`
- `plan_eval/test 0-99/judge_records.json`

Both runs used:

- `model_name = deepseek-chat`
- `model_endpoint = https://api.deepseek.com/beta`
- `action_set_tag = webrl_id`
- `observation_type = webrl`

The important difference is runner type:

- baseline: `runner_type = baseline`, `condition = baseline`
- plan run: `runner_type = plan_and_act_static`, `condition = fixed_plan`

Summary:

| Run | Passed | Failed (`0.0`) | Unfinished / Crash (`-0.1`) | Mean Score | `answered` | `incomplete` | `crash` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline `test 0-99` | 30 | 69 | 1 | 0.299 | 63 | 36 | 1 |
| fixed-plan `test 0-99` | 36 | 62 | 2 | 0.358 | 42 | 56 | 2 |

How to read that table:

- The fixed-plan run is better on task success rate: `36/100` vs `30/100`.
- The baseline run is more likely to emit a final answer string at all: `64` tasks with a parsed `final_answer` vs `42` for fixed-plan. That baseline count includes one early-stop message, which is why it is slightly higher than the `answered` count in the table.
- That means the baseline is more willing to terminate with an answer, but many of those answers are wrong.
- The fixed-plan run is more conservative and wins more tasks, but it also leaves more tasks incomplete.

Overlap and differences:

- both runs passed `19` of the same tasks:
  `4, 8, 10, 14, 17, 21, 39, 40, 42, 50, 55, 61, 62, 72, 73, 84, 87, 96, 99`
- baseline-only passes: `16, 19, 20, 26, 33, 38, 44, 69, 75, 77, 98`
- fixed-plan-only passes: `2, 6, 9, 23, 32, 36, 37, 43, 46, 49, 56, 66, 68, 83, 89, 91, 94`
- both runs failed or did not finish `53` tasks

Crash interpretation:

- task `81` is a crash in both runs, with score `-0.1`
- task `88` is an additional fixed-plan-only crash, also scored `-0.1`
- these are infrastructure or evaluator failures, not meaningful task failures, so they should usually be tracked separately from ordinary `0.0` results

Practical takeaway:

- If the goal is raw benchmark performance, the fixed-plan run is the better result set.
- If the goal is to study answerable trajectories, the baseline file has more terminal answers to inspect, but those extra answers do not translate into better task accuracy.
- For qualitative debugging, start from `judge_records.json`, then open the linked `actions_path` and `render_path` for any task you care about.

## Recommended LLM-Judge Comparison

For this repo, the most defensible setup is:

1. use WebArena score as the primary metric
2. use an LLM judge as a secondary comparison layer
3. judge final answers pairwise, not full trajectories

Why this is the recommended default:

- WebArena score is the closest thing here to a task-grounded automatic metric
- full-trajectory judging is easy to bias toward verbose or cleaner-looking traces
- final-answer pairwise judging is simpler, cheaper, and easier to explain in a report

### Recommended Comparison Protocol

Run both conditions on the same task set with the same non-planning settings:

- same task ids
- same `action_set_tag`
- same `observation_type`
- same model backend if the goal is isolating planning
- same endpoint, temperature policy, and max step budget where possible

In this checkout, the clean comparison shape is:

- baseline condition: `baseline_eval/<run_name>/`
- fixed-plan condition: `plan_eval/<run_name>/`
- extraction step: `scripts/extract_judge_records.py`

Then compare them as follows:

1. Extract `judge_records.json` for both runs.
2. Match records by `task_id`.
3. Blind the condition names and randomize answer order as `A` vs `B`.
4. Give the judge the task prompt plus each system's `final_answer`.
5. Optionally include `final_url` and a compact final-state evidence snippet.
6. Ask for `A wins`, `B wins`, or `tie`, plus a short rationale.
7. Aggregate win/loss/tie counts over all tasks.

### What To Give The Judge

Primary input:

- `intent`
- `final_answer` from system A
- `final_answer` from system B

Optional support:

- `final_url`
- a trimmed excerpt of `final_page_text`

Do not use the full raw trajectory by default.
Only include trajectories for targeted error analysis after the main comparison is done.

### Recommended Rubric

Ask the judge to compare on:

- task completion
- factual correctness
- completeness
- constraint satisfaction
- unsupported claims or hallucination

For WebArena-style tasks, constraint satisfaction matters a lot. The judge should explicitly check whether the answer obeys things like:

- date filters
- counts
- ranking requirements
- distance or route constraints
- entity restrictions such as `international` airports only

### Failure Policy

Define the policy before judging and apply it symmetrically.

Recommended default:

- no `final_answer` counts as a loss against any non-empty correct answer
- crash or `score = -0.1` counts as failure unless both sides crashed
- if both runs fail to answer, allow the judge to return `tie`
- do not manually rescue one run but not the other

### Recommended Use Of `judge_records.json`

`judge_records.json` already contains the right fields for a first-pass judge bundle:

- `task_id`
- `intent`
- `final_answer`
- `final_status`
- `score`
- `error`
- `final_url`
- `final_page_text`

For a judge-ready pairwise record, the minimal useful projection is:

```json
{
  "task_id": 4,
  "intent": "List out reviewers, if exist, who mention about good fingerprint resistant",
  "system_a": {
    "final_answer": "Rachel",
    "final_status": "answered",
    "score": 0.0
  },
  "system_b": {
    "final_answer": "Rachel; T. Gannon",
    "final_status": "answered",
    "score": 1.0
  }
}
```

### Reporting Recommendation

When writing up results, report both:

- automatic outcome summary: pass rate, mean score, crash count
- LLM-judge outcome summary: `A wins`, `B wins`, `tie`

Recommended wording for conclusions:

- "Under the shared `webrl` / `webrl_id` setup, the fixed-plan planner/executor system outperformed the direct baseline on WebArena score and was preferred by the LLM judge on final-answer quality."

Avoid overstating the claim as:

- "planning is better than no planning"

because differences may still reflect prompt structure, decomposition format, or interaction behavior rather than planning alone.

## Notes On DeepSeek Integration

This repo is not a native DeepSeek integration. It is an OpenAI-client-based stack that already supports OpenAI-compatible endpoints.

The current behavior is:

- `plan-and-act/plan_and_act/cot/models.py` switches to `DEEPSEEK_API_KEY` when the configured base URL contains `deepseek`
- `VisualAgentBench/VAB-WebArena-Lite/llms/providers/openai_utils.py` uses the OpenAI Python SDK as an OpenAI-compatible transport layer and can target DeepSeek endpoints
- baseline runs route both agent calls and LLM-based evaluation through `--planner_ip`, which is mirrored into `OPENAI_API_URL` by `run.py`
- `plan-and-act/plan_and_act/previous_rounds_evaluators.py` reuses the same WebArena-Lite OpenAI-compatible helpers for evaluation

In practice, that means:

- use DeepSeek endpoints in `--planner_ip` and `--actor_ip`
- export `DEEPSEEK_API_KEY`
- also mirror that key into `OPENAI_API_KEY`
- optionally set `OPENAI_API_URL` to a DeepSeek-compatible base URL as a shell-wide default


## Failure types with corresponding descriptions

Understanding
