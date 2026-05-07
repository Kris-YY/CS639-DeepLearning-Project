import os
import json
import re
import time
import pandas as pd
from tqdm import tqdm
from openai import OpenAI


# =========================
# 1. LLM client
# =========================

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

MODEL_NAME = "deepseek-chat"


# =========================
# 2. Load data
# =========================

def load_records(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # judge_records.json 外层一般是 {"records": [...]}
    if isinstance(data, dict) and "records" in data:
        return data["records"]

    # 防止有些文件直接是 list
    if isinstance(data, list):
        return data

    raise ValueError("Unknown JSON format")


# =========================
# 3. Build prompt
# =========================

def build_prompt(record):
    task_id = record.get("task_id")
    intent = record.get("intent", "")
    status = record.get("final_status", "")
    score = record.get("score", "")
    final_url = record.get("final_url", "")
    final_answer = record.get("final_answer", "")
    final_action = record.get("final_action", "")

    last_actions = record.get("last_actions", [])
    notes = record.get("notes", [])

    action_text = ""
    for i, action in enumerate(last_actions, start=1):
        note = notes[i - 1] if i - 1 < len(notes) else ""
        action_text += f"\nStep {i}:\nNote: {note}\nAction: {action}\n"

    prompt = f"""
You are doing failure step localization for a failed web navigation agent trajectory.

The agent was trying to complete this task:

Task ID: {task_id}
Task intent: {intent}

The task failed.
Final status: {status}
Score: {score}
Final URL: {final_url}
Final answer: {final_answer}
Final action: {final_action}

Below are the last actions recorded for this trajectory.
Each step is one recent action. The true failure may have started in one of these steps.

{action_text}

Your job:
Identify the FIRST step among the listed steps where the agent clearly made a mistake.

Definition of mistake:
- The agent navigates to a wrong page or wrong website.
- The agent clicks an irrelevant element.
- The agent repeats useless actions instead of recovering.
- The agent extracts the wrong information.
- The agent gives up without enough evidence.
- If all listed steps are already after the real mistake, choose the earliest visible step where the trajectory is already wrong.
- If no clear mistake is visible, choose "unclear".

Output ONLY valid JSON in this format:

{{
  "failure_step": 1,
  "failure_type": "Navigation Error",
  "reason": "The agent clicked an irrelevant link and moved away from the target task.",
  "confidence": "high"
}}

Allowed failure_type values:
- Instruction Understanding Error
- Planning / Strategy Error
- Navigation Error
- Element Grounding Error
- Information Extraction Error
- State Tracking / Memory Error
- Step Budget / Incomplete Error
- Unclear
"""
    return prompt


# =========================
# 4. Call LLM
# =========================

def call_llm(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a careful evaluator of web navigation agent failures. Always output valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"LLM call failed: {e}")
            time.sleep(2)

    return None


# =========================
# 5. Parse LLM output
# =========================

def parse_json_output(text):
    if text is None:
        return {
            "failure_step": None,
            "failure_type": "Unclear",
            "reason": "LLM call failed.",
            "confidence": "low"
        }

    # 去掉 markdown code block
    text = text.strip()
    text = re.sub(r"^```json", "", text)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        return {
            "failure_step": None,
            "failure_type": "Unclear",
            "reason": f"Could not parse LLM output: {text[:200]}",
            "confidence": "low"
        }


# =========================
# 6. Analyze one file
# =========================

def analyze_file(input_path, output_csv, condition_name):
    records = load_records(input_path)

    results = []

    for record in tqdm(records, desc=f"Analyzing {condition_name}"):

        score = record.get("score")

        # 只分析失败 case
        if score == 1 or score == 1.0:
            continue

        prompt = build_prompt(record)
        llm_output = call_llm(prompt)
        parsed = parse_json_output(llm_output)

        results.append({
            "condition": condition_name,
            "task_id": record.get("task_id"),
            "intent": record.get("intent"),
            "score": record.get("score"),
            "final_status": record.get("final_status"),
            "step_count": record.get("step_count"),
            "failure_step": parsed.get("failure_step"),
            "failure_type": parsed.get("failure_type"),
            "reason": parsed.get("reason"),
            "confidence": parsed.get("confidence"),
            "final_url": record.get("final_url"),
        })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")
    print(df.head())

    return df


# =========================
# 7. Main
# =========================

if __name__ == "__main__":

    baseline_path = "judge_records.json"
    plan_path = "plan_eval_judge_records.json"

    baseline_df = analyze_file(
        input_path=baseline_path,
        output_csv="baseline_failure_steps.csv",
        condition_name="baseline"
    )

    plan_df = analyze_file(
        input_path=plan_path,
        output_csv="plan_failure_steps.csv",
        condition_name="plan_and_act"
    )

    all_df = pd.concat([baseline_df, plan_df], ignore_index=True)
    all_df.to_csv("all_failure_steps.csv", index=False)

    print("\nFailure step distribution:")
    print(pd.crosstab(all_df["failure_step"], all_df["condition"]))

    print("\nFailure type distribution:")
    print(pd.crosstab(all_df["failure_type"], all_df["condition"]))