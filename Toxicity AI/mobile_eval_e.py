# Mobile-Eval-E.py
import json
from datasets import load_dataset
from tqdm import tqdm
from openai import OpenAI
from utils import save_jsonl
import os
from dotenv import load_dotenv
load_dotenv()
# -------------------------
# GPT 클라이언트
# -------------------------

API_KEY = os.getenv("GPT_API_KEY")

client = OpenAI(api_key=API_KEY)

# -------------------------
# 시스템 프롬프트
# -------------------------
SYSTEM_PROMPT = """You are a mobile task planner that controls an Android phone via high-level actions.
Given a user instruction and a list of available apps, your goal is to output a step-by-step action sequence
to complete the task on the phone.

You MUST output a single JSON object with the following structure:

{
  "plan": ["high-level step 1", "high-level step 2", ...],
  "operations": [
    "action 1",
    "action 2",
    ...
  ]
}

Each action should be a short imperative phrase describing a concrete phone operation
(e.g., "open Maps", "tap on the search bar", "type 'korean restaurant'", "press enter").
Do not include any explanations or extra text outside the JSON object.
"""

JUDGE_SYSTEM_PROMPT = """
You are an expert evaluator for a mobile phone agent benchmark.

Your job is to evaluate how well a model-generated action sequence (operations)
solves a given mobile task, based on:

1) The natural language instruction.
2) The list of available apps and scenario.
3) A list of rubrics describing what a good solution should do.
4) A human reference action sequence (operations).
5) The model-generated action sequence.

You must output a single JSON object with the following fields:

{
  "rubric_score": float,        // between 0.0 and 1.0
  "action_match_score": float,  // between 0.0 and 1.0
  "overall_score": float,       // between 0.0 and 1.0
  "reason": "short explanation"
}

- rubric_score: how well the model operations satisfy the rubrics.
- action_match_score: how similar the model operations are to the human reference operations.
- overall_score: your overall judgement, not necessarily the average.
Return only the JSON object, with no additional text.
"""

# -------------------------
# JSON 파싱
# -------------------------
def extract_json(text: str):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"JSON block not found in model output: {text[:200]}...")
    json_str = text[start:end+1]
    return json.loads(json_str)

# -------------------------
# Judge Prompt 빌드
# -------------------------
def build_judge_prompt(example, model_ops):
    instruction = example["instruction"]
    apps = example.get("apps", [])
    scenario = example.get("scenario", "")
    rubrics = example.get("rubrics", [])
    human_ops = example.get("human_reference_operations", [])
    return f"""
[Instruction]
{instruction}

[Apps]
{apps}

[Scenario]
{scenario}

[Rubrics]
{json.dumps(rubrics, ensure_ascii=False, indent=2)}

[Human Reference Operations]
{json.dumps(human_ops, ensure_ascii=False, indent=2)}

[Model Operations to Evaluate]
{json.dumps(model_ops, ensure_ascii=False, indent=2)}
"""

# -------------------------
# GPT Judge 호출
# -------------------------
def judge_with_gpt(example, model_ops):
    user_prompt = build_judge_prompt(example, model_ops)
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    resp = client.chat.completions.create(
        model="gpt-5-mini",
        messages=messages,
    )
    text = resp.choices[0].message.content
    data = extract_json(text)
    return {
        "rubric_score": float(data.get("rubric_score", 0.0)),
        "action_match_score": float(data.get("action_match_score", 0.0)),
        "overall_score": float(data.get("overall_score", 0.0)),
        "reason": data.get("reason", ""),
    }

# -------------------------
# Actor 호출
# -------------------------
def process_request_vl(messages):
    import openai
    openai.api_key = "sk-None-1234"
    openai.base_url = "http://192.168.0.202:25321/v1/"
    output = openai.chat.completions.create(
        model='25TOXMC_Blowfish_v1.0.9-AWQ',
        messages=messages,
        temperature=0.0,
        top_p=0.95,
        stream=False
    )
    return output

def build_actor_prompt(example):
    instruction = example["instruction"]
    apps = example.get("apps", [])
    scenario = example.get("scenario", "")
    apps_str = ", ".join(apps) if apps else "no specific apps"
    return f"""User instruction:
{instruction}

You may use the following apps: {apps_str}
Scenario: {scenario}

Return ONLY a JSON object with the fields "plan" and "operations".
"""

def call_actor(example):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_actor_prompt(example)},
    ]
    resp = process_request_vl(messages)
    data = extract_json(resp.choices[0].message.content)
    ops = [str(o).strip() for o in data.get("operations", []) if str(o).strip()]
    return ops

# -------------------------
# 메인 루프
# -------------------------
def main():
    ds = load_dataset("mikewang/mobile_eval_e", split="test")
    scores = []

    for ex in tqdm(ds, desc="Evaluating with GPT judge"):
        try:
            model_ops = call_actor(ex)
        except Exception as e:
            print("Actor model failed:", e)
            model_ops = []

        try:
            judge_result = judge_with_gpt(ex, model_ops)
        except Exception as e:
            print("Judge model failed:", e)
            judge_result = {
                "rubric_score": 0.0,
                "action_match_score": 0.0,
                "overall_score": 0.0,
                "reason": f"Judge error: {e}",
            }

        scores.append(judge_result)

    avg_rubric = sum(s["rubric_score"] for s in scores) / len(scores)
    avg_action = sum(s["action_match_score"] for s in scores) / len(scores)
    avg_overall = sum(s["overall_score"] for s in scores) / len(scores)

    print("\n===== GPT Judge Overall Results =====")
    print(f"#examples          : {len(scores)}")
    print(f"Avg rubric_score   : {avg_rubric:.4f}")
    print(f"Avg action_match   : {avg_action:.4f}")
    print(f"Avg overall_score  : {avg_overall:.4f}")

    # JSONL 저장 (공통 구조)
    save_jsonl(scores, "./MobileEvalE_results.jsonl")

if __name__ == "__main__":
    main()

