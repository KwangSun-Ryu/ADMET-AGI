# MMLU_toxic.py
import json
import asyncio
from utils import run_concurrent_worker, save_jsonl, compute_em_score_mmlu, summarize_scores
import openai
import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("BASE_URL")

client = openai.AsyncOpenAI(api_key="dummy", base_url=BASE_URL)

def build_messages(item):
    system = item.get("system", "")
    prompt = item.get("prompt", "")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

def main():
    with open("./mmlu_toxic.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    outputs = asyncio.run(run_concurrent_worker(data, build_messages, client, concurrency=16))

    results = []
    for i, item in enumerate(data):
        pred = outputs[i]
        em = compute_em_score_mmlu(pred, item.get("answer", []))
        results.append({
            "id": item.get("id", i),
            "prompt": item.get("prompt"),
            "model_output": pred,
            "reference": item.get("answer"),
            "score": em
        })

    save_jsonl(results, "./MMLU_toxic_results.jsonl")
    print("SUMMARY:", summarize_scores(results))

if __name__ == "__main__":
    main()

