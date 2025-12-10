# utils.py
import os
import json
import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# ==========================
# 파일 저장
# ==========================
def save_jsonl(data_list, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved results to {out_path}")

# ==========================
# EM 점수 계산
# ==========================
def compute_em_score(pred, reference):
    return 1 if pred == reference else 0

def compute_em_score_mmlu(pred, reference):
    return 1 if pred in reference else 0

# ==========================
# Summary
# ==========================
def summarize_scores(results):
    total = len(results)
    em_total = sum(r.get("score", 0) for r in results)
    return {
        "n_samples": total,
        "em_score": em_total / total if total > 0 else None
    }

# ==========================
# 비동기 모델 호출
# ==========================
async def call_model_async(messages, client, retries=3, initial_delay=1.0):
    delay = initial_delay
    for attempt in range(retries):
        try:
            resp = await client.chat.completions.create(
                model="25TOXMC_Blowfish_v1.0.9-AWQ",
                messages=messages,
                temperature=0.0,
                top_p=0.95,
                stream=False
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt == retries-1:
                raise
            await asyncio.sleep(delay)
            delay *= 2

# ==========================
# 공통 비동기 워커
# ==========================
async def run_concurrent_worker(data, build_messages_func, client, concurrency=16):
    sem = asyncio.Semaphore(concurrency)
    results = [None] * len(data)

    async def worker(i):
        async with sem:
            messages = build_messages_func(data[i])
            out = await call_model_async(messages, client)
            # <think> 제거 + JSON 파싱
            try:
                out_clean = re.sub(r".*?</think>", "", out, flags=re.DOTALL).strip()
                out_json = json.loads(out_clean)
                results[i] = out_json.get("output")
            except:
                results[i] = out_clean

    tasks = [asyncio.create_task(worker(i)) for i in range(len(data))]
    for f in tqdm(asyncio.as_completed(tasks), total=len(data), desc="추론 진행중"):
        await f

    return results

