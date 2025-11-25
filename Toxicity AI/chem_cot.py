# ChemCoT.py
import asyncio
import datasets
from utils import run_concurrent_worker, save_jsonl, compute_em_score, summarize_scores
import openai
from dotenv import load_dotenv
import os

load_dotenv()

BASE_URL = os.getenv("BASE_URL")
print(BASE_URL)
client = openai.AsyncOpenAI(api_key="dummy", base_url=BASE_URL)

def build_messages(item):
    system = '''You are a chemical assistant. Given the SMILES structural formula of a molecule, help me add a specified functional group and output the improved SMILES sequence of the molecule. 
Your response must be directly parsable JSON format:
{
    "output": "Modified Molecule SMILES"
}'''
    prompt = item.get("prompt") or item.get("query", "")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

def main():
    ds = datasets.load_from_disk('./ChemCoTBench')
    outputs = asyncio.run(run_concurrent_worker(ds, build_messages, client, concurrency=16))

    results = []
    for i, item in enumerate(ds):
        pred = outputs[i]
        gold = json.loads(item["meta"]).get("reference")
        em = compute_em_score(pred, gold)
        results.append({
            "id": item.get("id", i),
            "prompt": item.get("prompt") or item.get("query"),
            "model_output": pred,
            "reference": gold,
            "score": em
        })

    save_jsonl(results, "./ChemCoT_results.jsonl")
    print("SUMMARY:", summarize_scores(results))

if __name__ == "__main__":
    main()

