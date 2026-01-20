import time
import requests
import json
import logging
import psutil
from typing import List, Dict

# Usage:
# 1. Ensure Ollama is running (ollama serve)
# 2. Pull models first: ollama pull llama3.2:3b, ollama pull qwen2.5:1.5b
# 3. Run: python benchmark_models.py

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_PS_URL = "http://localhost:11434/api/ps"

MODELS_TO_TEST = [
    "phi4-mini:latest",  # Baseline
    "llama3.2:3b",       # Candidate for Planner/QA
    "qwen2.5:1.5b",      # Candidate for Step Definer
    # "biomistral:7b"    # Optional, likely too big for 4GB VRAM with context
]

TEST_PROMPT = """
Context:
Hirschsprung's disease (HSCR) is a birth defect. This disorder is characterized by the absence of particular nerve cells (ganglions) in a segment of the bowel in an infant. The absence of ganglion cells causes the muscles in the bowels to lose their ability to move stool through the intestine (peristalsis).

Question:
Explain the pathophysiology of Hirschsprung's disease based on the context provided.
"""

def check_model_loaded(model_name: str) -> bool:
    try:
        response = requests.get("http://localhost:11434/api/tags")
        models = [m['name'] for m in response.json()['models']]
        if model_name not in models:
            logger.warning(f"Model {model_name} not found locally. Please run 'ollama pull {model_name}'")
            return False
        return True
    except Exception as e:
        logger.error(f"Failed to check models: {e}")
        return False

def benchmark_model(model_name: str) -> Dict:
    logger.info(f"BENCHMARKING: {model_name}")
    
    if not check_model_loaded(model_name):
        return {"status": "not_found"}

    # 1. Warmup / Load Model
    logger.info(f"Loading {model_name}...")
    start_load = time.time()
    try:
        requests.post(OLLAMA_URL, json={"model": model_name, "prompt": "hi", "stream": False})
    except Exception as e:
        logger.error(f"Error loading {model_name}: {e}")
        return {"status": "error", "error": str(e)}
    load_time = time.time() - start_load
    
    # 2. Run Inference
    logger.info(f"Running inference test...")
    start_infer = time.time()
    response = requests.post(OLLAMA_URL, json={
        "model": model_name,
        "prompt": TEST_PROMPT,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_ctx": 2048
        }
    })
    end_infer = time.time()
    
    result = response.json()
    total_duration = result.get('total_duration', 0) / 1e9  # seconds
    eval_count = result.get('eval_count', 0)
    eval_duration = result.get('eval_duration', 0) / 1e9
    
    tps = eval_count / eval_duration if eval_duration > 0 else 0
    
    logger.info(f"Result: {tps:.2f} tokens/sec")
    
    return {
        "model": model_name,
        "load_time_s": load_time,
        "inference_time_s": total_duration,
        "tokens_generated": eval_count,
        "tokens_per_sec": tps,
        "response_sample": result.get("response", "")[:100] + "..."
    }

def main():
    results = []
    print("--------------------------------------------------")
    print(" STARTING BENCHMARK (Target: 4GB VRAM Optimization)")
    print("--------------------------------------------------")
    
    for model in MODELS_TO_TEST:
        res = benchmark_model(model)
        results.append(res)
        print(f"Finished {model}\n")
        # Cool down
        time.sleep(2)

    print("\n--------------------------------------------------")
    print(" FINAL RESULTS")
    print("--------------------------------------------------")
    print(f"{'Model':<20} | {'TPS':<10} | {'Load (s)':<10} | {'Infer (s)':<10}")
    print("-" * 60)
    for r in results:
        if r.get("status") == "not_found":
            print(f"{r['model']:<20} | NOT FOUND  | -          | -")
        elif r.get("status") == "error":
            print(f"{r['model']:<20} | ERROR      | -          | -")
        else:
            print(f"{r['model']:<20} | {r['tokens_per_sec']:.2f}       | {r['load_time_s']:.2f}       | {r['inference_time_s']:.2f}")

if __name__ == "__main__":
    main()
