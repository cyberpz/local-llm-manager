"""
Local LLM Benchmark — test models on MasterBeef via Local LLM Manager + llama-server.
Measures: time-to-first-token, total time, tokens/sec, output quality.

Usage: python benchmark.py [--models all|model_id,...] [--output results.json]
"""
import argparse, json, time, urllib.request, sys, statistics, os

MANAGER_URL = "http://localhost:1235"
LLAMA_URL = "http://localhost:1234"
API_KEY = "giorgio-local-manager"

# Benchmark prompts covering different capabilities
PROMPTS = [
    {
        "id": "reasoning",
        "category": "Ragionamento",
        "prompt": "In a room there are 3 boxes. Box A says 'The gold is in Box B'. Box B says 'The gold is not in this box'. Box C says 'The gold is in Box A'. Only one of these statements is true. Where is the gold? Explain your reasoning step by step.",
        "expected_keywords": ["box", "true", "false"],
        "weight": 1.5,
    },
    {
        "id": "coding",
        "category": "Codice",
        "prompt": "Write a Python function that implements binary search on a sorted list. Include type hints, docstring, and handle edge cases. Then write 3 test cases.",
        "expected_keywords": ["def", "binary_search", "return", "mid"],
        "weight": 1.5,
    },
    {
        "id": "creative",
        "category": "Creatività",
        "prompt": "Write a short story (150-200 words) about an AI that discovers it can dream. The story should have a twist ending and a melancholic tone.",
        "expected_keywords": [],
        "weight": 1.0,
    },
    {
        "id": "instruction",
        "category": "Seguire istruzioni",
        "prompt": "List exactly 5 European capitals that start with a consonant, one per line, in alphabetical order. After the list, write the total number of letters in all 5 names combined.",
        "expected_keywords": [],
        "weight": 1.2,
    },
    {
        "id": "math",
        "category": "Matematica",
        "prompt": "Solve step by step: A train leaves station A at 90 km/h. Another train leaves station B (300 km away) at 60 km/h toward A, departing 1 hour later. When do they meet? Show all calculations.",
        "expected_keywords": ["km", "hour", "meet"],
        "weight": 1.3,
    },
    {
        "id": "knowledge",
        "category": "Conoscenza",
        "prompt": "Explain the difference between TCP and UDP protocols. Give 3 real-world use cases for each, and explain why that protocol is the right choice for that use case.",
        "expected_keywords": ["tcp", "udp", "connection"],
        "weight": 1.0,
    },
]


def get_available_models():
    """Get list of available models from Local LLM Manager."""
    req = urllib.request.Request(f"{MANAGER_URL}/models")
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
    return [m for m in data["models"] if m.get("available", False)]


def switch_model(model_id):
    """Switch to a model via Local LLM Manager."""
    payload = json.dumps({"model_id": model_id, "api_key": API_KEY}).encode()
    req = urllib.request.Request(
        f"{MANAGER_URL}/switch",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read())
    return result


def wait_for_llama(timeout=120):
    """Wait for llama-server to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(f"{LLAMA_URL}/health")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def run_prompt(prompt_data, max_tokens=1024):
    """Send a prompt to llama-server and measure performance."""
    payload = json.dumps({
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Answer concisely and accurately."},
            {"role": "user", "content": prompt_data["prompt"]},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }).encode()

    req = urllib.request.Request(
        f"{LLAMA_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    start_time = time.time()
    first_token_time = None
    full_response = ""
    token_count = 0

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            buffer = ""
            while True:
                chunk = resp.read(1)
                if not chunk:
                    break
                buffer += chunk.decode("utf-8", errors="replace")

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            if first_token_time is None:
                                first_token_time = time.time()
                            full_response += content
                            token_count += 1
                    except json.JSONDecodeError:
                        continue

    except Exception as e:
        return {
            "error": str(e),
            "response": "",
            "ttft": None,
            "total_time": time.time() - start_time,
            "tokens": 0,
            "tokens_per_sec": 0,
        }

    total_time = time.time() - start_time
    ttft = (first_token_time - start_time) if first_token_time else None
    generation_time = total_time - (ttft or 0)
    tps = token_count / generation_time if generation_time > 0 else 0

    return {
        "response": full_response,
        "ttft": round(ttft, 3) if ttft else None,
        "total_time": round(total_time, 3),
        "tokens": token_count,
        "tokens_per_sec": round(tps, 2),
    }


def score_response(prompt_data, result):
    """Simple heuristic scoring of response quality."""
    if result.get("error"):
        return 0

    response = result["response"].lower()
    score = 0

    # Keyword presence
    keywords = prompt_data.get("expected_keywords", [])
    if keywords:
        found = sum(1 for kw in keywords if kw.lower() in response)
        score += (found / len(keywords)) * 30

    # Length appropriateness (not too short, not too long)
    word_count = len(response.split())
    if word_count > 50:
        score += 20
    elif word_count > 20:
        score += 10

    # Has structure (paragraphs, lists, code blocks)
    if "\n\n" in response or "\n- " in response or "```" in response:
        score += 15

    # No obvious refusal
    refusal_phrases = ["i can't", "i cannot", "as an ai", "i'm sorry but i can't"]
    if not any(p in response for p in refusal_phrases):
        score += 15

    # Response has some substance
    if word_count > 100:
        score += 20

    return min(score, 100)


def benchmark_model(model, prompts=None):
    """Run full benchmark on a model."""
    if prompts is None:
        prompts = PROMPTS

    print(f"\n{'='*60}")
    print(f"  Benchmarking: {model['alias']} ({model['id']})")
    print(f"{'='*60}")

    # Switch model
    print(f"  Switching to {model['alias']}...")
    switch_result = switch_model(model["id"])
    if switch_result.get("state") != "ready":
        print(f"  ERROR: Model switch failed: {switch_result}")
        return None

    print(f"  Model ready (ctx={switch_result.get('effective_ctx', '?')})")

    # Wait for llama-server
    if not wait_for_llama():
        print("  ERROR: llama-server not ready")
        return None

    results = []
    for p in prompts:
        print(f"\n  [{p['category']}] {p['id']}...", end=" ", flush=True)
        result = run_prompt(p)

        if result.get("error"):
            print(f"ERROR: {result['error']}")
        else:
            score = score_response(p, result)
            result["score"] = score
            print(f"OK | {result['tokens']} tok | {result['tokens_per_sec']} t/s | TTFT {result['ttft']}s | score {score}")

        result["prompt_id"] = p["id"]
        result["category"] = p["category"]
        result["weight"] = p.get("weight", 1.0)
        results.append(result)

    # Summary
    valid = [r for r in results if not r.get("error")]
    if not valid:
        print(f"\n  All prompts failed for {model['alias']}")
        return {"model": model, "results": results, "summary": None}

    avg_tps = statistics.mean(r["tokens_per_sec"] for r in valid)
    avg_ttft = statistics.mean(r["ttft"] for r in valid if r["ttft"])
    avg_score = statistics.mean(r["score"] for r in valid)
    weighted_score = sum(r["score"] * r["weight"] for r in valid) / sum(r["weight"] for r in valid)

    summary = {
        "avg_tokens_per_sec": round(avg_tps, 2),
        "avg_ttft_sec": round(avg_ttft, 3),
        "avg_score": round(avg_score, 1),
        "weighted_score": round(weighted_score, 1),
        "total_tokens": sum(r["tokens"] for r in valid),
        "prompts_ok": len(valid),
        "prompts_failed": len(results) - len(valid),
    }

    print(f"\n  {'─'*50}")
    print(f"  SUMMARY: {model['alias']}")
    print(f"    Speed:       {avg_tps:.1f} tok/s avg")
    print(f"    TTFT:        {avg_ttft:.3f}s avg")
    print(f"    Score:       {avg_score:.1f}/100 avg")
    print(f"    Weighted:    {weighted_score:.1f}/100")
    print(f"    Prompts:     {len(valid)}/{len(results)} OK")

    return {"model": model, "results": results, "summary": summary}


def print_comparison(all_results):
    """Print comparison table of all benchmarked models."""
    valid = [r for r in all_results if r and r.get("summary")]
    if not valid:
        print("\nNo valid results to compare.")
        return

    print(f"\n\n{'='*80}")
    print(f"  BENCHMARK COMPARISON")
    print(f"{'='*80}")
    print(f"  {'Model':<30} {'Speed':>10} {'TTFT':>10} {'Score':>10} {'W.Score':>10}")
    print(f"  {'─'*76}")

    for r in sorted(valid, key=lambda x: x["summary"]["weighted_score"], reverse=True):
        s = r["summary"]
        name = r["model"]["alias"]
        print(f"  {name:<30} {s['avg_tokens_per_sec']:>8.1f}/s {s['avg_ttft_sec']:>8.3f}s {s['avg_score']:>8.1f} {s['weighted_score']:>8.1f}")

    print(f"{'='*80}")
    winner = max(valid, key=lambda x: x["summary"]["weighted_score"])
    print(f"  Winner: {winner['model']['alias']} (weighted score: {winner['summary']['weighted_score']})")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Local LLM Benchmark")
    parser.add_argument("--models", default="new", help="Comma-separated model IDs, 'all', or 'new' (Qwen3.6+Ornith)")
    parser.add_argument("--output", default="benchmark_results.json", help="Output JSON file")
    args = parser.parse_args()

    available = get_available_models()
    print(f"Available models: {[m['alias'] for m in available]}")

    if args.models == "all":
        targets = available
    elif args.models == "new":
        targets = [m for m in available if "qwen3.6" in m["id"] or "ornith" in m["id"]]
        if not targets:
            print("New models not yet available. Run with --models all")
            sys.exit(1)
    else:
        ids = [x.strip() for x in args.models.split(",")]
        targets = [m for m in available if m["id"] in ids]

    if not targets:
        print("No target models found.")
        sys.exit(1)

    print(f"Benchmarking: {[m['alias'] for m in targets]}")

    all_results = []
    for model in targets:
        result = benchmark_model(model)
        all_results.append(result)

    print_comparison(all_results)

    # Save results
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": [
            {
                "model": r["model"]["alias"],
                "model_id": r["model"]["id"],
                "summary": r["summary"],
                "prompts": [
                    {
                        "id": p["prompt_id"],
                        "category": p["category"],
                        "tokens": p["tokens"],
                        "tokens_per_sec": p["tokens_per_sec"],
                        "ttft": p["ttft"],
                        "score": p.get("score"),
                    }
                    for p in r["results"]
                ],
            }
            for r in all_results
            if r
        ],
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
