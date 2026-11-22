"""Measure the fixed Inkling 2 Mini workload on a running llama-server."""

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path


MODEL = "inkling-2-mini"
CONTEXT = 4096
QUESTION_ID = "M4"
PROMPT_TOKENS = 76
GENERATED_TOKENS = 768


def request_json(base_url, endpoint, payload=None, timeout=5):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base_url + endpoint, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def verify_model(base_url):
    models = request_json(base_url, "/v1/models")["data"]
    model = next((entry for entry in models if entry["id"] == MODEL), None)
    metadata = model.get("meta", {}) if model is not None else {}
    if metadata.get("n_ctx") != CONTEXT or not metadata.get("ftype", "").startswith("IQ4_NL"):
        raise ValueError(f"Expected model {MODEL} with IQ4_NL weights and context {CONTEXT}")


def measure(base_url, prompt):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": 42,
        "cache_prompt": False,
        "max_tokens": GENERATED_TOKENS,
    }
    started = time.perf_counter()
    response = request_json(base_url, "/v1/chat/completions", payload, timeout=180)
    wall_seconds = time.perf_counter() - started
    timings = response["timings"]
    expected = {"cache_n": 0, "prompt_n": PROMPT_TOKENS, "predicted_n": GENERATED_TOKENS}
    for field, value in expected.items():
        if timings[field] != value:
            raise ValueError(f"Unexpected {field}: expected {value}, got {timings[field]}")
    return {
        "prompt_tokens": timings["prompt_n"],
        "generated_tokens": timings["predicted_n"],
        "cached_tokens": timings["cache_n"],
        "decode_tokens_per_second": timings["predicted_per_second"],
        "wall_seconds": wall_seconds,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18188", help="llama-server base URL")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    base_url = args.url.rstrip("/")
    suite = json.loads((Path(__file__).parent / "quality" / "suite.json").read_bytes())
    prompt = next(question["prompt"] for question in suite if question["id"] == QUESTION_ID)
    try:
        verify_model(base_url)
        samples = [measure(base_url, prompt) for _ in range(args.runs)]
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Benchmark failed: {error}\n")

    print(json.dumps({
        "model": MODEL,
        "question_id": QUESTION_ID,
        "context": CONTEXT,
        "runs": samples,
        "mean_decode_tokens_per_second": statistics.mean(s["decode_tokens_per_second"] for s in samples),
        "mean_wall_seconds": statistics.mean(s["wall_seconds"] for s in samples),
    }, indent=2))


if __name__ == "__main__":
    main()
