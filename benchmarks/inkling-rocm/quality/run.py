"""Run the frozen Japanese quality comparison with a b12606-based Windows ROCm build."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import urllib.request

from score import grade, read_suite


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def chat(url, prompt, cap, protocol):
    payload = {"model": "inkling-2-mini", "messages": [{"role": "user", "content": prompt}],
               "cache_prompt": protocol["cache_prompt"], "seed": protocol["seed"],
               "temperature": protocol["temperature"], "max_tokens": cap, "stream": True,
               "stream_options": {"include_usage": True}}
    request = urllib.request.Request(url + "/v1/chat/completions", json.dumps(payload).encode(),
                                     {"Content-Type": "application/json"})
    answer = []
    reasoning_chars = 0
    first_final = finish = timings = usage = None
    done = False
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=900) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            value = line[5:].strip()
            if value == b"[DONE]":
                done = True
                break
            data = json.loads(value)
            if "error" in data:
                raise RuntimeError(data["error"])
            timings = data.get("timings") or timings
            usage = data.get("usage") or usage
            for choice in data.get("choices", []):
                finish = choice.get("finish_reason") or finish
                delta = choice.get("delta", {})
                reasoning_chars += len(delta.get("reasoning_content") or "")
                piece = delta.get("content") or ""
                if piece:
                    answer.append(piece)
                    if first_final is None and piece.strip():
                        first_final = time.perf_counter() - start
    wall = time.perf_counter() - start
    if (not done or finish not in ("stop", "length") or not timings or not usage or
            timings["cache_n"] != 0 or usage["completion_tokens"] != timings["predicted_n"] or
            usage["prompt_tokens"] != timings["prompt_n"]):
        raise ValueError("Incomplete stream, cached prompt, or inconsistent token counts")
    return {"answer": "".join(answer), "reasoning_chars": reasoning_chars, "finish": finish,
            "first_final_s": first_final, "wall_s": wall, "timings": timings, "usage": usage}


@contextmanager
def server(args, protocol, block, mode, ctx, *, observe_dispatch=False):
    running = subprocess.check_output(["tasklist.exe", "/FI", "IMAGENAME eq llama-server.exe", "/FO", "CSV", "/NH"])
    if b'"llama-server.exe"' in running.lower():
        raise RuntimeError("A llama-server process is already running")
    with socket.socket() as check:
        check.bind(("127.0.0.1", args.port))
    url = f"http://127.0.0.1:{args.port}"
    cmd = [str(args.runtime / "llama-server.exe"), "--model", str(args.model),
           "--alias", "inkling-2-mini", "--host", "127.0.0.1", "--port", str(args.port),
           "--ctx-size", str(ctx), "--parallel", str(protocol["parallel"]), "--gpu-layers", "99",
           "--device", "ROCm0", "--jinja", "--verbosity", "4"]
    if args.mmproj is not None:
        cmd.extend(["--mmproj", str(args.mmproj)])
    env = os.environ.copy()
    for key in ("GGML_CUDA_DISABLE_GRAPHS", "GGML_CUDA_DISABLE_FUSION", "GGML_HIP_DISABLE_ROW_CLIP",
                "LLAMA_INKLING_DISABLE_MIX_REUSE", "GGML_HIP_DISABLE_CONV_RESIDUAL",
                "GGML_HIP_DISABLE_CONCAT_HISTORY", "GGML_IQ4_PROFILE_STRIPE", "GGML_IQ4_PROFILE_STRIDE",
                "GGML_HIP_IQ4_GATE4"):
        env.pop(key, None)
    env.update(GGML_HIP_DISABLE_SCALE_ADD_FUSION="1", GGML_CUDA_GRAPH_DEBUG="1")
    if mode == "four":
        env["GGML_HIP_IQ4_GATE4"] = "1"
    env["PATH"] = os.pathsep.join([str(args.sdk / "_rocm_sdk_libraries/bin"),
                                   str(args.sdk / "_rocm_sdk_core/bin"), env["PATH"]])
    log = args.output / f"{block}-{mode}-{ctx}.server.log"
    with log.open("wb") as out:
        proc = subprocess.Popen(cmd, cwd=args.runtime, stdout=out, stderr=out, env=env)
        try:
            for _ in range(130):
                if proc.poll() is not None:
                    raise RuntimeError(f"Server exited during startup. See {log}")
                try:
                    with urllib.request.urlopen(url + "/health", timeout=2) as response:
                        if json.load(response).get("status") == "ok":
                            break
                except OSError:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError(f"Server startup timed out. See {log}")
            chat(url, "整数の1だけを回答してください。", 32, protocol)
            yield url
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=40)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
    trace = log.read_text(encoding="utf8", errors="replace")
    four = "IQ4_NL gate/up: using 4 waves" in trace
    if ((four and mode != "four") or (mode == "four" and not observe_dispatch and not four)
            or not re.search(r"graphs reused\s*=\s*[1-9]", trace)
            or "ROCm0 (AMD Radeon RX 7900 XTX)" not in trace):
        raise RuntimeError(f"Unexpected ROCm device, wave selection or graph reuse. See {log}")


def run(args):
    protocol, questions = read_suite()
    for name in ("runtime", "sdk", "model", "mmproj"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    files = [args.model, args.mmproj, args.runtime / "llama-server.exe", args.runtime / "llama.dll",
             args.runtime / "ggml-hip.dll"]
    hashes = {f.name: sha256(f) for f in files}
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {"suite_sha256": protocol["suite_sha256"], "input_sha256": hashes}
    (args.output / "run-info.json").write_text(json.dumps(metadata, indent=2), encoding="utf8")
    path = args.output / "answers.jsonl"
    with path.open("x", encoding="utf8", newline="\n") as answers:
        for block, ids, modes in protocol["block_order"]:
            pending = list(ids)
            for ctx in protocol["ctx_tiers"]:
                truncated = set()
                for mode in modes:
                    print(f"Starting {block} {mode} at context {ctx}", flush=True)
                    with server(args, protocol, block, mode, ctx) as url:
                        for qid in pending:
                            r = chat(url, questions[qid]["prompt"], ctx - 256, protocol)
                            r.update(id=qid, mode=mode, ctx=ctx, block=block)
                            answers.write(json.dumps(r, ensure_ascii=False) + "\n")
                            answers.flush()
                            print(f'{qid} {mode} {r["finish"]} {r["wall_s"]:.2f}s', flush=True)
                            if r["finish"] == "length":
                                truncated.add(qid)
                pending = [qid for qid in pending if qid in truncated]
                if not pending:
                    break
            if pending:
                raise RuntimeError(f"Context exhausted before paired completion: {pending}")
    grade(path, args.output / "summary.json", args.output / "results.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("runtime", "sdk", "model", "mmproj", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--port", type=int, default=18203)
    run(parser.parse_args())
