"""Check five frozen text questions on each corrected Windows runtime before benchmarking."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "quality"))
from run import chat, server
from score import decode_answer, read_suite

CASES = ("M1", "M3", "C2", "D4", "L1")


def validate(args):
    protocol, questions = read_suite()
    args.output.mkdir(parents=True, exist_ok=False)
    args.model = args.model.resolve(strict=True)
    args.mmproj = None
    args.sdk = args.sdk.resolve(strict=True)
    for key in ("ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GGML_BACKEND_PATH"):
        os.environ.pop(key, None)
    os.environ["HIP_VISIBLE_DEVICES"] = "0"
    rows = []
    for mode in ("fix-only", "eight", "four"):
        args.runtime = (args.control if mode == "fix-only" else args.optimized).resolve(strict=True)
        with server(args, protocol, "text", mode, 4096, observe_dispatch=True) as url:
            for qid in CASES:
                q = questions[qid]
                result = chat(url, q["prompt"], 3840, protocol)
                answer = result["answer"]
                correct = result["finish"] == "stop" and decode_answer(answer) == q["expected"]
                rows.append({"mode": mode, "id": qid, "expected": q["expected"], "answer": answer,
                             "correct": correct, "finish": result["finish"], "usage": result["usage"],
                             "wall_s": result["wall_s"]})
                print(mode, qid, correct, result["finish"], flush=True)
        trace = (args.output / f"text-{mode}-4096.server.log").read_text(encoding="utf8", errors="replace")
        if "offloaded 49/49 layers to GPU" not in trace:
            raise ValueError("Layer placement differs from the expected single-device setup")
        (args.output / "answers.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf8")
    if not all(row["correct"] for row in rows):
        raise RuntimeError("Review the saved answers before benchmarking")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("control", "optimized", "sdk", "model", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--port", type=int, default=18203)
    validate(parser.parse_args())
