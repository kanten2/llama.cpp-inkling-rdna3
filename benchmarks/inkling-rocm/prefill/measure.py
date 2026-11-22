"""Compare prompt processing with fusion off/on using one extracted Windows ZIP."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess

ROUNDS = (("off", 2), ("on", 3), ("on", 2), ("off", 3))


def report(folder):
    for length in (4096, 16384):
        times = {"off": [], "on": []}
        for i, (mode, count) in enumerate(ROUNDS, 1):
            raw = folder / f"pp{length}-{i}-{mode}.json"
            row, = json.loads(raw.read_bytes())
            if (row["n_prompt"], row["n_gen"], row["n_depth"], len(row["samples_ns"])) != (length, 0, 0, count):
                raise ValueError(f"Unexpected workload or repetition count in {raw}")
            times[mode].extend(t / 1e9 for t in row["samples_ns"])
        for mode, samples in times.items():
            rates = [length / t for t in samples]
            print(f"pp{length} {mode:3s}  {statistics.mean(samples):.3f} s  "
                  f"{statistics.mean(rates):.2f} +/- {statistics.stdev(rates):.2f} tok/s")


def measure(runtime, model, output):
    runtime = runtime.resolve(strict=True)
    model = model.resolve(strict=True)
    output.mkdir(parents=True, exist_ok=False)
    system = Path(os.environ["SystemRoot"])
    env = {k: v for k, v in os.environ.items() if not k.startswith(
        ("GGML_", "LLAMA_", "INKLING_", "HIP_", "ROCR_", "CUDA_", "ROCBLAS_", "HIPBLAS"))}
    env.update(PATH=os.pathsep.join(map(str, [runtime, system / "System32", system])),
               HIP_VISIBLE_DEVICES="0", GGML_HIP_IQ4_GATE4="1", GGML_HIP_DISABLE_SCALE_ADD_FUSION="1")
    for length in (4096, 16384):
        for i, (mode, count) in enumerate(ROUNDS, 1):
            name = f"pp{length}-{i}-{mode}"
            cmd = [str(runtime / "llama-bench.exe"), "-m", model.name, "-dev", "ROCm0", "-sm", "none",
                   "-mg", "0", "-ngl", "99", "-t", "16", "-b", "2048", "-ub", "512", "-fa", "auto",
                   "-ctk", "f16", "-ctv", "f16", "-d", "0", "-p", str(length), "-n", "0",
                   "-r", str(count), "-o", "json", "-v"]
            env["INKLING_PP_FUSE"] = "1" if mode == "on" else "0"
            print(name, flush=True)
            log = output / (name + ".log")
            with (output / (name + ".json")).open("wb") as out, log.open("wb") as err:
                subprocess.run(cmd, cwd=model.parent, env=env, stdout=out, stderr=err, check=True)
            trace = log.read_text("utf8", errors="replace")
            if ("PP_FUSION_ACTIVE" in trace) != (mode == "on") or "49/49 layers to GPU" not in trace or "RX 7900 XTX" not in trace:
                raise RuntimeError(f"Unexpected fusion or device selection in {log}")
    report(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    for flag in ("runtime", "model", "output"):
        run.add_argument("--" + flag, type=Path, required=True)
    sub.add_parser("report").add_argument("folder", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        measure(args.runtime, args.model, args.output)
    else:
        report(args.folder)
