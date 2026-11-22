"""Run the interleaved Windows llama-bench comparison; summarize an existing set with 'report'."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
from statistics import mean, stdev
import subprocess


ORDER = [
    ("official", "eight", "four"),
    ("eight", "four", "official"),
    ("four", "official", "eight"),
    ("official", "four", "eight"),
    ("four", "eight", "official"),
]
TESTS = {(512, 0): "pp512", (0, 128): "tg128", (0, 768): "tg768"}
ARGS = ["-dev", "ROCm0", "-sm", "none", "-mg", "0", "-ngl", "99", "-t", "16",
        "-b", "2048", "-ub", "512", "-fa", "auto", "-d", "0",
        "-p", "512", "-n", "128,768", "-r", "1", "-o", "json", "-v"]
ARTIFACTS = ["llama-bench.exe", "llama-bench-impl.dll", "llama.dll", "ggml.dll",
             "ggml-base.dll", "ggml-hip.dll", "amdhip64_7.dll"]
FLAGS_OFF = ["GGML_CUDA_DISABLE_GRAPHS", "GGML_CUDA_DISABLE_FUSION", "GGML_HIP_DISABLE_ROW_CLIP",
             "LLAMA_INKLING_DISABLE_MIX_REUSE", "GGML_HIP_DISABLE_CONV_RESIDUAL",
             "GGML_HIP_DISABLE_CONCAT_HISTORY", "GGML_IQ4_PROFILE_STRIDE", "GGML_IQ4_PROFILE_STRIPE",
             "GGML_HIP_IQ4_GATE4", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GGML_BACKEND_PATH"]
COMMON_ENV = {"HIP_VISIBLE_DEVICES": "0", "GGML_HIP_DISABLE_SCALE_ADD_FUSION": "1", "GGML_CUDA_GRAPH_DEBUG": "1"}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf8")


def planned_files(round_order):
    for rnd, order in enumerate(round_order, start=1):
        for pos, variant in enumerate(order, start=1):
            yield variant, f"r{rnd}-{pos}-{variant}"


def verify_runs(folder, manifest):
    checks = {}
    for variant, name in planned_files(manifest["round_order"]):
        raw_hash = sha(folder/"raw"/(name+".json"))
        if not (folder/"logs").is_dir():
            checks[name] = {"raw_sha256": raw_hash}
            continue
        trace = (folder/"logs"/(name+".log")).read_text(encoding="utf8", errors="replace")
        contexts = [int(n) for n in re.findall(r"llama_context: n_ctx\s+= (\d+)", trace)]
        reuse = [int(n) for n in re.findall(r"llama_perf_context_print:\s+graphs reused\s*=\s*(\d+)", trace)]
        four = "IQ4_NL gate/up: using 4 waves" in trace
        if (len(contexts)!=3 or len(reuse)!=3 or min(reuse)<1 or (four and variant!="four") or
                (variant=="four" and manifest.get("require_four_wave_dispatch", True) and not four) or
                "found 1 ROCm devices" not in trace or "offloaded 49/49 layers to GPU" not in trace):
            raise ValueError(f"{name}: backend verification failed")
        checks[name] = {"raw_sha256": raw_hash, "visible_rocm_devices": 1, "layers_offloaded": 49,
                        "four_wave_marker": four, "context_sizes_pp512_tg128_tg768": contexts,
                        "graph_reuses_pp512_tg128_tg768": reuse}
    check_file = folder/"checks.json"
    if (folder/"logs").is_dir():
        save(check_file, checks)
    elif check_file.exists():
        published = json.loads(check_file.read_bytes())
        if any(published[n]["raw_sha256"]!=data["raw_sha256"] for n,data in checks.items()):
            raise ValueError("A raw JSON digest differs from checks.json")


def summarize(folder):
    manifest = json.loads((folder/"manifest.json").read_bytes())
    variants = (manifest.get("control_variant", "official"), "eight", "four")
    rates = {v: {test: [] for test in TESTS.values()} for v in variants}
    config = {}
    for variant, name in planned_files(manifest["round_order"]):
        rows = json.loads((folder/"raw"/(name+".json")).read_bytes())
        shapes = [(r["n_prompt"], r["n_gen"]) for r in rows]
        if len(rows) != 3 or set(shapes) != set(TESTS):
            raise ValueError(f"{name}: unexpected tests {shapes}")
        for row, shape in zip(rows, shapes):
            test = TESTS[shape]
            if (row["devices"] != "ROCm0" or "7900 XTX" not in row["gpu_info"] or
                    row["n_depth"] != 0 or row["model_filename"] != manifest["model_filename"] or
                    len(row["samples_ts"]) != 1 or len(row["samples_ns"]) != 1):
                raise ValueError(f"{name}/{test}: unexpected device, file, depth, or samples")
            ns = row["samples_ns"][0]
            value = row["samples_ts"][0]
            if ns <= 0 or not math.isclose(value, sum(shape)*1e9/ns, rel_tol=1e-5):
                raise ValueError(f"{name}/{test}: inconsistent token rate")
            settings = {k: v for k, v in row.items()
                        if k not in {"test_time", "avg_ns", "avg_ts", "stddev_ns", "stddev_ts",
                                     "samples_ns", "samples_ts", "build_number", "build_commit"}}
            if settings != config.setdefault(test, settings):
                raise ValueError(f"{name}/{test}: benchmark settings differ")
            rates[variant][test].append(sum(shape)*1e9/ns)
    result = {}
    for test in TESTS.values():
        result[test] = {}
        for variant in variants:
            samples = rates[variant][test]
            result[test][variant] = {"n": len(samples), "mean_tok_s": mean(samples),
                                     "sample_sd_tok_s": stdev(samples), "samples_tok_s": samples}
    verify_runs(folder, manifest)
    save(folder/"summary.json", result)
    return result


def print_table(result):
    control = next(iter(result["pp512"]))
    label = "Official b12606" if control == "official" else "Fix only"
    print(f"| Test | {label} | Optimized 8 waves | 4 waves requested | 8 vs control | 4 vs control | 4 vs 8 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for test, modes in result.items():
        variants = [modes[v] for v in (control, "eight", "four")]
        cells = [f'{v["mean_tok_s"]:.2f} ± {v["sample_sd_tok_s"]:.2f}' for v in variants]
        official, eight, four = [v["mean_tok_s"] for v in variants]
        cells.extend(f"{100*(ratio-1):+.2f}%" for ratio in (eight/official, four/official, four/eight))
        print("| " + " | ".join([test, *cells]) + " |")


def run(a):
    folder = a.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    raw = folder/"raw"
    raw.mkdir(exist_ok=False)
    logs = folder/"logs"
    logs.mkdir(exist_ok=True)
    model = a.model.resolve(strict=True)
    control = a.control_variant
    order = [tuple(control if v == "official" else v for v in rnd) for rnd in ORDER]
    runtimes = {control: a.control.resolve(strict=True), "optimized": a.optimized.resolve(strict=True)}
    manifest = {"model_filename": model.name, "model_bytes": model.stat().st_size, "model_sha256": sha(model),
                "control_variant": control, "require_four_wave_dispatch": not a.observe_dispatch,
                "arguments_after_model": ARGS, "round_order": order, "common_env": COMMON_ENV,
                "clear_env": FLAGS_OFF, "four_wave_env": {"GGML_HIP_IQ4_GATE4": "1"},
                "binary_sha256": {k: {name: sha(d/name) for name in ARTIFACTS} for k,d in runtimes.items()}}
    save(folder/"manifest.json", manifest)
    env = os.environ.copy()
    for key in FLAGS_OFF:
        env.pop(key, None)
    env.update(COMMON_ENV)
    sdk = a.sdk.resolve(strict=True)
    env["PATH"] = os.pathsep.join([str(sdk/"_rocm_sdk_libraries/bin"), str(sdk/"_rocm_sdk_core/bin"), env["PATH"]])
    for variant, name in planned_files(order):
        active = subprocess.check_output(["tasklist", "/FO", "CSV", "/NH"]).lower()
        if b'"llama-server.exe"' in active or b'"llama-bench.exe"' in active:
            raise RuntimeError("Another llama-server or llama-bench process is active")
        exe = runtimes[control if variant==control else "optimized"]/"llama-bench.exe"
        case_env = env.copy()
        if variant=="four":
            case_env["GGML_HIP_IQ4_GATE4"]="1"
        print(f"Starting {name}", flush=True)
        logpath = logs/(name+".log")
        with (raw/(name+".json")).open("xb") as out, logpath.open("wb") as log:
            subprocess.run([str(exe), "-m", model.name, *ARGS], cwd=model.parent, env=case_env,
                           stdout=out, stderr=log, check=True, timeout=600)
        trace = logpath.read_text(encoding="utf8", errors="replace")
        four = "IQ4_NL gate/up: using 4 waves" in trace
        if (four and variant != "four") or (variant == "four" and not a.observe_dispatch and not four):
            raise RuntimeError(f"{name}: unexpected gate/up dispatch")
        print(f"Finished {name}", flush=True)
    print_table(summarize(folder))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    bench = sub.add_parser("run")
    bench.add_argument("--control", "--official", dest="control", type=Path, required=True)
    bench.add_argument("--control-variant", choices=("official", "fix-only"), default="official")
    bench.add_argument("--observe-dispatch", action="store_true",
                       help="Record whether four-wave dispatch occurs on a different quantization")
    for key in ("optimized", "model", "sdk", "output"):
        bench.add_argument("--"+key, type=Path, required=True)
    sub.add_parser("report").add_argument("output", type=Path)
    args = parser.parse_args()
    if args.action=="run":
        run(args)
    else:
        print_table(summarize(args.output))
