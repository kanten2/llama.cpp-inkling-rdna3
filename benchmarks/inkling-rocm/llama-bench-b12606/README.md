# Inkling 2 Mini IQ4_NL on a single Radeon RX 7900 XTX

Windows llama-bench results for the official b12606 ROCm distribution and two decode configurations from this fork. Each cell summarizes five measurements from five fresh processes.

## Results

Mean throughput ± sample standard deviation, in tokens per second (n = 5 per cell). Percentage changes use the unrounded means.

| Test | Official b12606 | Optimized 8 waves | Optimized 4 waves | 8 vs official | 4 vs official | 4 vs 8 |
|---|---:|---:|---:|---:|---:|---:|
| pp512 | 1557.49 ± 10.93 | 1557.21 ± 6.99 | 1556.51 ± 4.46 | -0.02% | -0.06% | -0.04% |
| tg128 | 46.00 ± 0.27 | 52.94 ± 0.09 | 54.16 ± 0.26 | +15.11% | +17.75% | +2.29% |
| tg768 | 46.55 ± 0.08 | 53.34 ± 0.17 | 54.53 ± 0.20 | +14.58% | +17.14% | +2.23% |

On this GPU the four-wave build delivered 17.14% higher tg768 throughput than stock and 2.23% higher than the same optimized build in eight-wave mode. The largest difference between pp512 means was 0.98 tok/s, smaller than the within-variant sample standard deviations. These synthetic generation tests keep the amount of work fixed and complement the [natural-completion chat measurements](../README.md#time-to-a-complete-answer).

## Workload and environment

- One AMD Radeon RX 7900 XTX (gfx1100, 24 GiB), Ryzen 9 5950X, Windows 11 Pro Insider Preview build 28120. Windows display driver version 32.0.32029.2010. ROCm 10.0.0 for gfx1100; the bundled HIP runtime DLL has file version 10.0.3581.0.
- [Inkling-2-Mini-IQ4_NL.gguf](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF), 17,627,940,768 bytes. All runs read the same file. [Model and binary SHA-256 hashes](manifest.json).
- [Official b12606](https://github.com/ggml-org/llama.cpp/releases/tag/b12606) is the control. The optimized runtime uses the official benchmark executable and benchmark implementation DLL, with this fork's `llama.dll` and `ggml-hip.dll`. Both optimized modes use the exact same binaries. The benchmark implementation reports the official build ID `e412c6f46` in every JSON; distinguish variants by filename, manifest and backend checks.
- The IQ4_NL measurements predate upstream #32704. Both optimized modes enable the preceding model and HIP optimizations. Four-wave mode additionally sets `GGML_HIP_IQ4_GATE4=1` for the single-token, routed IQ4_NL gate/up projection. The separate scale/add fusion is disabled for this comparison. No profiling probe is enabled.
- Each process runs prompt processing of 512 tokens, then separate generation workloads of 128 and 768 tokens. All start at context depth zero with llama-bench's built-in warm-up. The allocated context sizes reported in the logs are 512, 256 and 768 respectively. llama-bench supplies its own token IDs; these tests measure throughput separately from chat sampling, answer length and model load time.
- One GPU is visible to HIP, split mode is `none`, and all 49 model layers are offloaded to ROCm0. HIP Graphs are enabled. The command requests 16 CPU threads, batch 2048, microbatch 512 and automatic flash attention. [Per-process checks from stderr](checks.json) include graph reuse, allocated contexts and detection of the four-wave dispatch.

## Reproduce

1. Extract the b12606 Windows x64 ROCm distribution into an official runtime directory. Make a separate copy for the optimized runtime.
2. To recreate the measured IQ4_NL source, use a separate clean copy of this release and reverse the four-file loop-scale patch as described in the [source preparation steps](../ud-q4km-pr32704/README.md#prepare-the-source-trees). Build the `llama` and `ggml-hip` targets with the matching gfx1100 ROCm development SDK. The measurement used Ninja, Release, AMD Clang 23.0.0git, shared libraries, `GGML_BACKEND_DL=ON`, `GGML_HIP=ON`, `GGML_HIP_GRAPHS=ON`, `GGML_NATIVE=OFF` and target gfx1100. The SDK's `_rocm_sdk_devel` directory provided `HIP_PATH` and `ROCM_PATH`; `HIP_DEVICE_LIB_PATH` pointed to its `lib/llvm/amdgcn/bitcode` directory. Place the resulting `llama.dll` and `ggml-hip.dll` in the optimized copy. See the [Windows HIP build instructions](../../../docs/build.md). The `--sdk` argument below points to the matching runtime SDK directory containing `_rocm_sdk_core` and `_rocm_sdk_libraries`.
3. With the GPU available for this benchmark and no llama-server running, execute the following in PowerShell from this directory. Substitute local paths. The runner launches benchmark processes only.

```powershell
python run.py run --official C:\bench\official --optimized C:\bench\optimized --model C:\models\Inkling-2-Mini-IQ4_NL.gguf --sdk C:\bench\rocm-sdk --output C:\bench\new-measurement
```

The runner creates a new set of raw output files. It uses one timed repetition per invocation (`-r 1`), three tests per invocation and 15 invocations. The five rounds rotate variant order to distribute first, middle and last positions. The [manifest](manifest.json) contains the exact order, arguments and relevant environment variables.

## Raw data and aggregation

Each file in [raw/](raw/) is the original llama-bench JSON stdout for one process. Filenames encode round, position in the round and variant. Each contains three test records with nanoseconds and token rates. llama-bench reports zero within-process standard deviation with one repetition. The result table uses the **across-process sample standard deviation** (divisor 4) of the five rates for each build and test. Each rate is the fixed token count divided by measured seconds from `samples_ns`.

The [machine-readable summary](summary.json) contains all 45 rates, means and standard deviations. Run `python run.py report .` to verify the published raw-file hashes and regenerate the summary with Python's standard library.
