# Japanese exact-answer comparison on the RX 7900 XTX

The two optimized decode modes each answered the same 25 Japanese questions. The set has five questions each on arithmetic, logic, code execution, data aggregation and reading a specification. The prompts and expected integer or ordered integer-array answers were frozen before inference. Accuracy here measures final-answer correctness on this set.

| Recorded result | Eight waves | Four waves |
|---|---:|---:|
| Correct | 25/25 | 25/25 |
| Total time to complete 25 answers | 430.55 s | 365.76 s |
| Median time per question | 12.36 s | 12.34 s |
| Generated tokens | 20,507 | 17,780 |
| Aggregate decode speed | 48.62 tok/s | 49.82 tok/s |

The four-wave configuration finished first on 21 questions. The 15.0% reduction in total completion time includes a 13.3% reduction in generated tokens. All 50 responses stopped naturally at context size 4096 with zero cached prompt tokens. [Per-question results](recorded/results.md) · [Recorded final answers and timings](recorded/answers.jsonl) · [Aggregate results](recorded/summary.json)

## Grade the recorded responses

From this directory, use Python 3.9 or newer. Offline grading uses the Python standard library and runs without a model or GPU.

```shell
python make_suite.py
python score.py recorded/answers.jsonl
```

`make_suite.py` recomputes the fixture and compares it byte for byte with [the frozen questions and key](suite.json). `score.py` grades the final-answer strings afresh, checks the matched natural completions and validates prompt and token counts. To write your own reports, add `--json summary.json --markdown results.md`. The scorer accepts width and whitespace variants, code fences and answer labels. It awards full credit only for the expected integer or the expected sequence in order.

## Rerun inference on Windows

1. Prepare the optimized b12606 server using the [runtime and build instructions](../llama-bench-b12606/README.md#reproduce). Use the same runtime for both modes. The reported runs used the IQ4_NL source described there, ROCm 10.0.0 for gfx1100 and one RX 7900 XTX.
2. Obtain `Inkling-2-Mini-IQ4_NL.gguf` and `mmproj-Inkling-2-Mini-Q8_0.gguf` from the [model repository](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF). Supply a runtime directory containing `llama-server.exe` and the matching DLLs, and an SDK directory containing `_rocm_sdk_core` and `_rocm_sdk_libraries`.
3. With the GPU available and the benchmark port free, run the following in PowerShell. Replace the example relative paths with your own. Use a new output directory per run. The runner checks for an existing llama-server before starting its own processes.

```powershell
python run.py --runtime .\deps\optimized --sdk .\deps\rocm-sdk --model .\models\Inkling-2-Mini-IQ4_NL.gguf --mmproj .\models\mmproj-Inkling-2-Mini-Q8_0.gguf --output .\runs\trial-01
```

The port defaults to 18203 on loopback and can be set with `--port`. Relative input and output paths resolve from the working directory. The runner writes per-attempt final answers and timings to `answers.jsonl`, input hashes to `run-info.json`, and server logs, `summary.json` and `results.md` to the output directory. Grading a fresh run uses the same command as offline grading with its new answers file.

## Protocol

The [recorded protocol](protocol.json) lists the exact run order, context tiers and SHA-256 of the original backend and question set. The first 12 questions run eight waves then four; the remaining 13 run four then eight. Each block starts a fresh server and issues the same 32-token warm-up request. The four-wave process enables `GGML_HIP_IQ4_GATE4=1`; the eight-wave process uses the default. Both enable HIP Graphs and disable the separate scale/add fusion.

Each question is a fresh single-message chat with temperature 0, seed 42, parallelism 1 and prompt caching disabled. The cap is context size minus 256 tokens. If either mode reaches its cap, the runner repeats that question in both modes at the next context size. Only complete matched pairs contribute to accuracy and time. Wall time runs from request submission to the end of the response stream on a loaded server; generated tokens include the reasoning portion. Timing varies across runs and systems.
