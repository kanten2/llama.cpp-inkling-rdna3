# Inkling 2 Mini on RDNA3 ROCm

This directory records the baseline for an RX 7900 XTX optimization experiment. The llama.cpp source is pinned to the official b12606 release (commit e412c6f468b7b9e0f16f2540244236f8ad68e9ae). The baseline uses the official Windows x64 ROCm build and matching ROCm 10.0.0 libraries for gfx1100.

The [three-way llama-bench comparison](llama-bench-b12606/README.md) includes pp512, tg128 and tg768 results for official b12606, optimized eight-wave and optimized four-wave decode, with the five samples per test and their original JSON outputs.

The [Unsloth UD-Q4_K_M comparison](ud-q4km-pr32704/README.md) repeats those tests with the looped expert-scale fix in all three configurations. It includes exact-answer checks and identifies the gate/up weight types in both GGUFs.

The [Japanese 25-question comparison](quality/README.md) includes the frozen answer key, all 50 final answers, per-question completion times, and commands for fresh inference and offline grading.

## Run a fixed-length chat measurement

Obtain the model and projector from [ggml-org/Inkling-2-Mini-GGUF](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF). With the matching ROCm libraries on PATH, start just one inference server on the GPU.

```powershell
.\llama-server.exe --model .\Inkling-2-Mini-IQ4_NL.gguf --mmproj .\mmproj-Inkling-2-Mini-Q8_0.gguf --alias inkling-2-mini --host 127.0.0.1 --port 18188 --ctx-size 4096 --parallel 1 --gpu-layers 99 --device ROCm0 --jinja
```

Run from the repository root once startup has finished. Python uses only the standard library.

```powershell
python benchmarks/inkling-rocm/bench.py --runs 3
```

The runner uses counting problem M4 from the [25-question suite](quality/suite.json). It uses temperature 0, seed 42 and an uncached request, and checks for 76 input tokens and 768 generated tokens. Check the server log for the expected ROCm device and all 49 layers offloaded.

## Historical chat baseline

RX 7900 XTX (gfx1100), Windows, AMD display driver 26.20.29.02, IQ4_NL, context 4096. The historical chat measurements below used a different, 103-token input. The independently reproducible throughput comparisons use llama-bench's built-in token sequences.

| Configuration | Decode tok/s | Total response time |
| --- | ---: | ---: |
| Stock b12606, mean of 3 runs | 45.45 | 17.18 s |
| Stock b12606, one ETW run | 45.07 | 17.34 s |
| b12606 with HIP graphs disabled, one ETW run | 43.43 | 17.98 s |

The stock comparison runs measured a peak of 17.855 GiB dedicated GPU memory per process. Debug graph assignments placed the input embedding lookup on the CPU and the expert matrix operations on ROCm. A separate 16.9 ms Radeon GPU Profiler capture sampled 27.85% GPU idle and frequently observed scale, broadcast multiply, and broadcast add kernels. The sample identifies candidates for fusion. Whole-request speedups must be measured independently.

The first experiment should preserve the model, quantization, prompt, context, and generation budget. Compare the patched binary with this baseline on the same machine and check numerical and model-quality regressions before drawing a conclusion.

## First experiment - fuse scale and residual add

The HIP backend can combine a contiguous F32 SCALE followed by an ADD with an equally shaped residual. The scale must have a single consumer, and the fused kernel accepts either disjoint buffers or a full elementwise alias. Broadcast and other layouts continue through the existing operations.

The control and experiment use the same locally built ROCm 10.0.0 gfx1100 backend DLL, the official b12606 server binaries, and the fixed 768-token generation workload above. Set `GGML_HIP_DISABLE_SCALE_ADD_FUSION=1` before starting the test server for the control and remove that variable before starting it for the experiment.

| Mode | Decode tok/s, mean of 3 | Total response time, mean of 3 |
| --- | ---: | ---: |
| Fusion disabled | 45.83 | 17.04 s |
| Fusion enabled | 46.07 | 16.95 s |

The observed decode gain was 0.54% in one control-then-experiment run order. Peak dedicated GPU memory stayed at 17.855 GiB. The six generated messages, including reasoning, were identical for this workload. The ROCm backend tests also passed for the decode shape, a batched shape, and a broadcast case. The result supports a small effect from this fusion on this workload. It does not establish a broader model-quality result or the 50 tok/s target.

## Row clipping during single-token decode

The HIP prototype matches the eight operations from SQR through the final MUL. It retains the original FP32 square, SUM_ROWS reduction tree, square root, addition of 1e-6, division, clamp and multiply. Q and R retain their preceding learned per-channel multiply. Both comparison modes disable the earlier SCALE + ADD patch with `GGML_HIP_DISABLE_SCALE_ADD_FUSION=1`. Set `GGML_HIP_DISABLE_ROW_CLIP=1` to measure the original graph and unset it to test the fusion.

Build `clip-bench.cpp` against the b12606 ggml and ggml-base libraries and run `clip-bench <ggml-hip.dll> <output-prefix>` once per process. Enable HIP Graphs in the build and leave `GGML_CUDA_DISABLE_GRAPHS` unset. The benchmark enqueues 32 independent clips per graph, warms the graph, and times nine batches of 64 replays with synchronization at the batch boundary. The following times are per-clip amortized wall times on the RX 7900 XTX. Each entry is the median of three process medians. The six process runs used the order off, on, on, off, off, on.

| Tensor (row width x rows) | Original, us | Fused, us |
| --- | ---: | ---: |
| Q (128 x 24) | 17.72 | 3.63 |
| K (128 x 8) | 13.92 | 2.24 |
| R (16 x 24) | 17.59 | 3.64 |

The initial prototype produced different output on the fixed prompt. Randomly generated rows exposed small differences in Q and K. The revised kernel writes the FP32 squares to shared memory before calling the same reduction function as SUM_ROWS. Its F32 outputs matched byte for byte across all six runs on fixed inputs and on 256 additional input sets per shape (1,146,880 random-case output elements per mode per run). The isolated-graph estimate from the three timing differences and 48 layer visits is 1.91 ms per token.

The full-model comparison used the same revised DLL in both modes on the RX 7900 XTX. Four fresh server processes ran in on/off/off/on order, with two uncached requests per process, HIP graphs enabled, and the model, projector, prompt, seed, temperature and token counts described above.

| Clip fusion | Mean decode tok/s (range) | Mean ms/token | Mean request time | Peak dedicated VRAM |
| --- | ---: | ---: | ---: | ---: |
| Off | 45.01 (44.75-45.11) | 22.19 | 17.35 s | 17.855 GiB |
| On | 47.98 (47.63-48.12) | 20.82 | 16.29 s | 17.855 GiB |

This saved 1.37 ms per generated token (+6.61% throughput). All eight response messages matched the separate official b12606 response exactly. On this question, the 768-token limit was reached during reasoning. The reasoning text matched and the final answer field was empty in all modes.

## Residual block mixing during single-token decode

With row clipping enabled, the graph builder can reuse the concatenation of completed blocks. Each later pass also reads the same message bank in 12 layers. Its normalized states and contiguous transpose can be shared across those layers. Each layer still computes scores with its own query, applies softmax and forms the original weighted sum. Reuse is scoped to the graph being built. Set `LLAMA_INKLING_DISABLE_MIX_REUSE=1` before starting the test process for the clip-only control. Both modes enable clip fusion and disable SCALE + ADD fusion.

The ROCm split of a one-token schedule contained 120 mixes of width 2304. The states column includes the current state. Times are amortized microseconds per mix on the RX 7900 XTX. Concat and transpose are separate component microbenchmarks with their own replay overhead.

| States | Queries | Mixes/token | Original | Concat only | Transpose copy only | Reuse |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1 | 13 | 14.16 | 3.44 | 4.04 | 11.98 |
| 2 | 3 | 1 | 14.80 | 3.35 | 4.01 | 13.28 |
| 2 | 4 | 24 | 15.20 | 3.31 | 3.79 | 8.76 |
| 3 | 1 | 33 | 16.00 | 4.31 | 2.53 | 15.93 |
| 3 | 3 | 3 | 16.21 | 4.25 | 2.52 | 16.44 |
| 4 | 1 | 31 | 17.17 | 6.06 | 2.70 | 16.16 |
| 4 | 3 | 3 | 17.70 | 6.05 | 3.00 | 17.23 |
| 5 | 1 | 11 | 18.84 | 8.16 | 3.15 | 16.36 |
| 5 | 3 | 1 | 19.32 | 8.00 | 3.34 | 16.86 |

Build `mix-bench.cpp` against ggml and ggml-base. Run `mix-bench HIP-backend-DLL mode output-prefix [dump]` with mode `stock`, `reuse`, `concat` or `transpose`. The median across three processes uses seven timed batches of 32 HIP Graph replays per process. Each graph contains 12 mixes with a shared prefix and distinct queries. The four-query case also shares the final bank state. Actual prefix lifetimes vary in the full graph. The stock and reuse outputs matched byte for byte for 16 random input sets, 12 mixes per shape and 8,847,360 FP32 outputs in total. Weighting the isolated timing differences by the observed visits predicts 0.248 ms saved per token.

Using the same locally built b12606-based llama DLL and the same clip-enabled HIP DLL for both arms, the 768-token chat benchmark ran in off/on/on/off process order with two requests per process. Clip alone averaged 48.08 tok/s (47.99 to 48.18); adding mix reuse averaged 48.98 tok/s (48.51 to 49.16). Mean decode time fell from 20.772 to 20.390 ms per token and mean request time from 16.27 to 15.98 s. Peak dedicated VRAM was 17.855 GiB in both arms. All eight messages matched the official b12606 reference, including the reasoning text and empty final answer at the length limit.

## Four-tap convolutions during decode

The HIP backend can compute a four-tap SSM_CONV, residual ADD and SCALE in one kernel. The matcher follows the intervening RESHAPE and checks input shapes, layout, intermediate uses and buffer overlap. It handles one token and one sequence. The original four products are accumulated in tap order.

Use the same clip-enabled, mix-reuse build for both arms. Keep HIP Graphs enabled and set `GGML_HIP_DISABLE_SCALE_ADD_FUSION=1`. Start the control with `GGML_HIP_DISABLE_CONV_RESIDUAL=1` and remove this variable for the fused run. A separate run with `GGML_HIP_TRACE_CONV_RESIDUAL=1` logs the matched nodes while the backend constructs the graph.

Build `conv-bench.cpp` against ggml and ggml-base. Run `conv-bench <ggml-hip.dll> <prefix> [dump]` for each configuration in a fresh process. Each graph includes 32 independent convolutions. Each process warms the graph, then measures nine batches of 64 replays. The table gives median amortized wall time per convolution across three fresh processes per mode. Run order was off, on, on, off, off, on. With the optional dump argument the benchmark writes 16 additional deterministic random input sets per shape for byte comparison.

| Channels | Original, us | Fused, us | Bitwise-matched FP32 outputs |
| ---: | ---: | ---: | ---: |
| 1024 | 5.40 | 2.01 | 524288 |
| 2304 | 5.28 | 2.04 | 1179648 |

A separate trace of the fixed prompt showed 80 matches at 1024 channels and 96 at 2304 channels in each of five observed single-token graph passes. These 176 sites account for 352 fewer kernel launches per decode graph replay. The other 16 convolutions have input-dependent taps.

The timed chat ran in off/on/on/off process order, two requests per process. Both arms used the same binaries with tracing disabled.

| Mode | Decode mean (range), tok/s | Decode ms/token | Mean request time, s | Peak dedicated VRAM, GiB |
| --- | ---: | ---: | ---: | ---: |
| Clip + mix reuse | 49.18 (49.08-49.32) | 20.307 | 15.92 | 17.855 |
| Clip + mix reuse + convolution fusion | 50.59 (50.38-50.75) | 19.743 | 15.49 | 17.855 |

The measured saving was 0.564 ms/token (+2.86% throughput). Weighting the isolated timings by the traced site counts gives 0.583 ms/token. All eight messages matched the official b12606 reference for this prompt. Each reached the 768-token limit with 3119 characters in the reasoning field and an empty final-answer field.

## Updating convolution history during decode

During one-token decode, each four-tap convolution needs the window `[t-3, t-2, t-1, t]` and stores `[t-2, t-1, t]` for the next token. The HIP kernel writes both buffers. It matches CONCAT, two views and a CPY into the state. The matcher checks the tail offset, tensor layouts, computed copy and disjoint input/output ranges. It leaves the full window available for every consumer, including input-dependent taps.

For the control, enable all three preceding clip, mix reuse and convolution optimizations, keep `GGML_HIP_DISABLE_SCALE_ADD_FUSION=1`, and set `GGML_HIP_DISABLE_CONCAT_HISTORY=1`. Unset the last variable in the experimental process. A separate run with `GGML_HIP_TRACE_CONCAT_HISTORY=1` matched 192 sites per single-token graph in five observed passes.

Run `conv-bench --history <ggml-hip.dll> <prefix> [dump]` using the same replay and process counts as the convolution benchmark above. The dump includes both outputs and eight guard elements on either side of the state slice. Both configurations produced byte-identical dumps. Every stored state also matched the last three taps of its corresponding window.

| Channels | Sites/token | Original, us | Fused, us |
| ---: | ---: | ---: | ---: |
| 1024 | 96 | 3.99 | 2.04 |
| 2304 | 96 | 3.99 | 2.02 |

The 768-token chat comparison used the same binary for both arms, two requests per process in off/on/on/off order, with tracing disabled.

| Mode | Decode mean (range), tok/s | Decode ms/token | Mean request time, s | Peak dedicated VRAM, GiB |
| --- | ---: | ---: | ---: | ---: |
| Previous convolution build, history fusion off | 50.55 (50.43-50.61) | 19.757 | 15.495 | 17.855 |
| History fusion on | 51.64 (51.52-51.73) | 19.341 | 15.171 | 17.855 |

History fusion saved 0.416 ms/token (+2.15% throughput). Weighting isolated timings by the traced site counts predicts 0.375 ms/token. All eight response messages matched the official b12606 reference, including the reasoning field and empty final answer at the 768-token limit.

A separate two-turn chat checked reuse of the conversation cache on fresh control and fused servers. Both answered `5` to "What is 2 + 3?" and `9` to "Add 4 to your last answer." The second request reused 25 prompt tokens in both modes and the full messages matched on both turns.

## In-model IQ4_NL matrix-vector profile

With history fusion enabled, a HIP Graph probe identifies the single-token Inkling graph from its first Q projection, then records GPU wall-clock stamps before and after Q8_1 activation quantization and the IQ4_NL matrix-vector dispatch. Two adjacent stamps give an empty interval. Elapsed times below subtract the per-process median empty interval from each measured interval. They estimate incremental dispatch time on the graph stream, including scheduling between markers.

Run one fresh single-GPU server per stripe with `GGML_IQ4_PROFILE_STRIDE=4` and `GGML_IQ4_PROFILE_STRIPE` set to 0, 1, 2 or 3. Use the same 768-token workload and prior fusion settings for every process. The probe reads back every 32nd cached graph replay starting at replay 16. The four stripes cover every site at each of 24 matching replay positions. A fifth process with stride 1 and stripe 0 times every site in one request. With the stripe variable absent the probe is inactive. Its two bounded timing buffers live for the lifetime of the profiling process.

| IQ4_NL projection (input columns x output rows per expert) | Dispatches/token | Matvec us/dispatch | Matvec ms/token | Q8_1 prep ms/token | Share of IQ4_NL matvec |
| --- | ---: | ---: | ---: | ---: | ---: |
| MoE gate/up (2304 x 2304) | 44 | 44.12 | 1.941 | 0.063 | 56.65% |
| MoE down (1152 x 2304) | 44 | 33.77 | 1.486 | 0.061 | 43.35% |

Each dispatch uses eight routed experts from a bank of 128. Each projection has 20 distinct IQ4_NL weight tensors across the visited layers. The all-site check measured 1.937 and 1.479 ms/token respectively. The lower-overhead four-stripe totals include 3.427 ms/token for IQ4_NL matvec and 0.125 ms/token for its input quantization. Other weight formats and GPU operations are outside these totals.

The uninstrumented history build generated 51.23 and 51.44 tok/s on either side of the five probe runs. Its mean server-reported decode time was 19.480 ms/token, so IQ4_NL matvec accounted for an estimated 17.6%. Peak dedicated VRAM was 17.855 GiB and all seven messages matched the official reference through the length limit. Under the same baseline, reaching 52 tok/s through gate/up alone would require saving 0.249 ms/token, or 5.67 us on each of the 44 dispatches, assuming other work takes the same time.

The benchmark log summarizer (`python benchmarks/inkling-rocm/iq4-profile.py <run-directory>`) checks site identity, stripe coverage, replay alignment, input/output lengths and exact response equality across seven requests. The directory contains the four stripe logs, the all-site log, before/after original-build controls, their saved server responses, VRAM summaries and the official b12606 reference.

## Time to a complete answer

Two fresh servers per build ran in official, history, history, official order. The model, prompt, greedy sampling, context of 4096, single RX 7900 XTX and HIP graphs match the 768-token test. The streaming client collected the first nonblank final-answer text and the end of the stream. Wall times start when the request is sent to an already loaded server. All four uncached requests took 103 input tokens, generated 2136 tokens and stopped naturally within the 3993-token ceiling. The final 889-character answer and the reasoning text each had the same SHA-256 across all four requests.

| Build | Mean decode tok/s | First final text (s) | Answer complete (s) |
| --- | ---: | ---: | ---: |
| Official b12606 | 39.97 | 38.94 | 53.75 |
| History fusion | 48.77 | 31.57 | 44.11 |

The 768-token control measures early decode throughput. At that limit the official and history builds have produced no final-answer text for this prompt.

## Four-wave gate/up experiment

Set `GGML_HIP_IQ4_GATE4=1` before starting the experimental server to select four waves per workgroup for the RDNA3 IQ4_NL single-token, 2304 x 2304 routed gate/up projection with 8 of 128 experts. Its default is eight waves. The 1152 x 2304 down projection uses the same eight-wave kernel in both modes.

| Measurement | Eight waves | Four waves |
| --- | ---: | ---: |
| Model decode tok/s, median of four fresh servers | 49.24 | 50.19 |
| Minimum to maximum tok/s | 46.13-50.21 | 48.34-52.11 |
| Peak sampled process dedicated VRAM, GiB | 17.855 | 17.855 |
| In-model gate/up, us/dispatch | 43.44 | 33.25 |
| In-model down, us/dispatch | 33.37 | 33.31 |
| Fixed expert IDs and inputs, gate/up graph, us/projection | 47.66 | 37.46 |

The eight uninstrumented 768-token requests used the order 8, 4, 4, 8, 4, 8, 8, 4. Each mode was internally deterministic. Only the eight-wave responses matched the official reference. Each four-wave run emitted the first 77 characters of its final answer before the cut. The server clocked the in-model dispatches in separate runs with a dense probe, 24 sampled replays and 44 gate/up calls per token. The gate/up difference is about 0.449 ms/token. The generated sequences and expert routing can diverge across modes.

Build `iq4-gate-check.cpp` against the same ggml libraries as `conv-bench.cpp` and run `iq4-gate-check <ggml-hip.dll> <model.gguf> <prefix> --time` in a fresh process for each mode. The graph contains 16 projections of layer 0 with deterministic FP32 activations and disjoint expert sets that span all 128 experts once. Timing includes Q8_1 preparation and graph execution. The table uses the mean of two process medians per mode in 8, 4, 4, 8 order. Against eight waves, 172462 of 294912 gate/up FP32 values differed in bits, the maximum absolute difference was 7.153e-7 and the relative L2 error was 8.181e-8. All 294912 down values matched bitwise.

One four-wave run to natural completion generated 1368 tokens at 48.70 tok/s. Final text arrived at 15.50 s and the full 922-character answer at 28.38 s. Eight-wave history fusion took 2136 tokens for its 889-character answer. The different number of generated tokens explains most of the wall-time difference.
