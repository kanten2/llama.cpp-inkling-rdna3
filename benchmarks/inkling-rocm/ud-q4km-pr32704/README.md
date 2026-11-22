# Unsloth UD-Q4_K_M and IQ4_NL on the RX 7900 XTX

## Throughput

| GGUF | Test | Control | Optimized, 8-wave setting | Optimized, 4-wave setting |
|---|---|---:|---:|---:|
| Unsloth UD-Q4_K_M | pp512 | 1486.26 ± 5.29 | 1484.88 ± 3.33 | 1486.49 ± 4.34 |
| Unsloth UD-Q4_K_M | tg128 | 48.56 ± 0.45 | 56.16 ± 0.50 | 56.40 ± 0.34 |
| Unsloth UD-Q4_K_M | tg768 | 48.63 ± 0.93 | 56.67 ± 0.48 | 56.75 ± 0.52 |
| IQ4_NL (earlier run) | pp512 | 1557.49 ± 10.93 | 1557.21 ± 6.99 | 1556.51 ± 4.46 |
| IQ4_NL (earlier run) | tg128 | 46.00 ± 0.27 | 52.94 ± 0.09 | 54.16 ± 0.26 |
| IQ4_NL (earlier run) | tg768 | 46.55 ± 0.08 | 53.34 ± 0.17 | 54.53 ± 0.20 |

Each cell is mean ± sample standard deviation in tok/s, with five fresh processes per cell. pp512 processes a 512-token prompt. tg128 and tg768 generate 128 and 768 tokens from context depth zero. The 4-wave column requests the fork's IQ4_NL gate/up specialization. The UD-Q4_K_M control includes the loop-scale fix. The historical IQ4_NL control is the original b12606 distribution. Compare the three columns within a GGUF to estimate the optimization gain for that file.

## What actually ran

On UD-Q4_K_M, the optimized default improved tg128 by **15.66%** and tg768 by **16.54%** over the fix-only control. The pp512 means differed by less than 0.1% between those configurations. All five samples are included, including the final fix-only tg768 run at 46.99 tok/s.

With the four-wave flag set, the specialized kernel logged **0/5** process activations on UD-Q4_K_M and **5/5** on the earlier IQ4_NL runs. The 0.14% tg768 difference between the two optimized UD settings amounts to 0.08 tok/s with the same dispatch. The IQ4_NL run measured an additional 2.23% with the specialization active.

The routed gate/up weight has shape `2304 × 2304 × 128` in both files. In IQ4_NL, physical layers 0–19 store that tensor in IQ4_NL and 20–23 use Q8_0. In Unsloth UD-Q4_K_M, layers 0–19 use NVFP4 and 20–23 use Q5_K. This fork selects the four-wave kernel inside the IQ4_NL single-token path. The two optimized modes use identical DLLs and differ only in the gate/up flag. [GGUF tensor inventory](tensor-types.json) · [Kernel selection code](../../../ggml/src/ggml-cuda/mmvq.cu) · [Per-process UD checks](checks.json) · [IQ4_NL checks](../llama-bench-b12606/checks.json)

## Answer checks

All three UD configurations scored **5/5**. All 15 responses ended naturally. The five final answers matched across modes, and each mode generated 3,274 tokens across the five requests.

| Question | Task | Final answer from all three modes |
|---|---|---|
| M1 | Price after discount, tax and shipping | `3338` |
| M3 | Conditional probability | `[3,19]` |
| C2 | Trace a loop with simultaneous assignment | `[8,6,8]` |
| D4 | Inventory over time | `[3,9]` |
| L1 | Ordering under constraints | `[2,4,3,1,5]` |

The five question IDs were fixed before inference. Each mode ran one fresh text-only server with a 4,096-token context, a single slot, seed 42, temperature 0, prompt caching disabled and the same warm-up. Each question allowed 3,840 generated tokens. The scorer required a natural stop and an exact integer or ordered list from the final answer. Scores cover these five questions. The earlier IQ4_NL test used 25 questions and two optimized modes. [Questions and answer key](../quality/suite.json) · [Recorded UD answers](answers.json) · [Earlier IQ4_NL quality report](../quality/README.md)

## Model and runtime identity

| GGUF | Bytes | SHA-256 of the measured file |
|---|---:|---|
| Unsloth Mainline_Support UD-Q4_K_M | 16,610,544,608 | `78fcedcf93c5d1bfe83e225513b71392bb52259be0eaa5f8899c856a2e5834ae` |
| ggml-org IQ4_NL | 17,627,940,768 | `cbdf3fa9dc50a4124977980f37da4f1fa764b0cfe800d275cd43d26ce2a81af4` |

Unsloth distributes the tested file as `Mainline_Support/Inkling-2-Mini-UD-Q4_K_M.gguf_file` at revision `45fd08764f2b0bf16b03a5483a80e5c7aaec8772`. We used a local `.gguf` filename for the same bytes. [Pinned Unsloth directory](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/tree/45fd08764f2b0bf16b03a5483a80e5c7aaec8772/Mainline_Support)

All UD runs used b12606 (`e412c6f468b7b9e0f16f2540244236f8ad68e9ae`) with the same upstream fix (`690b87d88151dbd775bb6eff5581663688785928`). It shares the physical layers with the loop slots after the loader attaches per-expert scales. All modes kept the official benchmark executable and benchmark implementation DLL. This is why raw JSON reports the official benchmark build ID for all modes. The manifest identifies the actual DLLs by hash. [Upstream fix #32704](https://github.com/ggml-org/llama.cpp/pull/32704) · [UD binary hashes, flags and round order](manifest.json) · [Earlier IQ4_NL manifest](../llama-bench-b12606/manifest.json)

| UD configuration | `llama.dll` source | HIP backend |
|---|---|---|
| Fix only | Upstream b12606 plus #32704 | Official b12606 ROCm DLL |
| Optimized, default | This release's source tree | Fork's gfx1100 DLL, gate/up flag unset |
| Optimized, 4 waves requested | Same as optimized default | Identical DLL, `GGML_HIP_IQ4_GATE4=1` |

## Prepare the source trees

The release contains the optimized UD-Q4_K_M source. For the UD fix-only control, apply the upstream loop-scale fix to upstream b12606. For the earlier IQ4_NL optimized build, reverse that fix in a separate copy of this release. The PowerShell example starts in this release's repository root and writes to new sibling directories.

```powershell
git clone --branch b12606 https://github.com/ggml-org/llama.cpp.git ../inkling-fix-only
$fixCommit = '690b87d88151dbd775bb6eff5581663688785928'
git -C ../inkling-fix-only fetch --no-tags origin $fixCommit
$patchFile = Join-Path (Resolve-Path ..).Path 'inkling-loop-scales.patch'
git -C ../inkling-fix-only diff "--output=$patchFile" "$fixCommit^" $fixCommit -- src
git -C ../inkling-fix-only apply --check $patchFile
git -C ../inkling-fix-only apply $patchFile
git clone --no-local . ../inkling-iq4-source
git -C ../inkling-iq4-source apply --check --reverse $patchFile
git -C ../inkling-iq4-source apply --reverse $patchFile
```

## Reproduce

Use a Windows ROCm 10.0.0 gfx1100 runtime and development SDK. This machine ran Windows 11 build 28120, a Ryzen 9 5950X and one visible RX 7900 XTX (driver 32.0.32029.2010). HIP Graphs were enabled. The benchmark used 16 CPU threads, all 49 layers offloaded, split mode none, batch 2048, microbatch 512 and automatic flash attention. Each process ran all three built-in llama-bench tests in order, one timed repetition each. Allocated context sizes for those tests were 512, 256 and 768. Only the answer check used context 4096. The runner interleaves the three configurations across five rounds. [Earlier IQ4_NL method and raw JSON](../llama-bench-b12606/README.md)

1. Extract the [official Windows ROCm b12606 release](https://github.com/ggml-org/llama.cpp/releases/tag/b12606) into two separate experiment directories. In the fix-only source directory above, build the `llama` target and copy its `llama.dll` into the first directory. In this release's source directory build `llama` and `ggml-hip`, then copy both DLLs into the second directory. Both optimized settings run from this same directory. The recorded run reused the optimized HIP DLL from the IQ4_NL experiment. The release, compiler, CMake flags and SDK setup are documented in the [gfx1100 build notes](../llama-bench-b12606/README.md#reproduce).
2. From the directory containing this README, download the exact GGUF and give it the measured filename. The following PowerShell commands use paths relative to the current directory. Set the three runtime variables to your local directories.
3. With the GPU available, run the answer checks, then the benchmark. Each output directory must be fresh. The `--observe-dispatch` option records whether the specialization actually ran.

```powershell
hf download unsloth/Inkling-2-Mini-GGUF Mainline_Support/Inkling-2-Mini-UD-Q4_K_M.gguf_file --revision 45fd08764f2b0bf16b03a5483a80e5c7aaec8772 --local-dir ./weights
$download = Get-Item ./weights/Mainline_Support/Inkling-2-Mini-UD-Q4_K_M.gguf_file
if ((Get-FileHash $download.FullName).Hash -ne '78fcedcf93c5d1bfe83e225513b71392bb52259be0eaa5f8899c856a2e5834ae') { throw 'GGUF hash mismatch' }
$model = (New-Item -ItemType HardLink -Path ./weights/Inkling-2-Mini-UD-Q4_K_M.gguf -Target $download.FullName).FullName
$fix = './runtimes/fix-only'
$opt = './runtimes/optimized'
$sdk = './runtimes/rocm-sdk'
$common = @('--model', $model, '--control', $fix, '--optimized', $opt, '--sdk', $sdk)
python validate.py @common --output ./new-answers
python ../llama-bench-b12606/run.py run @common --control-variant fix-only --observe-dispatch --output ./new-bench
```

To check the published dataset offline, run `python ../llama-bench-b12606/run.py report .` from this directory. It checks the 15 raw-file digests and recomputes all 45 rates from `samples_ns`. To repeat the log checks as well, run `Expand-Archive ./bench-stderr.zip ./logs` first. The archive contains full stderr for every benchmark process, with local DLL directory prefixes replaced by `<fix>/` or `<opt>/`. [Benchmark stderr](bench-stderr.zip) The sample standard deviation uses divisor four. To inspect a pair of local GGUFs, install this checkout's `gguf-py` dependencies and run `python tensor_types.py $iq4Model $model --output ./new-tensor-types.json`, where `$iq4Model` points to the IQ4_NL file. [Original UD JSON](raw/) · [All UD samples, means and deviations](summary.json)
