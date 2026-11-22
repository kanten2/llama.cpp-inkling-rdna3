# Long-prompt prefill on a Radeon RX 7900 XTX

The v0.1.0 HIP kernel gathers Inkling's 24 relative-position bias values directly into the F32 layout used to prepare the attention mask. The current RDNA3 implementation also fuses mask addition and the final F16 cast when the graph has a single stream and a compatible mask. It adds in F32 and rounds once, using the same conversion as ggml's standalone cast. Other layouts retain their existing execution path.

## Direct relative-position lookup in attention

Set both `INKLING_PP_FUSE=1` and `INKLING_ATTN_DIRECT=1` before starting the experimental HIP build. The `INKLING_ATTN_DIRECT_ACTIVE` log entry confirms that a prefill graph used the specialization. It reads the indexed F32 bias and shared input mask inside the 128-dimensional attention tile, then rounds their sum to F16 before applying the score adjustment. The KV partition follows the dense kernel so the softmax reduction keeps its original order. The graph matcher checks that the inputs remain live through attention and falls back to the ordinary mask when the tensors overlap. The specialization applies to RDNA3.0 prefill batches with more than 32 queries.

On the RX 7900 XTX, two fixed 16,384-token IQ4_NL prompts with 1,000 generated tokens per request gave median prefill times of 12.701 s with the fused dense mask and 12.837 s with direct lookup (four requests per build). Both prompts produced identical token sequences. On the first prompt all 200,058 FP32 logits matched byte for byte at all 1,000 generation positions. The option remains off by default.

### Experimental RDNA3 WMMA prefill

With direct lookup enabled, set `INKLING_ATTN_WMMA=1` before starting the HIP server to select the experimental 64-query WMMA tile. Look for `INKLING_WMMA_ACTIVE` in the log. Each wave reads a head-specific FP16 bias tile into shared memory. The kernel reuses ggml's WMMA and softmax implementation, with FP16 operands and FP32 accumulation for both the score and weighted-value products. Single-token decode uses the usual path. The WMMA prefill changes the floating-point reductions and can change greedy generation. Leave this flag unset to use the original direct-lookup kernel.

On the RX 7900 XTX, the mean GPU time in the prefill attention interval fell from 2.566 s to 1.862 s for two 16,384-token prompts. Median prefill time fell from 12.801 s to 12.060 s over four requests per build, each generating 1,000 tokens. The two greedy continuations both diverged at token five. On a fixed continuation, 973 of 1,000 top predictions agreed. For 27 sampled attention rows, the largest relative L2 error against an independent FP64 calculation was 0.097%.

## Fused mask addition and cast

On a Windows RX 7900 XTX with Inkling-2-Mini-IQ4_NL, the server processed two fixed 16,384-token prompts twice per build and generated 1,000 tokens per request. The table gives the median of four warm runs per build, with build and prompt order reversed on the repeat. Both used a 17,408-token context, batch 2,048, microbatch 512, F16 KV, one slot, greedy decoding and no prompt cache. Only the HIP backend DLL changed.

| HIP backend | Prompt (tok/s) | Generation (tok/s) | Request time (s) |
| --- | ---: | ---: | ---: |
| v0.1.0 | 1,044.2 | 49.16 | 36.07 |
| Fused mask addition and cast | 1,280.1 | 49.14 | 33.14 |

The generated token IDs matched on both prompts. A separate diagnostic compared every final F16 mask element produced by the new path with the previous operations on the same inputs. It found zero differing values across all 48 layers at 4,096 and 16,384 input tokens. Replaying the first prompt and its continuation through both backends also produced byte-identical FP32 logits for all 200,058 vocabulary entries at each of the 1,000 output positions.

## Original gather fusion throughput

One RX 7900 XTX (gfx1100), Windows ROCm, b12606 with the loop-scale fix (#32704). Each row compares fusion off and on in the **same build**. The IQ4_NL timing binary had diagnostic timestamps disabled. The UD-Q4_K_M binary had the instrumentation removed. These measurements were made on the preview 2–based prototype before packaging preview 3. Each cell is the mean ± sample standard deviation across five timed runs after a full-length warm-up. Load time is excluded. Seconds are the mean of the individual elapsed times.

| GGUF | Input tokens | Fusion off (tok/s) | Fusion on (tok/s) | Mean time off → on | Throughput gain |
|---|---:|---:|---:|---:|---:|
| IQ4_NL | 4,096 | 1,200.18 ± 2.75 | 1,339.48 ± 1.49 | 3.413 → 3.058 s | +11.61% |
| IQ4_NL | 16,384 | 884.48 ± 25.79 | 1,210.57 ± 2.22 | 18.537 → 13.534 s | +36.87% |
| UD-Q4_K_M | 4,096 | 1,215.17 ± 5.39 | 1,345.02 ± 2.02 | 3.371 → 3.045 s | +10.69% |
| UD-Q4_K_M | 16,384 | 883.90 ± 1.98 | 1,179.72 ± 1.67 | 18.536 → 13.888 s | +33.47% |

Batch 2,048, microbatch 512, K/V both F16, Flash Attention auto, context depth zero, 16 CPU threads, all 49 layers on one GPU. Each input length ran in four processes in this order: off ×2, on ×3, on ×2, off ×3. The official bench executable reports its base commit in the JSON. [Raw benchmark JSON](raw/) · [GPU and fusion dispatch records](dispatch.json)

The preview 3 ZIP was checked separately after extraction. The package record identifies the ZIP by SHA-256 and includes browser chat answers, `bench.cmd` generation rates and one timed pp16384 run per mode for each GGUF. [Windows package verification](preview-3-package.json)

| GGUF | SHA-256 |
|---|---|
| ggml-org IQ4_NL | `cbdf3fa9dc50a4124977980f37da4f1fa764b0cfe800d275cd43d26ce2a81af4` |
| Unsloth Mainline_Support UD-Q4_K_M | `78fcedcf93c5d1bfe83e225513b71392bb52259be0eaa5f8899c856a2e5834ae` |

## Check the measurements

From this directory, set `$runtime` to the extracted ZIP's `runtime/bin` directory and `$model` to the GGUF. Use a fresh output directory for each model. Close other GPU inference sessions before timing.

```powershell
python measure.py report raw/IQ4_NL
python measure.py report raw/UD-Q4_K_M
python measure.py run --runtime $runtime --model $model --output ./new-measurements
```

## Answer checks

The long-context task retrieves three reference numbers in order from a 14,999-token registry. Each case used one uncached request per mode, context 16,384, seed 42, temperature 0 and a 1,152-token output cap. IQ4_NL answered the original case correctly in both modes (17.73 → 13.72 s prompt time). UD-Q4_K_M answered the original and both variants correctly in both modes (3/3 each; average 18.18 → 14.22 s). All finished naturally. The variants change all three numbers and the requested order.

The IQ4_NL prototype also reproduced 25/25 Japanese answers with fusion on. UD-Q4_K_M reproduced all five previously selected answers with fusion off and on (5/5 each). The Japanese checks used context 4,096. [Recorded answers and timings](answers/) · [Japanese questions and grading](../quality/)

To repeat the registry test, start the packaged server directly with a 16,384-token context. Set `$env:INKLING_PP_FUSE` to `'0'` or `'1'` before starting it. Run each mode in a fresh process.

```powershell
$env:HIP_VISIBLE_DEVICES = '0'
$env:GGML_HIP_IQ4_GATE4 = '1'
$env:GGML_HIP_DISABLE_SCALE_ADD_FUSION = '1'
$env:INKLING_PP_FUSE = '1'
& "$runtime/llama-server.exe" -m $model -c 16384 -np 1 -ngl 99 --device ROCm0 -b 2048 -ub 512 -t 16 -ctk f16 -ctv f16 -fa auto --jinja --offline --no-cache-prompt --cache-ram 0 --no-context-shift --host 127.0.0.1 --port 8080
```

In a second PowerShell terminal in this directory:

```powershell
$cases = Get-Content -Raw -Encoding UTF8 retrieval-cases.json | ConvertFrom-Json
foreach ($case in $cases) {
    $body = @{ messages=$case.messages; seed=42; temperature=0; max_tokens=1152; cache_prompt=$false; stream=$false } | ConvertTo-Json -Depth 6
    $reply = Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8080/v1/chat/completions' -ContentType 'application/json' -Body $body
    $answer = $reply.choices[0].message.content.Trim()
    if ($answer -ne ($case.expected -join ' ') -or $reply.choices[0].finish_reason -ne 'stop' -or $reply.timings.cache_n -ne 0) { throw "Failed: $($case.id)" }
    Write-Output "$($case.id)  $answer  $($reply.timings.prompt_ms) ms"
}
```
