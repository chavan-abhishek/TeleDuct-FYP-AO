# The Decisive Piece of Evidence

**SGLang's own official documentation states directly:** quantized MoE models may encounter inference issues due to kernel limitations, such as lack of support for `mlp.gate` layer quantization.

- This is almost exactly the bug class that caused today's **Qwen1.5-MoE deadlock**.

- **This wasn't a coincidence.** It is a documented, acknowledged limitation of SGLang when serving third-party post-hoc quantized MoE checkpoints.

### What this changes
- The risk isn't really **which model** we choose.
- The real question is:
**Who quantized it, and how?**

---

# Candidates Rejected

| Model | Reason for Rejection |
|---|---|
| **Kimi K3** | `2.8T` parameters, `896` experts. Completely irrelevant for a single 24 GB GPU at any practical quantization level. **Reject outright.** |
| **DeepSeek-V3.2 family** | Official launch commands require `--tp-size 8` → 8-GPU tensor parallelism. Not deployable on a single GPU regardless of quantization. **Reject.** |
| **Gemma 4 26B MoE** | ~52 GB raw BF16 weights and ~60 GB with framework overhead. Even aggressive quantization doesn't comfortably leave enough room for KV cache on 24 GB. **Reject.** |
| **DeepSeek-V2-Lite** | Earlier research explicitly showed ~80 GB required with SGLang vs ~40 GB with vLLM. Known SGLang-specific memory inefficiency for this architecture. **Reject.** |
| **Nex-N2-mini** | Technically fits at ~20 GB Q4, but requires a **custom SGLang fork** and its benchmarks are self-reported. The source itself warns to *"budget an evening, not ten minutes."* Given today's debugging experience, this is unacceptable risk. **Reject.** |
| **Qwen3.6-27B** | Dense model, **not MoE**. Fails the project's core requirement of genuine sparse expert routing for telemetry purposes. **Reject for this project.** |
| **Qwen1.5-MoE / Qwen3-30B-A3B** | Already tested directly today. Qwen1.5-MoE hit the documented SGLang MoE-kernel deadlock; Qwen3-30B-A3B hit the total-parameter VRAM ceiling. **Eliminated by direct evidence.** |

---

# Why `gpt-oss-20b` Wins

This isn't a "safe default." It is the model that best satisfies the actual requirements.

| Requirement | `gpt-oss-20b` Status |
|---|---|
| **Genuine MoE + real routing** | ✅ 32 experts, top-4 gating — confirmed standard sparse architecture |
| **Fits within 24 GB VRAM** | ✅ 13.05 GB weight usage, leaving 10.18 GB free after loading |
| **Native / first-party quantization** | ✅ MXFP4 baked in by OpenAI at release — **not third-party post-hoc quantization** |
| **Avoids the documented SGLang kernel risk** | ✅ No dependence on the problematic third-party MoE quantization path |
| **Official SGLang compatibility** | ✅ `GptOssForCausalLM` is a distinct, purpose-built architecture class |
| **Actually verified on our stack** | ✅ Real generation confirmed twice via `/generate` and `/v1/completions` |
| **CUDA graphs** | ✅ Successfully captured |
| **Unknowns remaining** | **Minimal** — this is the only candidate with essentially no unresolved deployment blockers |

---

# The One-Line Answer

> ## **Stick with `gpt-oss-20b`.**

It isn't the *"safe default because we're tired of debugging."*

It is the model that **structurally avoids the exact documented SGLang failure mode** — third-party MoE quantization hitting kernel limitations — that cost us the entire day.

Every serious alternative falls into at least one of these categories:

- ❌ Reintroduces the same SGLang quantization risk
- ❌ Doesn't satisfy the genuine MoE-routing requirement
- ❌ Doesn't fit within 24 GB VRAM
- ❌ Requires multiple GPUs
- ❌ Requires an unofficial/custom SGLang fork
- ❌ Has already failed in direct testing

### Final Decision

**`gpt-oss-20b` is the correct choice on technical merit — not merely the path of least resistance.**