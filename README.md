# moe.c

Grok-1 Mixture-of-Experts transformer. One C file. Trains from scratch.

**by [Arianna Method](https://github.com/ariannamethod)**

---

```
cc moe.c -O3 -lm -lpthread -o moe && ./moe --depth 4
```

---

## what

~2000 lines of C. 4 experts + shared expert, top-2 routing, auxiliary load balancing loss. Full Llama 3 backbone: RMSNorm, RoPE, GQA, SwiGLU, double pre-norm, untied embeddings. Hand-written analytical backward passes through everything — including the router and aux_loss.

Trains its own BPE tokenizer. Downloads data from HuggingFace. Finetunes personality with LoRA. Exports GGUF. Chats with you. Optional CUDA/cuBLAS — 480 tok/s on A100.

no Python. no PyTorch. no dependencies.

## how

```bash
cc moe.c -O3 -lm -lpthread -o moe

./moe --depth 2    # ~1.5M params — fast demo
./moe --depth 4    # ~8M params   — starts having opinions
./moe --depth 8    # ~58M params  — opinions become concerning
```

### with CUDA (optional)

```bash
nvcc -c ariannamethod_cuda.cu -o ariannamethod_cuda.o -O3
cc moe.c ariannamethod_cuda.o -O3 -lm -lpthread -DUSE_CUDA -lcublas -lcudart \
   -L/usr/local/cuda/lib64 -o moe_cuda

./moe_cuda --depth 8    # 480 tok/s on A100
```

`--depth` is the only knob. dim, heads, kv_heads, hidden_dim, experts — all auto-scale.

## what happens when you run it

1. downloads training data from HuggingFace (FineWeb-Edu, paginated)
2. trains a byte-level BPE tokenizer from scratch (cached in `moe_bpe.cache`)
3. builds a Grok-1 MoE transformer
4. trains it with hand-written analytical backward passes
5. finetunes personality with LoRA (if `personality_sft.txt` present)
6. exports `moe.gguf` (DOE compatible)
7. drops you into interactive chat

## the architecture

```
Token Embedding (untied)
  ┌──────────────────────────────────────────┐
  │  RMSNorm (pre-attention)                 │
  │  RMSNorm (double pre-norm)               │
  │  RoPE                                    │
  │  Grouped Query Attention (GQA)           │ x depth
  │  Residual                                │
  │  RMSNorm (pre-FFN)                       │
  │  RMSNorm (double pre-norm)               │
  │  Router → top-2 experts + shared expert  │
  │  SwiGLU FFN per expert (gate · up · down)│
  │  Weighted sum + residual                 │
  └──────────────────────────────────────────┘
RMSNorm → LM Head → Softmax → Token
```

4 experts compete per layer. Router picks top-2 via softmax with temperature. Shared expert runs on every token. Auxiliary loss prevents expert collapse (the thing that killed every MoE before this one).

Every component has a hand-written forward **and** backward pass:
- Router backward — through softmax temperature, through expert selection
- Aux loss backward — gradient through router to prevent dead experts
- Expert SwiGLU backward — through gating, through both projections, per expert
- GQA backward — through grouped KV heads, through attention, through softmax
- RoPE backward — through rotation matrices
- Double pre-norm backward — through both normalization layers

## SFT (supervised fine-tuning)

Built-in chat SFT with loss masking. Special tokens `<user>`, `<assistant>`, `<end>` added automatically.

```bash
# train with SFT — loss computed only on assistant tokens
./moe --depth 8 --sft personality_sft.txt
```

## LoRA personality finetune

Full finetune kills coherence. LoRA freezes the base and trains only 0.79% of parameters.

```bash
# LoRA SFT — trains adapters, merges into base, saves
./moe --depth 8 --lora-sft personality_sft.txt

# load LoRA adapters before chat
./moe --depth 8 --lora adapters.bin --chat
```

- rank=16 on wq/wk/wv/wo (all attention projections)
- analytical backward through LoRA
- separate Adam optimizer for adapter params
- merge into base for zero-overhead inference

## training results

| Version | Params | Data | Steps | Loss | Speed | Notes |
|---------|--------|------|-------|------|-------|-------|
| v8 (depth 8) | 57.87M | 11.2MB | 50K | **1.76** | 480 tok/s | CUDA, personality 1000 steps |
| v9 (depth 8) | 57.87M | 300MB | 50K | training... | 480 tok/s | base only, SFT after |

## what we fixed (the hard way)

6 critical bugs found during a 3-day training marathon:

1. **Aux loss backward never existed** — forward computed aux_loss, backward was zero. Experts died.
2. **Double gradient clipping** — per-tensor + global. With 50+ tensors, everything crushed to 1/70th.
3. **attn_clamp tanh derivative missing** — attention gradients attenuated 3x.
4. **LR 3e-4 too high for MoE** — dropped to 1e-4.
5. **beta2=0.999 too slow** — 0.95 tracks MoE variance better.
6. **Router softmax no temperature** — top-2 polarized. Added temp=2.0.

Plus: gradient accumulation (batch=4), BPE 2000→4000, warmup 100→500.

## family

- **[actually.llama](https://github.com/ariannamethod/actually.llama)** — the single-expert sibling (Llama 3, same C lineage)
- **[AML](https://github.com/ariannamethod/ariannamethod.ai)** — the language that trains Janus
- **[DOE](https://github.com/ariannamethod/doe)** — inference engine that runs our GGUF exports

## credits

**Oleg** and **Claude**. Many sessions. Several GPUs harmed. Several experts killed and resurrected.

---

<div align="center">

`cc moe.c -O3 -lm -lpthread -o moe && ./moe --depth 4`

*one file. four experts. one shared expert doing all the work.*

</div>
