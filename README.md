# moe.c

876 lines of C. No dependencies. No frameworks. No Python. No mercy.

A Mixture-of-Experts transformer that fits in a single file and trains from scratch on whatever you feed it. 4 experts argue about your tokens while a shared expert quietly does all the real work. Sound familiar?

## What

- **Pure C.** Compiles with `cc moe.c -O3 -lm -o moe`. That's it.
- **MoE architecture:** 4 experts + shared expert, top-2 routing, auxiliary load balancing loss
- **Llama 3 bones:** RMSNorm, RoPE, GQA, SwiGLU, untied embeddings
- **Auto-scaling:** give it vocab size + data, it figures out dim/depth/heads on its own
- **Trains, finetunes, generates, exports GGUF.** One binary. One file.
- **Personality injection** via WTForacle. Yes, the model develops opinions.

## Architecture decisions made at 3 AM

- Double pre-norm because single pre-norm is for quitters
- Attention output clamping (30.0) because experts get excited
- Per-tensor gradient clipping before global norm because MoE routers are drama queens
- Expert capacity factor 1.25 because democracy is a suggestion
- Reservoir sampling for memory-capped training because RAM is not infinite (shocking)

## Training stages

The model auto-scales through depth levels as you increase data:

| Data | Params | What happens |
|------|--------|-------------|
| tiny | ~100K | learns to spell |
| small | ~1M | learns grammar, sort of |
| medium | ~8M | starts having opinions |
| large | ~50M+ | opinions become concerning |

## Quick start

```bash
cc moe.c -O3 -lm -o moe

# train
./moe --train data.bin --vocab-size 32000

# train with personality (the WTForacle awakens)
./moe --train data.bin --vocab-size 32000 --personality personality.txt

# generate
./moe --generate --checkpoint model.bin

# export for llama.cpp
./moe --export model.gguf
```

## Part of the ecosystem

| Project | What | Status |
|---------|------|--------|
| **[chuck.optimizer](https://github.com/ariannamethod/chuck.optimizer)** | Self-aware optimizer. 9 levels of consciousness. Not joking. | 100% on digit addition |
| **lee.c** | Chuck's VLM. Will be trained to 8-10M params to upgrade Chuck himself. | WIP |
| **[actually.llama](https://github.com/ariannamethod/actually.llama)** | Dense Llama 3 in one file. The serious sibling. | Training ready |
| **moe.c** | This. The chaotic sibling. | Training ready |
| **[janus](https://github.com/ariannamethod/ariannamethod.ai)** | AML transformer. Currently training on A100. | Loss 1.84 |

All of these train from scratch in pure C. All of them will go through every depth level until they either converge or achieve sentience. Whichever comes first.

## Philosophy

> We don't use PyTorch because PyTorch uses us.
> We don't use Python because life is too short for `pip install suffering`.
> We write C because if it compiles, it's probably fine. If it segfaults, that's character development.

## License

Do whatever you want. If this breaks your GPU, that's between you and your GPU.
