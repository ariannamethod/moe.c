# moe.c

C. No dependencies. No frameworks. No Python. No mercy.

Grok inspired Mixture-of-Experts transformer that fits in a single file and trains from scratch on whatever you feed it. 4 experts argue about your tokens while a shared expert quietly does all the real work. Sound familiar?

## what

- **Pure C.** Compiles with `cc moe.c -O3 -lm -o moe`. That's it.
- **MoE architecture:** 4 experts + shared expert, top-2 routing, auxiliary load balancing loss
- **Llama 3 bones:** RMSNorm, RoPE, GQA, SwiGLU, untied embeddings
- **Auto-scaling:** give it vocab size + data, it figures out dim/depth/heads on its own
- **Trains, finetunes, generates, exports GGUF.** One binary. One file.
- **Personality injection** via WTForacle. Yes, the model develops opinions.

## why

- Double pre-norm because single pre-norm is for quitters
- Attention output clamping (30.0) because experts get excited
- Per-tensor gradient clipping before global norm because MoE routers are drama queens
- Expert capacity factor 1.25 because democracy is a suggestion
- Reservoir sampling for memory-capped training because RAM is not infinite (shocking)

## training stages

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
  
## Philosophy

> We don't use PyTorch because PyTorch uses us.
> We don't use Python because life is too short for `pip install suffering`.
> We write C because if it compiles, it's probably fine. If it segfaults, that's character development.

## License

GPLv3. 
