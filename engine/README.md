# MMM C++ inference engine

Plan section 13. A small handwritten engine, the same approach Holden used for
LMM (the `nnet.h` file in his Motion Matching repo). MMM needs no attention, KV
cache, tokenizer or sampling, so this is a few hundred lines.

## The freeze rule

**Write this engine only after the PyTorch architecture is frozen**, since every
architecture change means changing the engine too. Until then these files hold
the plan and nothing else.

## What runs per frame (13.1)

- Linear in: one matrix multiply.
- 4 Mamba layers in recurrent mode: small matrix multiplies plus element wise
  math.
- Linear out: one matrix multiply.
- KNN search every 10 frames over the Matching Database.

Per character state is a conv buffer (last 4 inputs) plus an SSM state of
`d_inner x d_state` = 512 x 16 per layer, about 130 KB for 4 layers, so many
NPCs can run at once.

## Build plan (13.3)

1. **Export weights** — one float32 binary in a fixed order, from
   `mmm/export/weights_bin.py`.
2. **Engine** — load weights once at startup, preallocate every buffer, no
   allocation per frame; implement linear, RMSNorm, SiLU, softplus, conv step
   and SSM step.
3. **State per character** — own conv buffers and SSM states.
4. **KNN in C++** — brute force with Holden's bounding box acceleration, or
   nanoflann.
5. **Verify against PyTorch** — same 1000 frames through both, match within
   about 1e-5, *before any optimization*.
6. **Optimize** — `-O3 -march=native`, contiguous matrices, then int8
   quantization as an experiment.
7. **Connect** — pybind11 for GenoViewPython, or call directly from Holden's C++
   raylib demo.

`llama.cpp` and `ggml` show the same deployment approach: C/C++ CPU inference
with quantization, including Mamba models.

## Why it matters for the thesis

Honest microsecond timings comparable to LMM, cost measured for 1, 10 and 100
characters, and a path into Unreal or Unity.
