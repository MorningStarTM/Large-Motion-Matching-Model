"""
The MMM model: linear in, a backbone, linear out. No tokenizer.

Plan section 4.3. Inputs go through a trainable linear input layer to width
256, then the backbone (a hidden state updated one frame at a time), then a
trainable linear output layer to pose size, producing the next pose. There is
no autoencoder and no tokenizer: the model reads and writes raw pose numbers.
`policy.py`, `io_layers.py`, `heads.py` and `runtime.py` are backbone agnostic
-- `MmmPolicy` holds whichever backbone it is given and neither trains nor
runs any differently for it.

The backbone is the one part of the design actively under comparison, each in
its own package, same `forward(x, state)` / `step(x, state)` /
`initial_state(batch)` interface so only the config changes between runs:
  mamba/    MambaBackbone   -- trained so far; constant cost per frame in
                               recurrent mode, unlike attention, while still
                               training in parallel over long sequences
                               (plan section 4.5)
  gpt/      GptBackbone     -- a causal transformer with a KV cache, to
                               implement; tests whether attention over a
                               bounded window captures this problem's
                               structure better than a recurrent state does
  xlstm/    XlstmBackbone   -- Beck et al. 2024's gated recurrent cell, to
                               implement; keeps Mamba's constant per-frame
                               cost while testing a different recurrence
`baselines/` holds `GruBackbone` and the transformer ablation: a narrower,
already-decided question (is Mamba's extra machinery earning its keep over a
plain strong RNN, and what does attention's history-dependent cost actually
cost), not an open architecture candidate the way the three above are.

Where LMM's three networks went (plan section 4.2), for orientation only, since
LMM is background and baseline rather than our design:
  Projector      -> a real KNN search over the editable KB, plus the model
                    learning to blend the top candidates
  Stepper        -> the backbone's hidden state, whose recurrence (or, for the
                    GPT candidate, attention) is the dynamics
  Decompressor   -> the linear output head
"""
