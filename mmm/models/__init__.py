"""
The MMM model: linear in, four Mamba layers, linear out. No tokenizer.

Plan section 4.3. Inputs go through a trainable linear input layer to width 256,
then 4 Mamba layers with a hidden state, then a trainable linear output layer to
pose size, producing the next pose. There is no autoencoder and no tokenizer:
the model reads and writes raw pose numbers.

Where LMM's three networks went (plan section 4.2), for orientation only, since
LMM is background and baseline rather than our design:
  Projector      -> a real KNN search over the editable KB, plus the model
                    learning to blend the top candidates
  Stepper        -> the Mamba hidden state, whose recurrence is the dynamics
  Decompressor   -> the linear output head

Why Mamba (plan section 4.5): constant cost per frame in recurrent mode, unlike
attention whose cost grows with history, while still training in parallel over
long sequences. Because Mamba at runtime behaves like a strong RNN, a small GRU
is a required baseline, in baselines/.
"""
