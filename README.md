# Large Motion Matching Model

**Mamba Motion Matching (MMM):** real time character animation that learns from
an editable motion database, so new clips work without retraining.

A tiny state space model for real time character animation in games, with an
editable motion database.

---

## The idea

One tiny model with a **Mamba backbone** takes the player's joystick input and
outputs a full body pose every frame. The motion knowledge base stays *outside*
the weights: a new animation clip is appended to the database and is usable
immediately, with no retraining.

Learned Motion Matching bakes the animation data into network weights, so every
new clip means hours of retraining — a gap the LMM paper names itself. MMM keeps
retrieval explicit and learns how to use whatever is retrieved.

```
joystick ─┐
prev pose ─┼─> linear in (256) ─> 4 Mamba layers ─> linear out ─> next pose
KNN top-K ─┘        (hidden state carries frame to frame)
```

No tokenizer, no autoencoder: the model reads and writes raw pose numbers. A KNN
search over the editable database runs every 10 frames and supplies the top K
candidates as context; the model runs every frame.

## Training

| Phase | | Paradigm |
| --- | --- | --- |
| 0 | Classic KNN motion matching | no learning — this is the teacher |
| 1 | Pretrain | supervised next pose prediction on raw mocap |
| 2 | Distill from KNN | SFT plus DAgger |
| 3 | RL polish | PPO or direct gradient, KNN in the loop |

## Main claim

Hold 10 of the 100STYLE styles out of all training, add them to the knowledge
base only, and the character performs them. `scripts/add_clip_to_kb.py` is that
claim in executable form.

## Repository layout

```
mmm/
  data/       BVH, skeleton, rotations, FK, Matching/Animation databases
  teacher/    Phase 0: classic KNN motion matching
  models/     linear in -> 4 Mamba layers -> linear out, plus GRU/transformer baselines
  training/   Phase 1 pretrain, Phase 2 distill, Phase 3 RL
  envs/       kinematic world and synthetic joystick
  viewer/     GenoViewPython / raylib integration
  eval/       foot sliding, jitter, path error, timing
  export/     ONNX, and the weight blob for the C++ engine
engine/       C++ inference engine — written only after the architecture freezes
configs/      one config per phase and per experiment
scripts/      thin CLI entry points
tests/        rotation/FK round trips, feature layout, recurrent-equals-parallel
docs/         the research plan, and decision records
```

Every module carries a docstring stating what it implements and which section of
the plan it comes from.

## Status

Scaffold. Nothing implemented yet. Next milestone is **Phase 0**: BVH loading,
the 27-dimensional matching features, NumPy KNN search, then wiring it into the
viewer with joystick, trajectory and inertialization.

## Documentation

- [docs/research-plan.md](docs/research-plan.md) — the full plan, cited by
  section number throughout the code
- [docs/decisions/](docs/decisions/) — decision records
- [engine/README.md](engine/README.md) — the C++ engine and its freeze rule

## Requirements

Python 3.10 or newer (`mamba-ssm` and current torch do not support 3.8).

## Data and licences

Motion data is not in this repository. See [data/README.md](data/README.md) for
the datasets used and their licences. Most are noncommercial research licences:
fine for research, a product needs its own data.

## Topics

`motion-matching` `character-animation` `mamba` `state-space-model`
`reinforcement-learning` `game-ai` `motion-capture` `pytorch`
