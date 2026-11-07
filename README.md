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

## Running the simulators

Two simulators, each drivable either by the **KNN teacher** (classic motion
matching, playing back real mocap) or by the **trained model** (generating
every frame). Four combinations, one command each.

**Wireframe** — stick figure, retrieval candidates, trajectory and contacts.
Use it to see what the search is doing.

```
python scripts/run_obstacle_teacher.py --studio
python scripts/run_obstacle_teacher.py --studio --model runs/phase_1_v1/published.safetensors --model-manifest runs/phase_d/manifest.json
```

**Geno** — skinned character with shadows and ambient occlusion. Use it to
judge whether the motion looks right: foot sliding and ground penetration are
nearly invisible on a stick figure and obvious on a shaded character.

```
python scripts/run_geno_viewer.py --studio
python scripts/run_geno_viewer.py --studio --model runs/phase_1_v1/published.safetensors --model-manifest runs/phase_d/manifest.json
```

WASD and the mouse to drive, `space` to pause, `M` to release the cursor. In
the wireframe viewer `1`-`5` place obstacles at heights recovered from the
mocap. `--studio` loads the 31 recovered surfaces; leave it off for a bare
floor.

Full reference, including every key, `--root-source`, headless rendering and
troubleshooting: **[docs/simulators.md](docs/simulators.md)**.

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

**Phase 0 complete.** BVH loading, the 27-dimensional matching features, KNN
search, the spring damper, inertialization and two interactive viewers. Plus
an additive extension that recovers the physical obstacles LaFAN1 was captured
with and feeds them to the search: 31 surfaces recovered, agreeing across
clips to a median of 1.42 cm, which cuts foot penetration into a box from
40.8 cm to 15.7 cm.

**Phase 1 trained.** A 1.8M parameter Mamba policy, 50 epochs on all 77 clips.
It tracks requested speed well and beats every degenerate baseline, but its
joint-angle accuracy is weak — see the notebooks below. It can be driven in
either viewer with `--model`.

**Phase 2 (distill from KNN) is next**, and is where retrieval becomes
load-bearing in the model's input.

## Documentation

- [docs/simulators.md](docs/simulators.md) — running both simulators, with
  and without a trained model
- [docs/research-plan.md](docs/research-plan.md) — the full plan, cited by
  section number throughout the code
- [notebooks/](notebooks/) — the Phase 1 and Phase D training reports
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
