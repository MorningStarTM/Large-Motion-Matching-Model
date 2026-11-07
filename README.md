# Large Motion Matching Model

**Mamba Motion Matching (MMM):** real time character animation that learns from
an editable motion database, so new clips work without retraining.

---

## Standing on Daniel Holden's work

This project would not exist in this form without [Daniel
Holden](https://theorangeduck.com/). Much of what makes it possible is his,
and it is used here with gratitude:

| | |
| --- | --- |
| [**GenoView / GenoViewPython**](https://github.com/orangeduck/GenoViewPython) | the Geno character, its bind pose, the deferred renderer and all eleven GLSL shaders. MIT. The skinned viewer here is a port of his render loop. |
| [**LaFAN1 resolved**](https://github.com/orangeduck/lafan1-resolved) | the mocap this project trains on — Ubisoft's LaFAN1 re-solved onto the Geno skeleton |
| [**Motion-Matching**](https://github.com/orangeduck/Motion-Matching) | the reference implementation of the algorithm Phase 0 reproduces, including the simulation bone that owns the character's position and facing |
| [**Spring-It-On: The Game Developer's Spring-Roll-Call**](https://theorangeduck.com/page/spring-roll-call) (2021) | the half-life spring damper behind the trajectory predictor and inertialization |
| **Learned Motion Matching**, SIGGRAPH 2020 | Holden, Kanoun, Perepichka, Popa — the paper this project responds to, and the baseline it is measured against |
| [**100STYLE retargeted**](https://github.com/orangeduck/100style-retarget) | the style dataset behind the held-out-style experiment |

[`NOTICE.md`](NOTICE.md) records precisely what is redistributed verbatim, what
is adapted, and what was written independently — stated file by file, because
for a thesis the distinction matters.

Also used: **inertialization** from David Bollo, *Inertialization:
High-Performance Animation Transitions in Gears of War* (GDC 2018), and the
**6D rotation representation** from Zhou et al., CVPR 2019.

---

## The idea

One small model with a **Mamba backbone** takes the player's joystick input and
outputs a full body pose every frame. The motion knowledge base stays *outside*
the weights: a new animation clip is appended to the database and is usable
immediately, with no retraining.

```
joystick  ─┐
prev pose ─┼─> linear in (256) ─> 4 Mamba layers ─> linear out ─> next pose
KNN top-K ─┘         (hidden state carries frame to frame)
```

No tokenizer, no autoencoder: the model reads and writes raw pose numbers. A
KNN search over the editable database runs every 10 frames and supplies the top
K candidates as context; the model runs every frame.

| Phase | | |
| --- | --- | --- |
| 0 | Classic KNN motion matching | no learning — this is the teacher |
| 1 | Pre-train | supervised next pose prediction on raw mocap |
| 2 | Distill from KNN | SFT plus DAgger |
| 3 | RL polish | PPO or direct gradient, KNN in the loop |

---

## Getting started

Python 3.10 or newer.

```
pip install -r requirements.txt
```

**The mocap is not in this repository.** Download [LaFAN1
resolved](https://github.com/orangeduck/lafan1-resolved) and put the BVH files
in `data/bvh/bvh/`, or point `--bvh-dir` wherever you keep them.

**The Geno character is not here either**, because `*.bin` and `*.bvh` are
git-ignored. Copy `Geno.bin` and `Geno_bind.bvh` from a
[GenoViewPython](https://github.com/orangeduck/GenoViewPython) checkout into
`mmm/viewer/geno/resources/`. Without them the wireframe viewer still runs; the
skinned one does not.

Recover the obstacle layout once, so the simulators have something to place:

```
python scripts/recover_obstacles.py --bvh-dir data/bvh/bvh --out data/obstacles/lafan1_studio.json
```

---

# The simulators

There are two, and either can be driven by the **KNN teacher** (classic motion
matching, replaying real mocap) or by a **trained model** (generating every
frame). Four combinations.

Write each command on **one line**. PowerShell does not accept `\` as a line
continuation — a command split across lines runs only its first fragment and
then errors.

## Which one to use

**Wireframe** — a stick figure, plus the retrieval candidates as grey ghosts,
the predicted trajectory, and foot contact markers. Use it to see *what the
search is doing*.

**Geno** — Holden's skinned character with a shadow and ambient occlusion. Use
it to judge *whether the motion looks right*. Foot sliding and ground
penetration are nearly invisible on a stick figure and obvious on a shaded
character standing on a floor, which is the argument GenoView's own README
makes and the reason the renderer is worth having.

Neither replaces the other.

## Without a model — the KNN teacher

```
python scripts/run_obstacle_teacher.py --studio
```

```
python scripts/run_geno_viewer.py --studio
```

`--studio` puts the 31 obstacle surfaces recovered from the mocap into the
scene. Leave it off for an empty floor.

Startup takes about 20 seconds: the teacher needs the Matching Database, which
means loading eight clips and computing 27 features for 64,000 frames.

The teacher *can* see obstacles. Place cubes with the number keys and it
retrieves frames where the actor faced a comparable box — measured, that cuts
foot penetration into the cube from 40.8 cm to 15.7 cm.

## With a trained model

```
python scripts/run_geno_viewer.py --studio --model runs/phase_1_v1/published.safetensors --model-manifest runs/phase_d/manifest.json
```

```
python scripts/run_obstacle_teacher.py --studio --model runs/phase_1_v1/published.safetensors --model-manifest runs/phase_d/manifest.json
```

Nothing is retrieved and nothing is played back. The model is handed the
joystick and the pose it produced last frame, and generates the next one, so
errors compound — which is the point. It is the only way to find out whether
the motion holds up over minutes rather than the 64-frame windows a validation
loss measures.

Startup is about 3 seconds. A model searches nothing, so the Matching Database
is never built; one clip is loaded for the skeleton and a seed pose.

**`--model-manifest` matters.** A `.safetensors` file holds weights and nothing
else — not how wide the model is, how many layers it has, or whether retrieval
is in its input — so it cannot be rebuilt without the `manifest.json` published
beside it.

**The Phase 1 model has no obstacle input.** Boxes are drawn and the character
runs straight through them. Both viewers say so at startup and on the HUD. That
is not a failure: obstacle features belong to Phase 2.

## Controls

Both simulators:

| input | action |
| --- | --- |
| `W` `A` `S` `D`, or the left stick | move, relative to the camera |
| mouse, or the right stick | turn the camera |
| right mouse button, or `L1` | aim — face the camera rather than the direction of travel |
| scroll wheel | zoom |
| `space` | pause |
| `M` | release the mouse |
| `Esc` | quit |

Wireframe only:

| key | action |
| --- | --- |
| `1`–`5` | place a cube ahead at 20.9, 32.8, 47.3, 66.1 or 93.5 cm |
| `6` | place a 150 cm block, which the data has nothing for |
| `L` | load the full recovered studio layout |
| `backspace` / `0` | remove the last cube / clear all |
| `C` `T` `F` | candidates / trajectory / contact markers |
| `R` | reset the character |

Those heights are not round numbers — they are five of the surface heights
actually recovered from the mocap, so a cube placed with those keys is a height
the database has motion for. `6` exists to show the failure mode honestly:
there is no 150 cm vaulting in the data, so the search returns the nearest
thing it has and the result is wrong.

Geno only:

| key | action |
| --- | --- |
| `K` | overlay the skeleton |
| `B` | show or hide the boxes |

## Options worth knowing

**Choosing clips.** With the teacher this sets what can be retrieved; with a
model it only chooses where the seed pose comes from.

```
python scripts/run_geno_viewer.py --clips walk1_subject1 run1_subject2 --max-frames 4000
```

**Who controls the path** — `--root-source`, model only:

| value | facing | translation | what it shows |
| --- | --- | --- | --- |
| `hybrid` *(default)* | spring damper | the model | steers properly, feet keep up with the body |
| `animation` | the model | the model | the honest test of whether the model learned to turn |
| `simulation` | spring damper | spring damper | exactly controllable, but the feet slide |

**Where the model starts** — `--seed-frame`, default 1500. The model predicts
the *next* pose, so frame zero has to be given a real one. Not frame 0: LaFAN1
clips open with 104 to 475 frames of calibration T-pose.

**Speed.** The model costs 6.14 ms a frame on CPU, a ceiling of about 163 fps,
so 60 fps holds comfortably. If it does not, try `--device cuda`, or drop the
resolution with `--width 960 --height 540`.

**Headless**, for a screenshot without watching a window:

```
python scripts/run_geno_viewer.py --studio --frames 200 --walk --screenshot shot.png
```

```
python scripts/run_obstacle_teacher.py --studio --viewer-frames 200 --viewer-walk --viewer-screenshot shot.png
```

`--walk` matters: with no input a teacher sits on row 0, the calibration
T-pose, which looks exactly like broken skinning.

## Reading the HUD

Under a model you will see:

```
row -1   clip obstacles1_subject1   frame -1
searched False   jumped False
```

`row -1` and `frame -1` are correct, not a bug — they mean the pose was
generated rather than read from a clip, and nothing was retrieved. Under the
teacher those fields carry the real row, its clip, and whether the search
jumped this frame.

## When something goes wrong

| symptom | cause |
| --- | --- |
| `FileNotFoundError` on `Geno.bin` | the character is git-ignored; copy it from GenoViewPython into `mmm/viewer/geno/resources/` |
| `no clips matching ... in data/bvh/bvh` | `data/` is git-ignored; point `--bvh-dir` at your LaFAN1 copy |
| `--studio` places nothing | run `scripts/recover_obstacles.py` first |
| `ValueError: ... trained with retrieval in its input` | a Phase 2 checkpoint; it needs candidate poses this controller does not supply |
| the character stands in a T-pose | the calibration pose at the start of every clip, not a rendering bug — give it input, or use `--walk` |

---

## Training

```
python scripts/train_phase1.py --dataset data/bvh/bvh --pattern "*" --epochs 50 --batch-size 16 --lr-scale linear --trim-calibration --elevated-contacts --out runs/phase_e
```

Two data flags, both off by default:

- `--trim-calibration` drops the leading T-pose from every clip. Those frames
  are searchable database rows, so a query landing near one makes the character
  snap into a T-pose. About 15,900 frames across the 77 clips.
- `--elevated-contacts` labels a foot planted at any height. The default
  detector cannot see a foot standing on a prop, so on the obstacle clips it
  marks 40 to 52 per cent of frames "not in contact" while the actor is on a
  box. Level-ground labels are unchanged.

`notebooks/phase1_report.ipynb` reads the results and compares runs side by
side.

---

## Repository layout

```
mmm/
  data/       BVH, skeleton, rotations, FK, Matching/Animation databases
  teacher/    Phase 0: classic KNN motion matching
  obstacles/  recovering the physical props from the mocap, and searching with them
  models/     linear in -> 4 Mamba layers -> linear out, plus the runtime controller
  training/   Phase 1 pretrain, Phase 2 distill, Phase 3 RL
  envs/       kinematic world and synthetic joystick
  viewer/     the wireframe viewer, and the Geno renderer
  eval/       foot sliding, jitter, path error, timing
  export/     ONNX, and the weight blob for the C++ engine
engine/       C++ inference engine — written only after the architecture freezes
scripts/      thin CLI entry points
```

Every module carries a docstring stating what it implements and why.

---

## Licences

The code here is this project's. The Geno character and the shaders are Daniel
Holden's, MIT, redistributed under the terms in
[`licenses/GenoViewPython-LICENSE`](licenses/GenoViewPython-LICENSE).

Motion data is not redistributed. See [`data/README.md`](data/README.md) for the
datasets and their licences — most are noncommercial research licences: fine for
research, a product needs its own data.

---

`motion-matching` `character-animation` `mamba` `state-space-model`
`reinforcement-learning` `game-ai` `motion-capture` `pytorch`
