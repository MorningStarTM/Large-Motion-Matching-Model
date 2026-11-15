"""
Config-driven entry point for Phase 1 GPT-backbone training.

Scaling `mmm.models.gpt.backbone.GptBackbone` means changing a handful of
shape numbers together -- `d_model`, `n_layers`, `n_heads`, `window` -- plus
whatever optimisation settings that size calls for. A long `--flag` list
makes that easy to get wrong by hand and hard to diff between runs. This
script is a thin front end: it reads one YAML file, turns its keys into the
equivalent `scripts/train_phase1.py` flags, forces `--backbone gpt`, and
calls straight into that script's `main()`. There is no separate GPT
training loop here -- `PreTrainer` (see `mmm/training/pretrainer.py`) already
does not care which backbone it holds, since every backbone shares the same
`forward`/`step`/`initial_state` interface.

Usage:
    python scripts/gpt_trainer.py --config configs/gpt/config_10m.yaml --dataset data/bvh

    # anything after --config overrides the same key in the YAML
    python scripts/gpt_trainer.py --config configs/gpt/config_2m.yaml --dataset data/bvh --epochs 5

A config is a flat mapping of `train_phase1.py`'s long option names
(underscores or dashes, either works) to values, e.g. `d_model: 384`.
`--dataset` is deliberately left out of the example configs, since it is
environment specific (a local path vs. a Kaggle mount) -- pass it on the
command line.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml

import train_phase1


def _config_to_argv(config: dict) -> list[str]:
    argv = []
    for key, value in config.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        else:
            argv += [flag, str(value)]
    return argv


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="e.g. configs/gpt/config_10m.yaml")
    return parser.parse_known_args(argv)


def main(argv=None) -> int:
    args, overrides = parse_args(argv)

    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    config.setdefault(
        "run_name", os.path.splitext(os.path.basename(args.config))[0]
    )
    config["backbone"] = "gpt"

    # argparse keeps the last value when a flag repeats, so overrides placed
    # after the config-derived flags win over the YAML, matching the
    # docstring's promise.
    forwarded = _config_to_argv(config) + list(overrides)
    print(f"gpt_trainer: {args.config} -> train_phase1.py {' '.join(forwarded)}")
    return train_phase1.main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
