# Runs

Gitignored. Checkpoints, logs, teacher rollouts and DAgger datasets.

Suggested layout, one directory per run:

```
runs/<phase>-<date>-<short-name>/
  config.yaml          the resolved config, as actually run
  manifest.json        git commit, database manifest hash, split identity,
                       parent checkpoint
  checkpoints/
  logs/
  rollouts/            recorded poses and videos for visual inspection
```

The manifest is what makes a number in the thesis traceable back to the run that
produced it.
