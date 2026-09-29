# Environment snapshot

The repository records the complete current Conda environment named `mamballm`, not
only the packages imported by the compact InvaRNA package.

- `environment.yml` contains every Conda and pip package with a version pin but omits
  Conda build strings. This is the preferred review installation file.
- `environment.lock.yml` contains the same complete package set and retains Conda
  build strings from the current Linux x86-64 machine. It is more exact and less
  portable.
- `pyproject.toml` remains the minimal package metadata and is not a replacement for
  either full environment snapshot.

Create the version-pinned environment with:

```bash
conda env create -f environment.yml
conda activate mamballm
python -m pip install -e . --no-deps
```

To request the current build-resolved snapshot instead:

```bash
conda env create -f environment.lock.yml
conda activate mamballm
python -m pip install -e . --no-deps
```

Both YAML files were exported on 2026-09-20 and have no local `prefix`, editable
installation, username, or absolute project path. The full environment contains
packages used by external-model baselines and other tools in addition to the core
InvaRNA workflow because it is an exact inventory of the requested `mamballm`
environment.

This snapshot must not be described as the original training environment. A
historical production lockfile was not recovered, so equivalence of these versions to
the original pretraining, SFT, and baseline runs remains
`NEEDS_AUTHOR_CONFIRMATION`. Frozen checkpoints and saved manuscript outputs are the
primary reproduction path for expensive experiments.
