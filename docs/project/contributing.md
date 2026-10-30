# Contributing

Pull requests are welcome. For major changes, please [open an issue](https://github.com/MeqdadDev/detectors-world/issues) first to discuss what you would like to change.

## Set up

```bash
git clone https://github.com/MeqdadDev/detectors-world.git
cd detectors-world
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -e ".[test]" --group docs
```

On Linux and macOS, activate the environment with `source .venv/bin/activate`.

## Run the tests

```bash
pytest
```

The tests download the models and a few sample images the first time they run.

## Work on the documentation

Preview the site as you edit:

```bash
mkdocs serve
```

Before opening a pull request, check that it builds without warnings:

```bash
mkdocs build --strict
```

Some content comes from other files, so edit the source rather than the page:

| Content | Source |
|---|---|
| Example code | `examples/*.py` |
| API reference | Docstrings in `src/detectors_world/` |
| Changelog | `CHANGELOG.md` |
| Landmark maps and tables | `tools/generate_landmark_maps.py` |

To regenerate the maps, pass a photo with one clear frontal face:

```bash
python tools/generate_landmark_maps.py path/to/photo.png
```

## Add an example

1. Add a complete program to `examples/`.
2. Test it on your own camera or files.
3. Add a page in `docs/examples/` that embeds it, and list it in `docs/examples/index.md` and `mkdocs.yml`.
