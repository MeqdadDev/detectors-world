# Releasing

Publishing to PyPI is automatic once a GitHub Release is published.

## Version numbers

The project follows [Semantic Versioning](https://semver.org/):

| Change | Example |
|---|---|
| Bug fixes | `1.0.0` to `1.0.1` |
| New features that keep existing code working | `1.0.1` to `1.1.0` |
| Changes that break existing code | `1.1.0` to `2.0.0` |

The version lives in one place: `__version__` in `src/detectors_world/__init__.py`.

## Steps

1. On a branch, update `__version__`, and add a section to `CHANGELOG.md`.
2. Open a pull request, and merge it once the tests pass.
3. Create the release, with a tag matching the version:

    ```bash
    gh release create v1.0.1 --target main --title "v1.0.1" --notes "See the changelog."
    ```

4. Watch the publish workflow:

    ```bash
    gh run watch
    ```

5. Install the new version in a fresh environment, and try it.

## What the workflows do

| Workflow | Runs on | Does |
|---|---|---|
| `tests.yml` | Pull requests and pushes to `main` | Runs the tests on Linux, Windows and macOS, and checks the package build |
| `publish.yml` | A published GitHub Release | Runs the tests, builds the package, checks the version against the tag, and uploads to PyPI |
| `docs.yml` | Pull requests and pushes to `main` | Builds this site, and publishes it from `main` |

Publishing uses PyPI Trusted Publishing, so no token is stored in the repository.
