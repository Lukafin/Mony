# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

Mony is a Python CLI + Streamlit web UI for generating UI concept art by combining project briefs with designer persona prompts via the OpenRouter image API. See `README.md` for full usage and `INSTALL.md` for packaging details.

### Services

| Service | Command | Notes |
|---|---|---|
| CLI (dry-run) | `python3 -m mony "description" modern --dry-run` | Works without API keys |
| CLI (generate) | `python3 -m mony "description" modern funky` | Requires `OPENROUTER_API_KEY` |
| Streamlit UI | `python3 -m streamlit run mony/ui.py --server.headless true --server.port 8501` | Runs on port 8501 |

### Lint and test

- **Lint:** `python3 -m ruff check .` — 28 pre-existing lint issues (import sorting + line length); these are in the original code.
- **Test:** `python3 -m pytest --override-ini="addopts=-v"` — no test files exist yet (`tests/` directory is absent). The `pyproject.toml` `addopts` contains `--cov-report=term-only` which is invalid for current pytest-cov; override with `--override-ini` as shown.

### Gotchas

- `mony` console script may not be on `$PATH` after `pip install -e .`; prefer `python3 -m mony` to invoke the CLI.
- The `python` command is not available on the VM; always use `python3`.
- `OPENROUTER_API_KEY` is required for actual image generation (both CLI and Streamlit UI). Use `--dry-run` on the CLI to test prompt assembly without an API key.
- `PERPLEXITY_API_KEY` is only needed for persona research features.
- Generated images are saved to `generatedDesigns/` in the working directory.
