# Installing & Using Mony

Your tool is now a proper Python package. Here's everything you need to know.

## Quick Start (TL;DR)

```bash
cd /path/to/Mony
pip install -e .
mony "Your UI description" modern funky
```

---

## Installation Options

### Option 1: Local Development (Recommended)

Install in editable mode so changes take effect immediately:

```bash
cd /path/to/Mony
pip install -e .
```

Then use from anywhere:

```bash
mony --help
mony "Dashboard" modern funky conservative
```

### Option 2: Install from Git Repository

If you push to GitHub, others can install directly:

```bash
pip install git+https://github.com/Lukafin/mony.git
```

### Option 3: Direct Install from Source

```bash
cd /path/to/Mony
pip install .
```

### Option 4: From PyPI (After Publishing)

Once published to PyPI:

```bash
pip install mony
```

See **Publishing to PyPI** section below.

---

## Usage Examples

Once installed:

```bash
# Basic usage
mony "Cross-platform personal finance dashboard" modern funky conservative

# Dry-run mode (preview prompts without API calls)
mony "Meditation mobile onboarding flow" modern funky --dry-run

# Custom model and size
mony "AI writing assistant workspace" modern \
  --model google/gemini-2.5-flash-image-preview \
  --size 9:16

# With reference images
mony "Smart home control hub" modern \
  --reference ./reference.png \
  --reference https://example.com/inspiration.png

# Get help
mony --help
```

---

## Environment Setup

Create a `.env` file in your working directory with your OpenRouter API key:

```bash
echo "OPENROUTER_API_KEY=sk-or-..." > .env
```

Or set it directly in your shell:

```bash
export OPENROUTER_API_KEY=sk-or-...
mony "Your description" designer1 designer2
```

---

## How It Works

### Files Created

```
pyproject.toml   ← Modern Python packaging config (most important)
setup.py         ← Traditional setup script (fallback compatibility)
```

The key setting:

```toml
[project.scripts]
mony = "mony.cli:main"
```

This tells pip to create a `mony` command that calls the `main()` function from `mony/cli.py`.

### Your Code

Everything in the `mony/` directory is unchanged - your actual CLI logic is untouched.

---

## Publishing to PyPI

To make `pip install mony` work globally, publish to PyPI:

### Step 1: One-Time Setup

Create accounts:
- https://pypi.org/account/register/ (production)
- https://test.pypi.org/account/register/ (testing)

Install tools:

```bash
pip install build twine
```

Get an API token from https://pypi.org/manage/account/tokens/ and save it.

### Step 2: Update Version

Edit `pyproject.toml` and `setup.py` with new version:

```toml
version = "0.2.0"
```

Follow semantic versioning: `MAJOR.MINOR.PATCH`

### Step 3: Build Package

```bash
python -m build
```

Creates `dist/mony-0.2.0-py3-none-any.whl` and `dist/mony-0.2.0.tar.gz`.

### Step 4: Test Upload

Upload to TestPyPI first:

```bash
twine upload --repository testpypi dist/*
```

Test installation:

```bash
pip install --index-url https://test.pypi.org/simple/ mony
mony --help
```

### Step 5: Upload to Production

```bash
twine upload dist/*
```

Verify:

```bash
pip install --upgrade mony
mony --help
```

### Step 6: Create Git Release

```bash
git tag v0.2.0
git push origin v0.2.0
```

---

## Automated Publishing with GitHub Actions

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  push:
    tags:
      - 'v*'

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - run: pip install build twine
      - run: python -m build
      - run: twine check dist/*
      
      - name: Publish to PyPI
        run: twine upload dist/*
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
```

Then:

1. Go to GitHub repo → Settings → Secrets and variables → Actions
2. Create secret `PYPI_API_TOKEN` with your PyPI token
3. Push a tag: `git tag v0.2.0 && git push origin v0.2.0`
4. GitHub Actions automatically publishes to PyPI

---

## Troubleshooting

### Command not found after installation

```bash
pip install --force-reinstall -e .
```

### Python version error

Mony requires Python 3.10+. Check your version:

```bash
python --version
```

If needed, specify Python version:

```bash
python3.10 -m pip install -e .
```

### Module import errors

```bash
pip install --force-reinstall .
```

### "File already exists" when uploading

You can't overwrite existing versions on PyPI. Increment the version number and reupload.

### Installation from Git fails

Make sure you've pushed the packaging files:

```bash
git add pyproject.toml setup.py README.md mony/
git commit -m "Add packaging configuration"
git push
```

Then others can install with:

```bash
pip install git+https://github.com/Lukafin/mony.git
```

---

## Customization Before Publishing

Update these placeholders before publishing to PyPI:

**pyproject.toml** and **setup.py**:
- `"Your Name"` → your name
- `"your.email@example.com"` → your email
- `https://github.com/yourusername/mony` → your GitHub repo URL

---

## Summary

| Need | Command |
|------|---------|
| Use locally | `pip install -e .` |
| Share via GitHub | `pip install git+https://github.com/Lukafin/mony.git` |
| Publish to PyPI | `python -m build && twine upload dist/*` |
| Uninstall | `pip uninstall mony` |
| Check installation | `mony --help` |
