# Mony

Visualize different UI directions by combining project briefs with designer
personas and generating concept images through the OpenRouter image API.

## Features

- Describe the UI you want to explore via a CLI argument.
- Mix and match designer prompt templates stored as markdown files.
- Research the latest trendy design directions to inspire and create new personas.
- Generate concept images for each selected designer using OpenRouter.
- Preview prompts without generating images via `--dry-run` mode.
- Supply one or more reference images or URLs to steer the generation results.

## Installation

Requires Python 3.9+

Install locally:

```bash
pip install -e .
```

Or from GitHub:

```bash
pip install git+https://github.com/Lukafin/mony.git
```

Create a `.env` file in your working directory with your OpenRouter key:

```bash
echo "OPENROUTER_API_KEY=sk-or-..." > .env
```

## Usage

Mony includes 8 designer personas: **modern**, **funky**, **conservative**, **brutalist**, **cyberpunk**, **material**, **playful**, **skeuomorphic**

### Streamlit UI

Prefer a visual interface? Launch the Streamlit app to pick personas, enter your
own prompt, and manage reference images without typing CLI commands:

```bash
streamlit run mony/ui.py
```

The UI loads designer personas from the configured directory, lets you upload or
link reference imagery, and displays generated outputs inline.

#### Run the Streamlit UI inside a virtual environment

```bash
# create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install the minimum dependencies for the UI
python -m pip install -r requirements.txt

# launch Streamlit
python -m streamlit run mony/ui.py
```

When you're done, exit the app with `Ctrl+C` and deactivate the environment with `deactivate`.

### Command line

#### Running from source (without installation)

If you haven't installed the package with `pip install -e .`, you can run the CLI directly using Python's `-m` flag:

```bash
# activate your virtual environment first
source .venv/bin/activate

# run the CLI
python -m mony "Cross-platform personal finance dashboard" modern funky conservative
```

#### Running after installation

Once installed with `pip install -e .`, you can use the `mony` command directly:

```bash
mony "Cross-platform personal finance dashboard" modern funky conservative
```

#### Examples

Run in dry-run mode to inspect the assembled prompts without making API calls:

```bash
python -m mony "Meditation mobile onboarding flow" modern funky --dry-run
```

Customize options such as aspect ratio/size, prompt suffix, and output directory:

```bash
python -m mony "AI writing assistant workspace" modern \
  --size 9:16 \
  --prompt-suffix "Render as a Figma mockup"
```

`--size` accepts `WIDTHxHEIGHT` or `W:H` (e.g., `768x1344` or `9:16`). When supported by the model, this maps to `image_config.aspect_ratio`.

Default model is `google/gemini-2.5-flash-image-preview`. You can override with `--model`.

Provide reference imagery from a local file or URL. Local files are base64-inlined and
sent as data URLs within the chat message; URLs are passed directly. Repeat `--reference`
to include multiple images:

```bash
python -m mony "Smart home control hub" modern \
  --size 9:16 \
  --reference ./promo.png \
  --reference https://example.com/layout-inspiration.png
```

Generated images are saved under the `generatedDesigns/` directory with timestamps.

Note: Use an image-capable chat model (output modality includes "image"). See the OpenRouter docs: https://openrouter.ai/docs/features/multimodal/image-generation

## Environment variables

- `OPENROUTER_API_KEY`: required for non dry-run usage. Loaded from the current
  environment or the `.env` file specified via `--env-file`.
