# Mony

Visualize different UI directions by combining project briefs with designer
personas and generating concept images through the OpenRouter image API.

## Features

- Describe the UI you want to explore via a CLI argument.
- Mix and match designer prompt templates stored as markdown files.
- Generate concept images for each selected designer using OpenRouter.
- Preview prompts without generating images via `--dry-run` mode.
- Supply one or more reference images or URLs to steer the generation results.

## Prerequisites

1. Create a Python 3.10+ virtual environment and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the project root with your OpenRouter key:

   ```bash
   echo "OPENROUTER_API_KEY=sk-or-..." > .env
   ```

## Usage

Designer prompts live inside the `designers/` directory. The CLI resolves the
provided names to markdown files (e.g. `modern` → `designers/modern.md`).

Generate images for multiple designer personas:

```bash
python -m mony "Cross-platform personal finance dashboard" modern funky conservative
```

Run in dry-run mode to inspect the assembled prompts without making API calls:

```bash
python -m mony "Meditation mobile onboarding flow" modern funky --dry-run
```

Customize options such as model, aspect ratio/size, prompt suffix, and output directory:

```bash
python -m mony "AI writing assistant workspace" modern \
  --model google/gemini-2.5-flash-image-preview \
  --size 9:16 \
  --prompt-suffix "Render as a Figma mockup"
```

`--size` accepts `WIDTHxHEIGHT` or `W:H` (e.g., `768x1344` or `9:16`). When supported by the model, this maps to `image_config.aspect_ratio`.

Provide reference imagery from a local file or URL. Local files are base64-inlined and
sent as data URLs within the chat message; URLs are passed directly. Repeat `--reference`
to include multiple images:

```bash
python -m mony "Smart home control hub" modern \
  --model google/gemini-2.5-flash-image-preview \
  --size 9:16 \
  --reference ./referenceImages/promo.png \
  --reference https://example.com/layout-inspiration.png
```

Generated images are saved under the `outputs/` directory with timestamps.

Note: Use an image-capable chat model (output modality includes "image"). See the OpenRouter docs: https://openrouter.ai/docs/features/multimodal/image-generation

## Environment variables

- `OPENROUTER_API_KEY`: required for non dry-run usage. Loaded from the current
  environment or the `.env` file specified via `--env-file`.
