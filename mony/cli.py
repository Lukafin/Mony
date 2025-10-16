"""Command line interface for generating UI concept images with designer prompts."""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import requests

API_URL = "https://openrouter.ai/api/v1/images"
DEFAULT_MODEL = "openai/dall-e-3"
DEFAULT_IMAGE_SIZE = "1024x1024"


@dataclass
class DesignerPrompt:
    """Container for a designer prompt template."""

    name: str
    prompt: str


class DesignerPromptError(RuntimeError):
    """Raised when designer prompt loading fails."""


class ImageGenerationError(RuntimeError):
    """Raised when the OpenRouter API returns an invalid response."""


class ReferenceInputError(RuntimeError):
    """Raised when reference image inputs cannot be prepared."""


@dataclass
class ReferenceInput:
    """Metadata about a reference image provided to the API."""

    source: str
    payload: dict


def load_env_file(path: pathlib.Path) -> None:
    """Load environment variables from a .env style file if it exists."""

    if not path.exists():
        return

    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        os.environ.setdefault(key, value)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate UI concept art images by combining a project brief with "
            "designer-specific prompts and calling the OpenRouter image API."
        )
    )
    parser.add_argument(
        "description",
        help="A short description of the UI you want to visualize."
    )
    parser.add_argument(
        "designers",
        nargs="+",
        help=(
            "Designer prompt names to use. Each name resolves to a .md file inside "
            "the designer directory."
        ),
    )
    parser.add_argument(
        "--designer-dir",
        default="designers",
        help="Directory containing designer prompt markdown files.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where generated images will be stored.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenRouter image model identifier to use.",
    )
    parser.add_argument(
        "--size",
        default=DEFAULT_IMAGE_SIZE,
        help="Image size in WIDTHxHEIGHT format (API dependent).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts without calling the OpenRouter API.",
    )
    parser.add_argument(
        "--prompt-suffix",
        default="",
        help="Optional additional instructions appended to every prompt.",
    )
    parser.add_argument(
        "--reference",
        action="append",
        default=[],
        metavar="PATH_OR_URL",
        help=(
            "Reference image path or URL to send alongside the prompt. Repeat the "
            "flag to include multiple references."
        ),
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to a .env file containing OPENROUTER_API_KEY.",
    )
    return parser.parse_args(argv)


def load_designer_prompt(designer_dir: pathlib.Path, name: str) -> DesignerPrompt:
    candidate_names: List[str] = []
    path = designer_dir / name
    candidate_names.append(str(path))
    if path.suffix.lower() != ".md":
        candidate = designer_dir / f"{name}.md"
        candidate_names.append(str(candidate))
        path = candidate
    if not path.exists():
        raise DesignerPromptError(
            f"Designer prompt '{name}' not found. Tried: {', '.join(candidate_names)}"
        )
    prompt_text = path.read_text().strip()
    if not prompt_text:
        raise DesignerPromptError(f"Designer prompt '{name}' is empty.")
    return DesignerPrompt(name=path.stem, prompt=prompt_text)


def ensure_output_dir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_prompt(description: str, prompt_template: str, suffix: str = "") -> str:
    description_block = description.strip()
    full_prompt = f"{prompt_template.strip()}\n\nProject brief: {description_block}"
    if suffix:
        full_prompt = f"{full_prompt}\n\nAdditional guidance: {suffix.strip()}"
    return full_prompt


def is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def prepare_reference_inputs(values: Sequence[str]) -> List[ReferenceInput]:
    references: List[ReferenceInput] = []
    for value in values:
        trimmed = value.strip()
        if not trimmed:
            continue
        if is_url(trimmed):
            references.append(
                ReferenceInput(source=trimmed, payload={"type": "input_image", "image_url": trimmed})
            )
            continue
        path = pathlib.Path(trimmed)
        if not path.exists():
            raise ReferenceInputError(f"Reference image '{value}' does not exist.")
        data = path.read_bytes()
        encoded = base64.b64encode(data).decode("ascii")
        references.append(
            ReferenceInput(
                source=str(path),
                payload={"type": "input_image", "image_base64": encoded},
            )
        )
    return references


def request_image(
    api_key: str,
    prompt: str,
    model: str,
    size: str,
    references: Optional[Sequence[ReferenceInput]] = None,
) -> bytes:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/",
        "X-Title": "Mony UI Generator",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "prompt": prompt, "size": size}
    if references:
        payload["image[]"] = [
            {"type": "input_text", "text": prompt},
            *[ref.payload for ref in references],
        ]
    response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    if response.status_code >= 400:
        raise ImageGenerationError(
            f"OpenRouter API returned status {response.status_code}: {response.text}"
        )
    try:
        data = response.json()
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive programming
        raise ImageGenerationError("Failed to parse OpenRouter response as JSON") from exc

    if not isinstance(data, dict) or "data" not in data:
        raise ImageGenerationError("OpenRouter response missing 'data' field")

    items = data.get("data")
    if not items:
        raise ImageGenerationError("OpenRouter response data is empty")

    image_info = items[0]
    if "b64_json" in image_info:
        return base64.b64decode(image_info["b64_json"])
    if "url" in image_info:
        download = requests.get(image_info["url"], timeout=60)
        download.raise_for_status()
        return download.content
    raise ImageGenerationError("OpenRouter response did not include image content")


def save_image(content: bytes, output_dir: pathlib.Path, designer_name: str) -> pathlib.Path:
    ensure_output_dir(output_dir)
    timestamp = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    filename = f"{timestamp}_{designer_name}.png"
    path = output_dir / filename
    path.write_bytes(content)
    return path


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    load_env_file(pathlib.Path(args.env_file))
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key and not args.dry_run:
        print(
            "OPENROUTER_API_KEY is not set. Provide it in the environment or via the .env file.",
            file=sys.stderr,
        )
        return 1

    designer_dir = pathlib.Path(args.designer_dir)
    output_dir = pathlib.Path(args.output_dir)
    try:
        references = prepare_reference_inputs(args.reference)
    except ReferenceInputError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    results = []
    for name in args.designers:
        try:
            designer_prompt = load_designer_prompt(designer_dir, name)
        except DesignerPromptError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        full_prompt = build_prompt(args.description, designer_prompt.prompt, args.prompt_suffix)
        if args.dry_run:
            print(f"=== {designer_prompt.name} ===")
            print(full_prompt)
            if references:
                print("References:")
                for ref in references:
                    print(f"  - {ref.source}")
            print()
            continue
        try:
            image_bytes = request_image(
                api_key,
                full_prompt,
                args.model,
                args.size,
                references=references,
            )
        except ImageGenerationError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        except requests.RequestException as exc:
            print(f"Failed to call OpenRouter API: {exc}", file=sys.stderr)
            return 1
        image_path = save_image(image_bytes, output_dir, designer_prompt.name)
        results.append((designer_prompt.name, image_path))

    if args.dry_run:
        return 0

    for designer_name, image_path in results:
        print(f"Generated image for {designer_name}: {image_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
