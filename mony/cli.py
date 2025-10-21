"""Command line interface for generating UI concept images with designer prompts."""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import logging
import os
import pathlib
import re
import sys
import time
import threading
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# OpenRouter chat completions endpoint (supports image generation with modalities)
API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-2.5-flash-image-preview"
DEFAULT_IMAGE_SIZE = "1024x1024"
LOG_LEVEL_NAMES = {
    "CRITICAL",
    "ERROR",
    "WARNING",
    "INFO",
    "DEBUG",
}

# Find the bundled designers directory (works whether installed via pip or run locally)
_PACKAGE_DIR = pathlib.Path(__file__).parent
DEFAULT_DESIGNER_DIR = _PACKAGE_DIR / "designers"
if not DEFAULT_DESIGNER_DIR.exists():
    # Fallback for running from source
    DEFAULT_DESIGNER_DIR = _PACKAGE_DIR.parent / "designers"


DEFAULT_RESEARCH_INSTRUCTIONS = (
    "You are an award-winning UI creative director who researches cutting-edge digital "
    "product design trends. Using reputable, current sources, summarize the most "
    "influential UI/UX aesthetics, interaction patterns, and visual motifs gaining "
    "traction in {year}. Synthesize them into a concise creative brief for a designer "
    "persona named '{name}'. Focus on color palettes, typography, layout structures, "
    "motion principles, and signature flourishes that should inspire UI concept art. "
    "Provide actionable instructions suitable for a text-to-image prompt. Limit the "
    "response to 4-6 sentences without markdown headings."
)


logger = logging.getLogger("mony.cli")


def normalize_log_level(value: str) -> str:
    """Normalize user-provided log level strings."""

    upper_value = value.strip().upper()
    if upper_value == "WARN":
        upper_value = "WARNING"
    if upper_value not in LOG_LEVEL_NAMES:
        valid = ", ".join(sorted(LOG_LEVEL_NAMES))
        raise argparse.ArgumentTypeError(f"Invalid log level '{value}'. Choose one of: {valid}")
    return upper_value


def configure_logging(level_name: str) -> None:
    """Configure root logging once based on the requested level."""

    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(threadName)s %(name)s: %(message)s",
    )


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


class DesignerResearchError(RuntimeError):
    """Raised when designer research via the Perplexity API fails."""


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
        default=str(DEFAULT_DESIGNER_DIR),
        help="Directory containing designer prompt markdown files.",
    )
    parser.add_argument(
        "--output-dir",
        default="generatedDesigns",
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
    parser.add_argument(
        "--research-designer",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "Generate or refresh a designer prompt markdown file named NAME using Perplexity "
            "before running. Requires the PERPLEXITY_API_KEY environment variable."
        ),
    )
    parser.add_argument(
        "--research-instructions",
        default=None,
        help=(
            "Custom Perplexity instructions used with --research-designer. "
            "Placeholders {name} and {year} are replaced automatically."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        type=normalize_log_level,
        help="Logging level to emit (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
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


def sanitize_designer_slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", name.strip().lower())
    slug = slug.strip("-")
    if not slug:
        slug = "designer"
    return slug


def extract_text_from_perplexity_message(message: object) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, list):
        texts: List[str] = []
        for part in content:
            if isinstance(part, str):
                candidate = part
            else:
                candidate = getattr(part, "text", None)
                if candidate is None and isinstance(part, dict):
                    candidate = part.get("text")
            if candidate:
                candidate = str(candidate).strip()
                if candidate:
                    texts.append(candidate)
        return "\n\n".join(texts)
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return str(content)


def strip_perplexity_think_blocks(text: str) -> str:
    """Remove Perplexity <think> ... </think> segments from responses."""

    pattern = re.compile(r"<think>.*?</think>\s*", re.IGNORECASE | re.DOTALL)
    return re.sub(pattern, "", text)


def research_trendy_designer_prompt(
    designer_dir: pathlib.Path,
    name: str,
    *,
    client: object,
    instructions: Optional[str] = None,
) -> pathlib.Path:
    slug = sanitize_designer_slug(name)
    filename = designer_dir / f"{slug}.md"

    current_year = dt.datetime.utcnow().year
    logger.info("Researching updated designer persona for %s", slug)

    prompt_template = (instructions or DEFAULT_RESEARCH_INSTRUCTIONS).strip()
    prompt = (
        prompt_template.replace("{year}", str(current_year)).replace("{name}", name)
    ).strip()
    if not prompt:
        raise DesignerResearchError(
            "Research instructions are empty after formatting. Provide a non-empty prompt."
        )

    try:
        if hasattr(client, "messages"):
            response = client.messages.create(  # type: ignore[attr-defined]
                model="sonar-reasoning",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a meticulous design trend researcher who writes vivid persona "
                            "prompts for generating UI concept art."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )
        elif hasattr(client, "chat") and hasattr(client.chat, "completions"):
            response = client.chat.completions.create(  # type: ignore[attr-defined]
                model="sonar-reasoning",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a meticulous design trend researcher who writes vivid persona "
                            "prompts for generating UI concept art."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                stream=False,
            )
        else:  # pragma: no cover - unexpected client interface
            raise DesignerResearchError("Perplexity client does not expose a supported completion endpoint.")
    except Exception as exc:  # pragma: no cover - network failure handling
        raise DesignerResearchError(f"Failed to call Perplexity API: {exc}") from exc

    choices = getattr(response, "choices", None)
    if not choices:
        raise DesignerResearchError("Perplexity response did not include choices")
    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if message is None:
        raise DesignerResearchError("Perplexity response missing message content")
    text = extract_text_from_perplexity_message(message).strip()
    text = strip_perplexity_think_blocks(text).strip()
    if not text:
        raise DesignerResearchError("Perplexity response did not contain usable text")

    ensure_output_dir(designer_dir)
    filename.write_text(text + "\n")
    logger.info("Saved researched designer prompt to %s", filename)
    return filename


def run_designer_research(
    designer_dir: pathlib.Path,
    names: Sequence[str],
    instructions: Optional[str] = None,
) -> None:
    if not names:
        return
    try:
        from perplexityai import Perplexity  # type: ignore
    except ImportError:
        try:  # pragma: no cover - optional dependency
            from perplexity import Perplexity  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise DesignerResearchError(
                "The Perplexity client library is required for --research-designer. Install it via 'pip install perplexityai'."
            ) from exc

    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key:
        raise DesignerResearchError(
            "PERPLEXITY_API_KEY is not set. Provide it in the environment or .env file to use --research-designer."
        )

    client = Perplexity(api_key=api_key)
    for name in names:
        research_trendy_designer_prompt(
            designer_dir, name, client=client, instructions=instructions
        )


def build_prompt(description: str, prompt_template: str, suffix: str = "") -> str:
    description_block = description.strip()
    full_prompt = f"{prompt_template.strip()}\n\nProject brief: {description_block}"
    if suffix:
        full_prompt = f"{full_prompt}\n\nAdditional guidance: {suffix.strip()}"
    return full_prompt


def derive_aspect_ratio(size: str) -> Optional[str]:
    """Convert size into an aspect ratio string acceptable by providers.

    Accepts either WIDTHxHEIGHT or W:H. Returns None if parsing fails.
    """
    text = size.strip().lower()
    # Direct W:H support (e.g., "9:16")
    if ":" in text:
        parts = text.split(":", 1)
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return f"{int(parts[0])}:{int(parts[1])}"
        return None
    if "x" not in text:
        return None
    try:
        width_text, height_text = text.split("x", 1)
        width = int(width_text)
        height = int(height_text)
        if width <= 0 or height <= 0:
            return None
    except ValueError:
        return None

    def gcd(a: int, b: int) -> int:
        while b:
            a, b = b, a % b
        return a

    d = gcd(width, height)
    return f"{width // d}:{height // d}"


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
    # Build chat completions payload with image modalities per OpenRouter docs
    # Build messages content: use plain string when no references for max compatibility
    if references:
        # Convert internal reference payloads to chat image_url parts
        parts: List[dict] = [{"type": "text", "text": prompt}]
        for ref in references:
            if "image_url" in ref.payload:
                parts.append({"type": "image_url", "image_url": {"url": ref.payload["image_url"]}})
            elif "image_base64" in ref.payload:
                parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{ref.payload['image_base64']}"}})
        content_value = parts
    else:
        content_value = prompt

    payload: dict = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": content_value,
            }
        ],
        "modalities": ["image", "text"],
    }

    aspect_ratio = derive_aspect_ratio(size)
    if aspect_ratio:
        payload["image_config"] = {"aspect_ratio": aspect_ratio}

    prompt_preview = prompt.strip().replace("\n", " ")[:160]
    references_count = len(references) if references else 0
    logger.debug(
        "Submitting request to model=%s size=%s aspect_ratio=%s prompt_preview=%r references=%d",
        model,
        size,
        payload.get("image_config", {}).get("aspect_ratio"),
        prompt_preview,
        references_count,
    )
    request_start = time.perf_counter()
    response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    elapsed = time.perf_counter() - request_start
    logger.debug(
        "Received response for model=%s status=%s in %.2fs",
        model,
        response.status_code,
        elapsed,
    )
    if response.status_code >= 400:
        logger.error(
            "OpenRouter API responded with status=%s for model=%s: %s",
            response.status_code,
            model,
            response.text[:500],
        )
        raise ImageGenerationError(
            f"OpenRouter API returned status {response.status_code}: {response.text}"
        )
    try:
        data = response.json()
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive programming
        print(f"DEBUG: Response status: {response.status_code}", file=sys.stderr)
        print(f"DEBUG: Response text: {response.text}", file=sys.stderr)
        raise ImageGenerationError("Failed to parse OpenRouter response as JSON") from exc

    if logger.isEnabledFor(logging.DEBUG):
        try:
            serialized = json.dumps(data)
        except (TypeError, ValueError):
            serialized = str(data)
        logger.debug(
            "Raw provider response for model=%s: %s",
            model,
            serialized[:2000],
        )

    # Parse images from chat completions-style response
    if not isinstance(data, dict) or "choices" not in data:
        raise ImageGenerationError("OpenRouter response missing 'choices' field")

    choices = data.get("choices") or []
    if not choices or not isinstance(choices, list):
        raise ImageGenerationError("OpenRouter response has no choices")

    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    images = message.get("images") or []

    # Fallback: some providers may stream images in content; try to extract if present
    if not images and isinstance(message.get("content"), list):
        # Look for dicts with type 'image_url'
        images = [part for part in message["content"] if isinstance(part, dict) and part.get("type") == "image_url"]
        # Normalize shape to match expected images format
        images = [{"image_url": part.get("image_url") or {"url": part.get("url")}} for part in images]

    if not images:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Response for model=%s missing images; message payload: %r",
                model,
                message,
            )
        raise ImageGenerationError("OpenRouter response did not include images")

    first = images[0]
    image_url_info = first.get("image_url") or {}
    url_value = image_url_info.get("url")
    if not url_value or not isinstance(url_value, str):
        raise ImageGenerationError("Image URL missing in response")

    if url_value.startswith("data:image/"):
        # data URL: data:image/png;base64,....
        try:
            base64_index = url_value.index(",")
            b64 = url_value[base64_index + 1 :]
        except ValueError:
            raise ImageGenerationError("Malformed data URL for image")
        return base64.b64decode(b64)

    # Otherwise download the image from the provided URL
    logger.debug("Downloading image asset for model=%s from %s", model, url_value)
    download = requests.get(url_value, timeout=60)
    download.raise_for_status()
    return download.content


def save_image(content: bytes, output_dir: pathlib.Path, designer_name: str) -> pathlib.Path:
    ensure_output_dir(output_dir)
    timestamp = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    filename = f"{timestamp}_{designer_name}.png"
    path = output_dir / filename
    path.write_bytes(content)
    return path


def generate_image_for_designer(
    *,
    api_key: str,
    designer_dir: pathlib.Path,
    output_dir: pathlib.Path,
    description: str,
    designer_name: str,
    model: str,
    size: str,
    prompt_suffix: str,
    references: Sequence[ReferenceInput],
) -> Tuple[str, pathlib.Path]:
    designer_prompt = load_designer_prompt(designer_dir, designer_name)
    full_prompt = build_prompt(description, designer_prompt.prompt, prompt_suffix)
    # Retry transient failures, e.g., provider returns no images
    max_attempts = 3
    for attempt_index in range(max_attempts):
        try:
            attempt_number = attempt_index + 1
            logger.info(
                "Generating image for designer=%s attempt=%d/%d model=%s size=%s references=%d",
                designer_name,
                attempt_number,
                max_attempts,
                model,
                size,
                len(references),
            )
            image_bytes = request_image(
                api_key,
                full_prompt,
                model,
                size,
                references=references,
            )
            break
        except (ImageGenerationError, requests.RequestException) as exc:
            is_last = attempt_index == max_attempts - 1
            if is_last:
                logger.error(
                    "Exhausted retries for designer=%s after %d attempts (last_error=%s: %s)",
                    designer_name,
                    max_attempts,
                    type(exc).__name__,
                    exc,
                    exc_info=logger.isEnabledFor(logging.DEBUG),
                )
                raise
            delay_seconds = 1.0 * (attempt_index + 1)
            logger.warning(
                "Retrying designer=%s due to %s: %s (attempt=%d/%d) sleeping=%.1fs",
                designer_name,
                type(exc).__name__,
                exc,
                attempt_number,
                max_attempts,
                delay_seconds,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            time.sleep(delay_seconds)
    image_path = save_image(image_bytes, output_dir, designer_prompt.name)
    logger.info(
        "Saved image for designer=%s at %s (thread=%s)",
        designer_prompt.name,
        image_path,
        threading.current_thread().name,
    )
    return designer_prompt.name, image_path


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    configure_logging(args.log_level)

    load_env_file(pathlib.Path(args.env_file))

    designer_dir = pathlib.Path(args.designer_dir)
    output_dir = pathlib.Path(args.output_dir)

    try:
        run_designer_research(
            designer_dir, args.research_designer, instructions=args.research_instructions
        )
    except DesignerResearchError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key and not args.dry_run:
        print(
            "OPENROUTER_API_KEY is not set. Provide it in the environment or via the .env file.",
            file=sys.stderr,
        )
        return 1

    try:
        references = prepare_reference_inputs(args.reference)
    except ReferenceInputError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    results: List[Tuple[str, pathlib.Path]] = []
    # Dry-run preserves sequential prompt printing for readability
    if args.dry_run:
        for name in args.designers:
            try:
                designer_prompt = load_designer_prompt(designer_dir, name)
            except DesignerPromptError as exc:
                print(str(exc), file=sys.stderr)
                return 1
            full_prompt = build_prompt(args.description, designer_prompt.prompt, args.prompt_suffix)
            print(f"=== {designer_prompt.name} ===")
            print(full_prompt)
            if references:
                print("References:")
                for ref in references:
                    print(f"  - {ref.source}")
            print()
        return 0

    # Parallelize per-designer image generation
    errors: List[str] = []
    results_map: Dict[str, pathlib.Path] = {}
    try:
        max_workers = min(8, max(1, len(args.designers)))
        logger.info(
            "Starting parallel generation for %d designers with max_workers=%d",
            len(args.designers),
            max_workers,
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_name = {
                executor.submit(
                    generate_image_for_designer,
                    api_key=api_key,
                    designer_dir=designer_dir,
                    output_dir=output_dir,
                    description=args.description,
                    designer_name=name,
                    model=args.model,
                    size=args.size,
                    prompt_suffix=args.prompt_suffix,
                    references=references,
                ): name
                for name in args.designers
            }

            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    designer_name, image_path = future.result()
                    results_map[designer_name] = image_path
                    logger.debug(
                        "Future completed for designer=%s image_path=%s",
                        designer_name,
                        image_path,
                    )
                except DesignerPromptError as exc:
                    errors.append(str(exc))
                except ImageGenerationError as exc:
                    errors.append(str(exc))
                except requests.RequestException as exc:
                    errors.append(f"Failed to call OpenRouter API for {name}: {exc}")
    except Exception as exc:
        print(f"Unexpected error during parallel generation: {exc}", file=sys.stderr)
        return 1

    # Preserve input order in final output
    for name in args.designers:
        if name in results_map:
            results.append((name, results_map[name]))

    if errors:
        for msg in errors:
            print(msg, file=sys.stderr)
        # Continue to print any successful results below, but exit non-zero
        exit_code = 1
    else:
        exit_code = 0

    for designer_name, image_path in results:
        print(f"Generated image for {designer_name}: {image_path}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
