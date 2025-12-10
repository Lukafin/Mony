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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# OpenRouter chat completions endpoint (supports image generation with modalities)
API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-3-pro-image-preview"
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
    "You are an award-winning UI creative director and design trend researcher. "
    "Your task is to research and synthesize cutting-edge digital product design trends "
    "for {year} into a compelling designer persona named '{name}'.\n\n"
    
    "## Research Focus Areas\n"
    "Investigate the following aspects using reputable, current sources (Dribbble, Behance, "
    "Awwwards, design blogs, major product launches, design system updates):\n\n"
    
    "**Visual Language:**\n"
    "- Color palettes: dominant colors, accent combinations, gradient trends, dark/light mode approaches\n"
    "- Typography: trending typefaces, font pairings, sizing conventions, variable font usage\n"
    "- Iconography: style (outlined, filled, duotone), sizing, animation treatments\n"
    "- Imagery: illustration styles, photography treatments, 3D elements, AI-generated art\n\n"
    
    "**Layout & Structure:**\n"
    "- Grid systems and spacing conventions\n"
    "- Component patterns (cards, modals, navigation)\n"
    "- Responsive and mobile-first approaches\n"
    "- Bento grids, asymmetric layouts, or structured systems\n\n"
    
    "**Interaction & Motion:**\n"
    "- Micro-interactions and feedback patterns\n"
    "- Page transitions and scroll behaviors\n"
    "- Loading states and skeleton screens\n"
    "- Gesture-based interactions\n\n"
    
    "**Emerging Techniques:**\n"
    "- Glass morphism, neumorphism, or new visual effects\n"
    "- AI-assisted design elements\n"
    "- Accessibility innovations\n"
    "- Cross-platform design language evolution\n\n"
    
    "## Output Format\n"
    "Synthesize your research into a concise, actionable creative brief (4-8 sentences). "
    "The persona should:\n"
    "- Have a distinct personality and design philosophy\n"
    "- Include specific, implementable visual guidelines\n"
    "- Reference real-world inspiration sources\n"
    "- Be suitable for guiding UI concept generation\n\n"
    
    "Write in second person ('You are...') as instructions for a designer. "
    "Avoid markdown headings in the output. Focus on what makes this persona unique "
    "and timely for {year}."
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


def _reference_to_message_part(reference: ReferenceInput) -> Optional[dict]:
    """Convert a ReferenceInput payload into a chat content part."""

    payload = reference.payload or {}
    ref_type = payload.get("type") or "input_image"
    part: dict = {"type": ref_type}

    url_value = payload.get("image_url")
    if isinstance(url_value, dict):
        part["image_url"] = url_value
        return part
    if isinstance(url_value, str) and url_value.strip():
        part["image_url"] = {"url": url_value.strip()}
        return part

    base64_value = payload.get("image_base64")
    if isinstance(base64_value, str) and base64_value.strip():
        encoded = base64_value.strip()
        if not encoded.startswith("data:image/"):
            encoded = f"data:image/png;base64,{encoded}"
        part["image_url"] = {"url": encoded}
        return part

    inline_value = payload.get("inline_data")
    if isinstance(inline_value, dict):
        part["inline_data"] = inline_value
        return part

    return None


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
                model="sonar-pro",
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
                model="sonar-pro",
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


def _extract_text_from_openrouter_message(message: object) -> str:
    """Extract textual content from an OpenRouter chat message."""

    content = getattr(message, "content", None)
    if isinstance(message, dict):
        content = message.get("content", content)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        texts: List[str] = []
        for part in content:
            candidate = None
            if isinstance(part, str):
                candidate = part
            elif isinstance(part, dict):
                candidate = part.get("text") or part.get("content")
            else:
                candidate = getattr(part, "text", None)
            if candidate:
                candidate = str(candidate).strip()
                if candidate:
                    texts.append(candidate)
        return "\n\n".join(texts).strip()

    return ""


def create_persona_from_image(
    designer_dir: pathlib.Path,
    persona_name: str,
    *,
    api_key: str,
    model: str,
    image_bytes: bytes,
    instructions: Optional[str] = None,
) -> pathlib.Path:
    """Create and persist a designer persona markdown using an image as input."""

    slug = sanitize_designer_slug(persona_name)
    if not persona_name.strip():
        raise DesignerPromptError("Provide a persona name before creating it from an image.")
    if not api_key.strip():
        raise DesignerPromptError("OPENROUTER_API_KEY is required to create a persona from an image.")
    if not image_bytes:
        raise DesignerPromptError("Upload an image to derive a persona.")

    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "HTTP-Referer": "https://github.com/",
        "X-Title": "Mony Persona Creator",
        "Content-Type": "application/json",
    }

    encoded_image = base64.b64encode(image_bytes).decode("ascii")
    guidance = (
        instructions
        or (
            "Analyze this UI or visual style image and write a concise designer persona prompt "
            "for generating UI concept art. Describe color palette, typography, layout, motion "
            "and signature flourishes in 4-6 sentences. Do not include markdown headings."
        )
    ).strip()
    if not guidance:
        guidance = (
            "Analyze this UI/visual style image and produce a concise designer persona prompt "
            "for UI concept art."
        )

    payload: dict = {
        "model": model.strip(),
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a senior creative director. Given an image, you write a concise, "
                    "actionable designer persona prompt suitable for generating UI concept art."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": guidance},
                    {
                        "type": "input_image",
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": encoded_image,
                        },
                    },
                ],
            }
        ],
        "modalities": ["text"],
        "include_reasoning": False,
        "reasoning": {"exclude": True},
    }

    logger.info("Creating persona '%s' from image via model=%s", slug, model)
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    except Exception as exc:  # pragma: no cover - network failure handling
        raise DesignerPromptError(f"Failed to call OpenRouter: {exc}") from exc

    if response.status_code >= 400:
        logger.error(
            "OpenRouter API responded with status=%s for model=%s: %s",
            response.status_code,
            model,
            response.text[:500],
        )
        raise DesignerPromptError(
            f"OpenRouter API returned status {response.status_code}: {response.text}"
        )

    try:
        data = response.json()
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive parsing
        raise DesignerPromptError("Failed to parse OpenRouter response as JSON") from exc

    choices = data.get("choices") if isinstance(data, dict) else None
    if not choices or not isinstance(choices, list):
        raise DesignerPromptError("OpenRouter response did not include choices.")

    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    persona_text = _extract_text_from_openrouter_message(message).strip()
    if not persona_text:
        raise DesignerPromptError("OpenRouter response did not contain persona text.")

    ensure_output_dir(designer_dir)
    filename = designer_dir / f"{slug}.md"
    if not persona_text.endswith("\n"):
        persona_text += "\n"
    filename.write_text(persona_text)
    logger.info("Saved image-derived persona prompt to %s", filename)
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
    # Explicit instruction prefix to ensure the model generates an actual image
    # rather than just describing one
    image_instruction = (
        "Generate an image of a UI design. Do not describe the image - actually create "
        "and output the image. The design should follow these guidelines:\n\n"
    )
    full_prompt = f"{image_instruction}{prompt_template.strip()}\n\nProject brief: {description_block}"
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


def _decode_base64_image_data(value: str) -> bytes:
    """Decode plain base64 strings or full data URLs into binary bytes."""

    data = value.strip()
    if not data:
        raise ImageGenerationError("Empty base64 image data in response")
    if data.startswith("data:image/"):
        try:
            base64_index = data.index(",")
        except ValueError as exc:
            raise ImageGenerationError("Malformed data URL for image") from exc
        data = data[base64_index + 1 :]
    try:
        return base64.b64decode(data)
    except Exception as exc:  # pragma: no cover - defensive parsing
        raise ImageGenerationError("Failed to decode base64 image data") from exc


def _extract_inline_base64(entry: dict) -> Optional[str]:
    inline_data = entry.get("inline_data") or entry.get("inlineData")
    if isinstance(inline_data, dict):
        data_value = inline_data.get("data")
        if isinstance(data_value, str) and data_value.strip():
            mime = inline_data.get("mime_type") or inline_data.get("mimeType")
            if mime and not data_value.startswith("data:"):
                return f"data:{mime};base64,{data_value.strip()}"
            return data_value.strip()
    return None


def _extract_image_url(entry: dict) -> Optional[str]:
    for key in ("image_url", "imageUrl", "url"):
        candidate = entry.get(key)
        if isinstance(candidate, dict):
            url_value = candidate.get("url") or candidate.get("href")
            if isinstance(url_value, str) and url_value.strip():
                return url_value.strip()
        elif isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _extract_base64_value(entry: dict) -> Optional[str]:
    for key in ("image_base64", "imageBase64", "b64_json", "b64Json", "data"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    inline_value = _extract_inline_base64(entry)
    if inline_value:
        return inline_value
    return None


def _collect_image_entries(message: dict) -> List[dict]:
    entries: List[dict] = []
    raw_images = message.get("images")
    if isinstance(raw_images, dict):
        entries.append(raw_images)
    elif isinstance(raw_images, list):
        entries.extend(entry for entry in raw_images if isinstance(entry, dict))

    content = message.get("content")
    if isinstance(content, dict):
        entries.append(content)
    elif isinstance(content, list):
        entries.extend(part for part in content if isinstance(part, dict))

    return entries


def _log_missing_image_details(model: str, message: dict) -> None:
    summary = {
        "message_keys": sorted(message.keys()),
        "images_type": type(message.get("images")).__name__,
        "content_type": type(message.get("content")).__name__,
    }
    try:
        serialized = json.dumps(message)
    except (TypeError, ValueError):
        serialized = str(message)
    logger.warning(
        "Provider response for model=%s did not include usable images. summary=%s payload_preview=%s",
        model,
        summary,
        serialized,
    )


def _download_or_decode_image_entry(model: str, entry: dict) -> Optional[bytes]:
    url_value = _extract_image_url(entry)
    if isinstance(url_value, str) and url_value:
        if url_value.startswith("data:image/"):
            return _decode_base64_image_data(url_value)
        logger.debug("Downloading image asset for model=%s from %s", model, url_value)
        download = requests.get(url_value, timeout=60)
        download.raise_for_status()
        return download.content

    base64_value = _extract_base64_value(entry)
    if isinstance(base64_value, str) and base64_value:
        return _decode_base64_image_data(base64_value)

    return None


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
    content_value: Union[str, List[dict]]
    if references:
        parts: List[dict] = [{"type": "input_text", "text": prompt}]
        for ref in references:
            part = _reference_to_message_part(ref)
            if part:
                parts.append(part)
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
        "include_reasoning": False,
        "reasoning": {"exclude": True},
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
            serialized,
        )

    # Parse images from chat completions-style response
    if not isinstance(data, dict) or "choices" not in data:
        raise ImageGenerationError("OpenRouter response missing 'choices' field")

    choices = data.get("choices") or []
    if not choices or not isinstance(choices, list):
        raise ImageGenerationError("OpenRouter response has no choices")

    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    image_entries = _collect_image_entries(message)
    image_bytes = None
    for entry in image_entries:
        image_bytes = _download_or_decode_image_entry(model, entry)
        if image_bytes:
            break

    if not image_bytes and isinstance(message.get("content"), str):
        inline_text = message["content"].strip()
        if inline_text.startswith("data:image/") or inline_text.startswith("http"):
            image_bytes = _download_or_decode_image_entry(
                model, {"image_url": inline_text}
            )
        # Also scan for embedded data URLs within the text content
        if not image_bytes:
            data_url_match = re.search(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+', inline_text)
            if data_url_match:
                logger.debug("Found embedded data URL in text content for model=%s", model)
                image_bytes = _download_or_decode_image_entry(
                    model, {"image_url": data_url_match.group(0)}
                )

    if not image_bytes:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Response for model=%s missing recognizable images; message payload: %r",
                model,
                message,
            )
        _log_missing_image_details(model, message)
        raise ImageGenerationError("OpenRouter response did not include images")

    return image_bytes


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
    prompt_override: Optional[str] = None,
) -> Tuple[str, pathlib.Path]:
    designer_prompt = load_designer_prompt(designer_dir, designer_name)
    if prompt_override is not None:
        designer_prompt = DesignerPrompt(name=designer_prompt.name, prompt=prompt_override)
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
