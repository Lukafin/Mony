"""Streamlit UI for the Mony design persona image generator."""

from __future__ import annotations

import base64
import os
import pathlib
from typing import TYPE_CHECKING, List, Sequence

import streamlit as st

from mony import cli

if TYPE_CHECKING:
    from streamlit.runtime.uploaded_file_manager import UploadedFile


def _load_available_designers(designer_dir: pathlib.Path) -> List[str]:
    """Return a sorted list of designer prompt names present on disk."""

    if not designer_dir.exists():
        return []
    names = [path.stem for path in designer_dir.glob("*.md") if path.is_file()]
    return sorted(names, key=str.lower)


def _uploaded_files_to_references(
    uploads: Sequence["UploadedFile"],
) -> List[cli.ReferenceInput]:
    """Convert uploaded files into ReferenceInput objects."""

    references: List[cli.ReferenceInput] = []
    for upload in uploads:
        data = upload.read()
        if not data:
            continue
        encoded = base64.b64encode(data).decode("ascii")
        references.append(
            cli.ReferenceInput(
                source=upload.name,
                payload={"type": "input_image", "image_base64": encoded},
            )
        )
    return references


def _prepare_reference_urls(url_text: str) -> List[cli.ReferenceInput]:
    """Parse newline-separated URL input into ReferenceInput objects."""

    urls = [line.strip() for line in url_text.splitlines() if line.strip()]
    if not urls:
        return []
    return cli.prepare_reference_inputs(urls)


def _ensure_env_loaded(env_path: pathlib.Path) -> None:
    """Load environment variables from the given env file if present."""

    if "_MONY_ENV_LOADED" in st.session_state:
        return
    cli.load_env_file(env_path)
    st.session_state["_MONY_ENV_LOADED"] = True


def _display_generated_image(path: pathlib.Path, label: str) -> None:
    """Display an image from disk with a caption inside the Streamlit app."""

    if not path.exists():
        st.warning(f"Generated image for {label} could not be found at {path}.")
        return
    st.image(str(path), caption=f"{label} – {path.name}")


def _research_designer_via_perplexity(
    designer_dir: pathlib.Path, persona_name: str, api_key: str
) -> tuple[pathlib.Path, str]:
    """Research and persist a trendy designer persona using Perplexity."""

    if not api_key.strip():
        raise cli.DesignerResearchError(
            "Provide a Perplexity API key before running designer research."
        )

    try:
        from perplexityai import Perplexity
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise cli.DesignerResearchError(
            "Install the 'perplexityai' package to enable designer research."
        ) from exc

    client = Perplexity(api_key=api_key.strip())
    path = cli.research_trendy_designer_prompt(designer_dir, persona_name, client=client)
    return path, path.read_text().strip()


def run() -> None:
    """Execute the Streamlit UI."""

    st.set_page_config(page_title="Mony Persona UI", layout="wide")

    default_env = pathlib.Path(".env")
    _ensure_env_loaded(default_env)

    st.title("Mony – UI Concept Generator")
    st.write(
        "Generate UI concept art by combining a project description with "
        "designer personas and optional reference images."
    )

    with st.sidebar:
        st.header("Configuration")
        designer_dir_text = st.text_input(
            "Designer directory", value=str(cli.DEFAULT_DESIGNER_DIR)
        )
        designer_dir = pathlib.Path(designer_dir_text).expanduser().resolve()
        output_dir = pathlib.Path(
            st.text_input("Output directory", value="generatedDesigns")
        ).expanduser().resolve()

        env_file = st.text_input(".env path", value=str(default_env))
        if st.button("Reload .env"):
            cli.load_env_file(pathlib.Path(env_file))
            st.success("Environment variables reloaded.")

        api_key_default = os.environ.get("OPENROUTER_API_KEY", "")
        api_key = st.text_input(
            "OpenRouter API key", value=api_key_default, type="password"
        )
        model = st.text_input("Model", value=cli.DEFAULT_MODEL)
        size = st.text_input("Image size", value=cli.DEFAULT_IMAGE_SIZE)
        prompt_suffix = st.text_area("Prompt suffix", value="", height=80)
        perplexity_api_key_default = os.environ.get("PERPLEXITY_API_KEY", "")
        perplexity_api_key = st.text_input(
            "Perplexity API key",
            value=perplexity_api_key_default,
            type="password",
            help="Required for researching new designer personas.",
        )

    if perplexity_api_key:
        os.environ["PERPLEXITY_API_KEY"] = perplexity_api_key

    st.subheader("Research trendy designer personas")
    research_name = st.text_input(
        "Persona name to research",
        placeholder="e.g., Neo Brutalist Visionary",
    )
    if "_latest_research" not in st.session_state:
        st.session_state["_latest_research"] = None

    if st.button("Research persona", type="secondary"):
        if not research_name.strip():
            st.error("Enter a persona name before running research.")
        else:
            with st.spinner("Researching latest design trends..."):
                try:
                    path, text = _research_designer_via_perplexity(
                        designer_dir, research_name.strip(), perplexity_api_key
                    )
                except cli.DesignerResearchError as exc:
                    st.error(str(exc))
                except Exception as exc:  # pragma: no cover - runtime feedback
                    st.error(f"Unexpected error while researching persona: {exc}")
                else:
                    st.session_state["_latest_research"] = {
                        "name": research_name.strip(),
                        "path": str(path),
                        "text": text,
                    }
                    st.success(
                        f"Saved updated persona prompt to {path.name}. It is now available in the selector below."
                    )

    latest_research = st.session_state.get("_latest_research")
    if latest_research:
        with st.expander(
            f"Latest researched persona: {latest_research['name']}", expanded=False
        ):
            st.markdown(
                f"**Stored at:** `{latest_research['path']}`\n\n"
                "You can review or edit the generated prompt below.",
            )
            st.text_area(
                "Persona prompt",
                value=latest_research["text"],
                height=200,
                disabled=True,
            )

    available_designers = _load_available_designers(designer_dir)

    st.subheader("Project brief")
    description = st.text_area(
        "Describe the UI you would like to visualize", height=120
    )

    st.subheader("Designer persona")
    col1, col2 = st.columns(2)
    with col1:
        selected_designers = st.multiselect(
            "Select built-in designers", available_designers
        )
        if not available_designers:
            st.info(
                "No designer prompts found. Adjust the designer directory path in the sidebar."
            )

    with col2:
        custom_name = st.text_input("Custom persona name", value="")
        custom_prompt = st.text_area(
            "Custom persona prompt", value="", height=160,
            help=(
                "Provide your own persona instructions. The project brief and optional "
                "suffix will be appended automatically."
            ),
        )

    st.subheader("Reference images")
    uploaded_refs = st.file_uploader(
        "Upload reference images", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True
    )
    reference_urls_text = st.text_area(
        "Reference image URLs (one per line)", value="", height=80
    )

    references: List[cli.ReferenceInput] = []
    url_error: str | None = None
    if reference_urls_text.strip():
        try:
            references.extend(_prepare_reference_urls(reference_urls_text))
        except cli.ReferenceInputError as exc:
            url_error = str(exc)
    references.extend(_uploaded_files_to_references(uploaded_refs or []))

    if url_error:
        st.error(url_error)

    if st.button("Generate images", type="primary"):
        if not description.strip():
            st.error("Please provide a project description before generating images.")
        elif not api_key.strip():
            st.error("Please enter your OpenRouter API key in the sidebar.")
        else:
            ensure_dir_error = None
            try:
                cli.ensure_output_dir(output_dir)
            except OSError as exc:  # pragma: no cover - user environment dependent
                ensure_dir_error = str(exc)
            if ensure_dir_error:
                st.error(f"Failed to create output directory: {ensure_dir_error}")
            else:
                results = []
                with st.spinner("Generating images..."):
                    for designer_name in selected_designers:
                        try:
                            _, image_path = cli.generate_image_for_designer(
                                api_key=api_key.strip(),
                                designer_dir=designer_dir,
                                output_dir=output_dir,
                                description=description,
                                designer_name=designer_name,
                                model=model.strip(),
                                size=size.strip(),
                                prompt_suffix=prompt_suffix,
                                references=references,
                            )
                            results.append((designer_name, image_path))
                        except Exception as exc:  # pragma: no cover - runtime feedback
                            st.error(f"Failed to generate for {designer_name}: {exc}")

                    if custom_prompt.strip():
                        persona_name = custom_name.strip() or "custom"
                        try:
                            full_prompt = cli.build_prompt(
                                description, custom_prompt, prompt_suffix
                            )
                            image_bytes = cli.request_image(
                                api_key.strip(),
                                full_prompt,
                                model.strip(),
                                size.strip(),
                                references=references,
                            )
                            saved_path = cli.save_image(
                                image_bytes, output_dir, persona_name
                            )
                            results.append((persona_name, saved_path))
                        except Exception as exc:  # pragma: no cover - runtime feedback
                            st.error(f"Failed to generate for custom persona: {exc}")

                if results:
                    st.success("Generation complete!")
                    for persona, image_path in results:
                        _display_generated_image(image_path, persona)
                else:
                    st.info(
                        "No images were generated. Ensure you selected a designer or provided a custom persona."
                    )


if __name__ == "__main__":
    run()
