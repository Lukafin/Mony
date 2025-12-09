"""Streamlit UI for the Mony design persona image generator."""

from __future__ import annotations

import base64
import datetime as dt
import json
import os
import pathlib
import uuid
from typing import TYPE_CHECKING, Dict, List, Sequence

import streamlit as st

from mony import cli

if TYPE_CHECKING:
    from streamlit.runtime.uploaded_file_manager import UploadedFile


RESEARCH_HISTORY_FILENAME = "research_history.json"
PROMPT_EDIT_PREFIX = "designer_prompt_edit::"
PERSONA_EDITOR_PREFIX = "persona_editor::"
VOTING_PROMPT_PREFIX = "voting_prompt_edit::"


def _persona_state_key(prefix: str, persona_name: str) -> str:
    """Build a consistent Streamlit session key for persona text areas."""

    return f"{prefix}{persona_name}"


def _designer_prompt_file(designer_dir: pathlib.Path, persona_name: str) -> pathlib.Path:
    """Resolve the markdown file that stores a persona prompt."""

    candidate = designer_dir / f"{persona_name}.md"
    if candidate.exists():
        return candidate
    without_suffix = designer_dir / persona_name
    if without_suffix.exists():
        return without_suffix
    for path in designer_dir.glob("*.md"):
        if path.stem == persona_name:
            return path
    return candidate


def _load_persona_text(designer_dir: pathlib.Path, persona_name: str) -> str:
    """Load the persona prompt text, showing an error if it fails."""

    try:
        return cli.load_designer_prompt(designer_dir, persona_name).prompt
    except cli.DesignerPromptError as exc:  # pragma: no cover - runtime feedback
        st.error(f"Could not load persona '{persona_name}': {exc}")
        return ""


def _set_persona_state(
    designer_dir: pathlib.Path,
    persona_name: str,
    *,
    prefix: str,
    value: str,
) -> str:
    """Persist the latest persona text in Streamlit session state."""

    key = _persona_state_key(prefix, persona_name)
    st.session_state[key] = value
    key_dirs = st.session_state.setdefault("_designer_prompt_key_dirs", {})
    key_dirs[key] = str(designer_dir)
    return key


def _ensure_persona_state(
    designer_dir: pathlib.Path,
    persona_name: str,
    *,
    prefix: str,
) -> str:
    """Ensure a session state entry exists for the given persona."""

    key = _persona_state_key(prefix, persona_name)
    key_dirs = st.session_state.setdefault("_designer_prompt_key_dirs", {})
    current_dir = str(designer_dir)
    if key not in st.session_state or key_dirs.get(key) != current_dir:
        value = _load_persona_text(designer_dir, persona_name)
        _set_persona_state(designer_dir, persona_name, prefix=prefix, value=value)
    return key


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


def _voting_session_dir(output_dir: pathlib.Path) -> pathlib.Path:
    """Return the directory where voting sessions are stored."""

    return output_dir / "voting_sessions"


def _voting_session_file(output_dir: pathlib.Path, session_id: str) -> pathlib.Path:
    """Return the path to a voting session JSON file."""

    return _voting_session_dir(output_dir) / f"{session_id}.json"


def _load_voting_session(
    output_dir: pathlib.Path, session_id: str
) -> Dict[str, object] | None:
    """Load a persisted voting session from disk if available."""

    path = _voting_session_file(output_dir, session_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _save_voting_session(output_dir: pathlib.Path, data: Dict[str, object]) -> None:
    """Persist the voting session JSON to disk."""

    sessions_dir = _voting_session_dir(output_dir)
    cli.ensure_output_dir(sessions_dir)
    session_id = data.get("id") or uuid.uuid4().hex
    data["id"] = session_id
    path = _voting_session_file(output_dir, session_id)
    path.write_text(json.dumps(data, indent=2))


def _display_generated_image(path: pathlib.Path, label: str) -> None:
    """Display an image from disk with a caption inside the Streamlit app."""

    if not path.exists():
        st.warning(f"Generated image for {label} could not be found at {path}.")
        return
    st.markdown(f"**{label}**")
    st.image(str(path))
    st.caption(path.name)


def _research_designer_via_perplexity(
    designer_dir: pathlib.Path,
    persona_name: str,
    api_key: str,
    instructions: str,
) -> tuple[pathlib.Path, str]:
    """Research and persist a trendy designer persona using Perplexity."""

    if not api_key.strip():
        raise cli.DesignerResearchError(
            "Provide a Perplexity API key before running designer research."
        )

    try:
        from perplexityai import Perplexity  # type: ignore
    except ImportError:
        try:  # pragma: no cover - optional dependency
            from perplexity import Perplexity  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise cli.DesignerResearchError(
                "Install the Perplexity client library via 'pip install perplexityai'."
            ) from exc

    client = Perplexity(api_key=api_key.strip())
    prompt_instructions = instructions.strip()
    path = cli.research_trendy_designer_prompt(
        designer_dir,
        persona_name,
        client=client,
        instructions=prompt_instructions or None,
    )
    return path, path.read_text().strip()


def _history_file(designer_dir: pathlib.Path) -> pathlib.Path:
    """Return the path to the research history file for the given designer directory."""

    return designer_dir / RESEARCH_HISTORY_FILENAME


def _load_research_history(designer_dir: pathlib.Path) -> Dict[str, object]:
    """Load persisted research history, falling back to sensible defaults."""

    history_file = _history_file(designer_dir)
    if not history_file.exists():
        return {"runs": [], "settings": {}}

    try:
        data = json.loads(history_file.read_text())
    except Exception:
        return {"runs": [], "settings": {}}

    runs = data.get("runs") if isinstance(data, dict) else []
    settings = data.get("settings") if isinstance(data, dict) else {}
    return {
        "runs": runs if isinstance(runs, list) else [],
        "settings": settings if isinstance(settings, dict) else {},
    }


def _save_research_history(designer_dir: pathlib.Path, history: Dict[str, object]) -> None:
    """Persist research history to the designer directory."""

    cli.ensure_output_dir(designer_dir)
    history_file = _history_file(designer_dir)
    history_file.write_text(json.dumps(history, indent=2))


def _record_research_run(
    designer_dir: pathlib.Path,
    history: Dict[str, object],
    *,
    persona_name: str,
    path: pathlib.Path,
    prompt: str,
) -> None:
    """Add a research run entry and persist the updated history."""

    runs = history.setdefault("runs", [])
    if not isinstance(runs, list):
        runs = []
        history["runs"] = runs

    runs.insert(
        0,
        {
            "name": persona_name,
            "path": str(path),
            "prompt_preview": prompt.strip()[:280],
            "timestamp": dt.datetime.utcnow().isoformat() + "Z",
        },
    )
    history["runs"] = runs[:50]
    _save_research_history(designer_dir, history)


def _calculate_next_monthly_run(history: Dict[str, object]) -> dt.date | None:
    """Return the next due date for monthly research reminders, if enabled."""

    settings = history.get("settings", {}) if isinstance(history, dict) else {}
    if not isinstance(settings, dict) or not settings.get("monthly_enabled"):
        return None

    day = int(settings.get("monthly_day", 1))
    day = max(1, min(day, 28))

    runs = history.get("runs", []) if isinstance(history, dict) else []
    last_run_date = None
    if runs and isinstance(runs, list):
        timestamp = runs[0].get("timestamp") if isinstance(runs[0], dict) else None
        if isinstance(timestamp, str):
            try:
                parsed = dt.datetime.fromisoformat(timestamp.replace("Z", ""))
                last_run_date = parsed.date()
            except ValueError:
                last_run_date = None

    today = dt.date.today()
    base = last_run_date or today
    if base.day >= day:
        month = base.month + 1
        year = base.year + (month - 1) // 12
        month = ((month - 1) % 12) + 1
    else:
        month = base.month
        year = base.year
    return dt.date(year, month, day)


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

    query_params = st.query_params
    session_param = query_params.get("session", [None])[0]

    research_tab, history_tab, generate_tab, voting_tab, personas_tab = st.tabs(
        [
            "Research personas",
            "Research history",
            "Generate images",
            "Voting",
            "Personas",
        ]
    )

    with research_tab:
        st.subheader("Research trendy designer personas")
        research_name = st.text_input(
            "Persona name to research",
            placeholder="e.g., Neo Brutalist Visionary",
        )
        if "_research_history" not in st.session_state or st.session_state.get(
            "_history_dir"
        ) != str(designer_dir):
            st.session_state["_research_history"] = _load_research_history(
                designer_dir
            )
            st.session_state["_history_dir"] = str(designer_dir)

        research_history = st.session_state.get("_research_history", {})
        history_settings = research_history.get("settings", {})
        if not isinstance(history_settings, dict):
            history_settings = {}
            research_history["settings"] = history_settings

        if "research_instructions" not in st.session_state:
            saved_prompt = history_settings.get("saved_prompt")
            st.session_state["research_instructions"] = (
                saved_prompt or cli.DEFAULT_RESEARCH_INSTRUCTIONS
            )
        research_instructions = st.text_area(
            "Research instructions",
            key="research_instructions",
            height=160,
            help=(
                "Customize the Perplexity request used when generating persona prompts. "
                "Placeholders {name} and {year} are substituted automatically."
            ),
        )
        monthly_enabled = st.toggle(
            "Enable monthly persona research reminders",
            value=bool(history_settings.get("monthly_enabled", False)),
            help=(
                "Track when you last refreshed personas and surface the next suggested run."
            ),
        )
        preferred_day = st.number_input(
            "Preferred day of month for research",
            min_value=1,
            max_value=28,
            step=1,
            value=int(history_settings.get("monthly_day", 1)),
            help="Choose when the monthly research task should recur.",
        )
        if (
            monthly_enabled != history_settings.get("monthly_enabled")
            or preferred_day != history_settings.get("monthly_day")
        ):
            history_settings["monthly_enabled"] = monthly_enabled
            history_settings["monthly_day"] = int(preferred_day)
            _save_research_history(designer_dir, research_history)

        next_due = _calculate_next_monthly_run(research_history)
        if monthly_enabled and next_due:
            st.info(
                f"Monthly research enabled. Next suggested run: {next_due.strftime('%Y-%m-%d')}"
            )
        elif monthly_enabled:
            st.info("Monthly research enabled. Run a research task to start the schedule.")
        if "_latest_research" not in st.session_state:
            st.session_state["_latest_research"] = None

        if st.button("Research persona", type="secondary"):
            if not research_name.strip():
                st.error("Enter a persona name before running research.")
            else:
                with st.spinner("Researching latest design trends..."):
                    try:
                        path, text = _research_designer_via_perplexity(
                            designer_dir,
                            research_name.strip(),
                            perplexity_api_key,
                            research_instructions or "",
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
                        _record_research_run(
                            designer_dir,
                            research_history,
                            persona_name=research_name.strip(),
                            path=path,
                            prompt=text,
                        )
                        st.success(
                            f"Saved updated persona prompt to {path.name}. It is now available in the selector on the Generate tab."
                        )

        if st.button("Save prompt as new default", type="secondary"):
            history_settings["saved_prompt"] = research_instructions
            st.session_state["research_instructions"] = research_instructions
            _save_research_history(designer_dir, research_history)
            st.success("Updated default research prompt saved.")

        latest_research = st.session_state.get("_latest_research")
        if latest_research:
            with st.expander(
                f"Latest researched persona: {latest_research['name']}", expanded=False
            ):
                st.markdown(
                    f"**Stored at:** `{latest_research['path']}`\n\n"
                    "Review the generated prompt below. Edit directly in the file if you need tweaks."
                )
                st.text_area(
                    "Persona prompt",
                    value=latest_research["text"],
                    height=200,
                    disabled=True,
                )

    available_designers = _load_available_designers(designer_dir)

    with history_tab:
        st.subheader("Recent research runs")
        research_history = st.session_state.get("_research_history", {})
        runs = research_history.get("runs", []) if isinstance(research_history, dict) else []
        if runs:
            for run in runs:
                if not isinstance(run, dict):
                    continue
                timestamp = run.get("timestamp", "Unknown time")
                title = run.get("name", "Unnamed persona")
                with st.expander(f"{title} – {timestamp}"):
                    st.markdown(f"**Stored at:** `{run.get('path', 'unknown')}`")
                    persona_text = ""
                    path_value = run.get("path")
                    if isinstance(path_value, str) and path_value.strip():
                        persona_path = pathlib.Path(path_value)
                        if persona_path.exists():
                            try:
                                persona_text = persona_path.read_text().strip()
                            except OSError as exc:  # pragma: no cover - user environment dependent
                                st.error(f"Failed to read persona file: {exc}")
                        else:
                            st.warning("Persona file no longer exists on disk.")
                    if not persona_text:
                        persona_text = run.get("prompt_preview", "")
                    if persona_text:
                        st.text_area(
                            "Persona prompt",
                            value=persona_text,
                            height=160,
                            disabled=True,
                        )
        else:
            st.info("No research runs recorded yet. Run a persona research task to populate history.")

        st.subheader("Available designer personas")
        if available_designers:
            st.write(", ".join(available_designers))
        else:
            st.write("No personas found in the configured designer directory.")

    with generate_tab:
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
                "Custom persona prompt",
                value="",
                height=160,
                help=(
                    "Provide your own persona instructions. The project brief and optional "
                    "suffix will be appended automatically."
                ),
            )

        selected_prompt_overrides: Dict[str, str] = {}
        if selected_designers:
            st.caption(
                "Expand a persona below to preview or tweak its prompt for this generation."
            )
            for designer_name in selected_designers:
                prompt_state_key = _ensure_persona_state(
                    designer_dir, designer_name, prefix=PROMPT_EDIT_PREFIX
                )
                with st.expander(f"{designer_name} prompt", expanded=False):
                    st.text_area(
                        "Persona prompt",
                        key=prompt_state_key,
                        height=180,
                        label_visibility="collapsed",
                        help=(
                            "Changes here only affect the current run. Use the Personas tab to save edits."
                        ),
                    )
                selected_prompt_overrides[designer_name] = st.session_state.get(
                    prompt_state_key, ""
                )

        st.subheader("Reference images")
        uploaded_refs = st.file_uploader(
            "Upload reference images",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
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
                            prompt_override = selected_prompt_overrides.get(
                                designer_name
                            )
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
                                    prompt_override=prompt_override,
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

    with voting_tab:
        st.subheader("Persona variations voting")
        st.write(
            "Create small prompt variations for multiple personas, generate images, "
            "and collect quick votes from collaborators."
        )

        voting_description = st.text_area(
            "Project brief for this voting round",
            height=120,
            key="voting_description",
        )

        selected_voting_personas = st.multiselect(
            "Choose personas to compare (up to 3)",
            available_designers,
            max_selections=3,
            key="voting_persona_selector",
        )

        if selected_voting_personas:
            st.markdown("#### Tweak prompts for each variation")
            for persona_name in selected_voting_personas:
                editor_key = _ensure_persona_state(
                    designer_dir, persona_name, prefix=VOTING_PROMPT_PREFIX
                )
                st.text_area(
                    f"Prompt for {persona_name}",
                    key=editor_key,
                    height=140,
                )
        else:
            st.info("Select personas to prepare voting variations.")

        create_col, stop_col = st.columns(2)
        generated_session_id = st.session_state.get("_active_voting_session")

        if create_col.button(
            "Create variations and start voting", type="primary", use_container_width=True
        ):
            if not selected_voting_personas:
                st.error("Select at least one persona to create variations.")
            elif not voting_description.strip():
                st.error("Provide a project brief before generating variations.")
            else:
                results: List[Dict[str, object]] = []
                votes: Dict[str, int] = {}
                with st.spinner("Generating persona variations..."):
                    for persona_name in selected_voting_personas:
                        prompt_key = _persona_state_key(
                            VOTING_PROMPT_PREFIX, persona_name
                        )
                        prompt_text = st.session_state.get(prompt_key, "")
                        try:
                            full_prompt = cli.build_prompt(
                                voting_description, prompt_text, prompt_suffix
                            )
                            image_bytes = cli.request_image(
                                api_key.strip(),
                                full_prompt,
                                model.strip(),
                                size.strip(),
                                references=[],
                            )
                            image_path = cli.save_image(
                                image_bytes, output_dir, f"voting-{persona_name}"
                            )
                        except Exception as exc:  # pragma: no cover - runtime feedback
                            st.error(f"Failed to generate for {persona_name}: {exc}")
                            continue

                        votes[persona_name] = 0
                        results.append(
                            {
                                "persona": persona_name,
                                "prompt": prompt_text,
                                "image": str(image_path),
                            }
                        )

                if results:
                    session_data: Dict[str, object] = {
                        "id": uuid.uuid4().hex,
                        "status": "open",
                        "created": dt.datetime.utcnow().isoformat() + "Z",
                        "description": voting_description,
                        "variants": results,
                        "votes": votes,
                    }
                    _save_voting_session(output_dir, session_data)
                    generated_session_id = session_data["id"]
                    st.session_state["_active_voting_session"] = generated_session_id
                    st.experimental_set_query_params(session=generated_session_id)
                    st.success("Voting session created. Share the link below.")
                    share_link = f"?session={generated_session_id}"
                    st.code(share_link, language="text")
                else:
                    st.info("No images were generated for this voting round.")

        active_session_id = session_param or generated_session_id
        active_session = (
            _load_voting_session(output_dir, active_session_id)
            if active_session_id
            else None
        )

        if active_session:
            st.markdown(f"#### Current session: `{active_session['id']}`")
            st.caption(f"Status: {active_session.get('status', 'unknown')}")
            variants = active_session.get("variants", [])
            votes = active_session.get("votes", {}) if isinstance(active_session, dict) else {}
            if not variants:
                st.warning("This session has no stored variants.")
            else:
                voted_key = f"voted::{active_session['id']}"
                has_voted = st.session_state.get(voted_key)
                for variant in variants:  # type: ignore[assignment]
                    if not isinstance(variant, dict):
                        continue
                    persona_label = variant.get("persona", "unknown")
                    st.markdown(f"**{persona_label}**")
                    image_path = variant.get("image")
                    if isinstance(image_path, str) and pathlib.Path(image_path).exists():
                        st.image(str(image_path))
                    prompt_text = variant.get("prompt", "")
                    if prompt_text:
                        st.caption(prompt_text)
                    if active_session.get("status") == "open":
                        if not has_voted and st.button(
                            f"Vote for {persona_label}", key=f"vote::{persona_label}"
                        ):
                            votes[persona_label] = int(votes.get(persona_label, 0)) + 1
                            active_session["votes"] = votes
                            _save_voting_session(output_dir, active_session)
                            st.session_state[voted_key] = persona_label
                            st.success("Vote recorded! Share the link so others can vote.")
                    else:
                        st.info("Voting closed.")

                if active_session.get("status") == "closed" or st.session_state.get(voted_key):
                    st.markdown("#### Live results")
                    vote_items = sorted(
                        votes.items(), key=lambda item: item[1], reverse=True
                    )
                    for persona_label, count in vote_items:
                        st.write(f"{persona_label}: **{count}** votes")
        else:
            st.info(
                "Create a new voting session or open a shared link with ?session=<id> to view voting."
            )

        if active_session and active_session.get("status") == "open":
            if stop_col.button(
                "Stop voting and show results", use_container_width=True
            ):
                active_session["status"] = "closed"
                _save_voting_session(output_dir, active_session)
                st.experimental_set_query_params(session=active_session["id"])
                st.success("Voting stopped. Results are now visible to everyone with the link.")

    with personas_tab:
        st.subheader("Persona library")
        st.write("Inspect and edit the saved persona prompt files in your designer directory.")
        if not available_designers:
            st.info(
                "No personas available. Research or add personas first, then return to edit them."
            )
        else:
            persona_to_edit = st.selectbox(
                "Choose a persona",
                available_designers,
                key="persona_editor_selector",
                help="Select which persona markdown file to edit.",
            )
            if persona_to_edit:
                editor_key = _ensure_persona_state(
                    designer_dir, persona_to_edit, prefix=PERSONA_EDITOR_PREFIX
                )
                persona_file = _designer_prompt_file(designer_dir, persona_to_edit)
                st.caption(f"File: `{persona_file}`")
                st.text_area(
                    "Persona prompt text",
                    key=editor_key,
                    height=240,
                    label_visibility="collapsed",
                )
                save_col, reload_col = st.columns(2)
                if save_col.button(
                    "Save persona",
                    key=f"save_persona_{persona_to_edit}",
                    use_container_width=True,
                ):
                    updated_text = st.session_state.get(editor_key, "")
                    try:
                        cli.ensure_output_dir(designer_dir)
                        content = updated_text
                        if content and not content.endswith("\n"):
                            content += "\n"
                        persona_file.write_text(content)
                    except OSError as exc:  # pragma: no cover - user environment dependent
                        st.error(f"Failed to save persona: {exc}")
                    else:
                        _set_persona_state(
                            designer_dir,
                            persona_to_edit,
                            prefix=PERSONA_EDITOR_PREFIX,
                            value=updated_text,
                        )
                        prompt_key = _persona_state_key(
                            PROMPT_EDIT_PREFIX, persona_to_edit
                        )
                        if prompt_key in st.session_state:
                            _set_persona_state(
                                designer_dir,
                                persona_to_edit,
                                prefix=PROMPT_EDIT_PREFIX,
                                value=updated_text,
                            )
                        st.success(f"Saved updates to {persona_file.name}.")
                if reload_col.button(
                    "Reload from file",
                    key=f"reload_persona_{persona_to_edit}",
                    use_container_width=True,
                ):
                    refreshed_text = _load_persona_text(designer_dir, persona_to_edit)
                    _set_persona_state(
                        designer_dir,
                        persona_to_edit,
                        prefix=PERSONA_EDITOR_PREFIX,
                        value=refreshed_text,
                    )
                    prompt_key = _persona_state_key(PROMPT_EDIT_PREFIX, persona_to_edit)
                    if prompt_key in st.session_state:
                        _set_persona_state(
                            designer_dir,
                            persona_to_edit,
                            prefix=PROMPT_EDIT_PREFIX,
                            value=refreshed_text,
                        )
                    st.info("Persona reloaded from disk.")


if __name__ == "__main__":
    run()
