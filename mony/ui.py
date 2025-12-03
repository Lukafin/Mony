"""Streamlit UI for the Mony design persona image generator."""

from __future__ import annotations

import base64
import datetime as dt
import json
import os
import pathlib
import hashlib
from typing import TYPE_CHECKING, Dict, List, Sequence

import streamlit as st

from mony import cli

if TYPE_CHECKING:
    from streamlit.runtime.uploaded_file_manager import UploadedFile


RESEARCH_HISTORY_FILENAME = "research_history.json"
PROMPT_EDIT_PREFIX = "designer_prompt_edit::"
PERSONA_EDITOR_PREFIX = "persona_editor::"
DEFAULT_CREDENTIALS_FILENAME = "ui_credentials.json"
AUTH_STATE_KEY = "_mony_authenticated"


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


def _credentials_file_path() -> pathlib.Path:
    """Return the absolute path where UI credentials are stored."""

    override = os.environ.get("MONY_CREDENTIALS_PATH")
    base = pathlib.Path(override).expanduser() if override else pathlib.Path(DEFAULT_CREDENTIALS_FILENAME)
    return base.expanduser().resolve()


def _load_credentials(path: pathlib.Path) -> Dict[str, str]:
    """Load persisted username/password hash if present."""

    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    username = data.get("username") if isinstance(data, dict) else None
    password_hash = data.get("password_hash") if isinstance(data, dict) else None
    if isinstance(username, str) and isinstance(password_hash, str):
        return {"username": username, "password_hash": password_hash}
    return {}


def _hash_secret(secret: str) -> str:
    """Hash the provided secret for storage."""

    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def _save_credentials(path: pathlib.Path, username: str, password_hash: str) -> None:
    """Persist credentials to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"username": username, "password_hash": password_hash}
    path.write_text(json.dumps(payload, indent=2))


def _clear_credentials(path: pathlib.Path) -> None:
    """Remove stored credentials if present."""

    try:
        path.unlink()
    except FileNotFoundError:
        return


def _require_authentication(credentials: Dict[str, str]) -> None:
    """Prompt for credentials when login protection is enabled."""

    if not credentials:
        st.session_state[AUTH_STATE_KEY] = True
        return

    if st.session_state.get(AUTH_STATE_KEY):
        return

    st.info("Sign in to access the Mony UI.")
    with st.form("mony_login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")

    if submitted:
        if (
            username == credentials.get("username")
            and _hash_secret(password) == credentials.get("password_hash")
        ):
            st.session_state[AUTH_STATE_KEY] = True
            st.success("Signed in successfully.")
            if hasattr(st, "rerun"):
                st.rerun()
            elif hasattr(st, "experimental_rerun"):
                st.experimental_rerun()  # pragma: no cover - legacy Streamlit fallback
        else:
            st.error("Invalid username or password.")
    st.stop()


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

    credentials_path = _credentials_file_path()
    stored_credentials = _load_credentials(credentials_path)

    st.title("Mony – UI Concept Generator")
    st.write(
        "Generate UI concept art by combining a project description with "
        "designer personas and optional reference images."
    )

    _require_authentication(stored_credentials)

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

    research_tab, history_tab, generate_tab, personas_tab, settings_tab = st.tabs(
        [
            "Research personas",
            "Research history",
            "Generate images",
            "Personas",
            "Settings",
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

    with settings_tab:
        st.subheader("Security settings")
        st.write(
            "Set a username and password to require sign-in before anyone uses the UI."
        )
        st.caption(f"Credentials file: `{credentials_path}`")
        current_credentials = _load_credentials(credentials_path)
        if current_credentials:
            st.success(
                f"Login required for user '{current_credentials.get('username', 'unknown')}'."
            )
        else:
            st.info("No credentials saved. Anyone with access to this page can use the UI.")

        settings_username = st.text_input(
            "Username",
            value=current_credentials.get("username", ""),
        )
        settings_password = st.text_input("Password", type="password")
        settings_confirm = st.text_input("Confirm password", type="password")

        if st.button("Save credentials", use_container_width=True):
            username_value = settings_username.strip()
            if not username_value:
                st.error("Enter a username before saving credentials.")
            elif not settings_password:
                st.error("Enter a password before saving credentials.")
            elif settings_password != settings_confirm:
                st.error("Passwords do not match. Re-enter to confirm.")
            else:
                hashed = _hash_secret(settings_password)
                try:
                    _save_credentials(credentials_path, username_value, hashed)
                except OSError as exc:  # pragma: no cover - user environment dependent
                    st.error(f"Failed to save credentials: {exc}")
                else:
                    st.success(
                        "Credentials saved. On the next app reload you will be prompted to sign in."
                    )

        remove_col, logout_col = st.columns(2)
        remove_disabled = not credentials_path.exists()
        if remove_col.button(
            "Remove credentials",
            use_container_width=True,
            disabled=remove_disabled,
        ):
            try:
                _clear_credentials(credentials_path)
            except OSError as exc:  # pragma: no cover - user environment dependent
                st.error(f"Failed to remove credentials: {exc}")
            else:
                st.session_state.pop(AUTH_STATE_KEY, None)
                st.success("Login requirement removed. Reload the page to reflect the change.")

        if st.session_state.get(AUTH_STATE_KEY) and logout_col.button(
            "Log out", use_container_width=True
        ):
            st.session_state.pop(AUTH_STATE_KEY, None)
            st.experimental_rerun()


if __name__ == "__main__":
    run()
