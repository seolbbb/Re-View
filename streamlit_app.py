from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from src.adk_pipeline.paths import DEFAULT_OUTPUT_BASE, sanitize_video_name
from src.services.pipeline_service import (
    build_adk_state,
    get_default_output_base,
    run_pre_adk_pipeline,
    send_adk_message,
    start_adk_session,
)

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "data" / "inputs"
DEFAULT_OUTPUT_BASE_STR = str(DEFAULT_OUTPUT_BASE)
ADK_OUTPUT_BASE = get_default_output_base()
VIDEO_EXTENSIONS = [".mp4", ".mov", ".mkv", ".avi"]


def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _save_upload(uploaded_file) -> Path:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    stem = Path(uploaded_file.name).stem or "video"
    safe_stem = sanitize_video_name(stem)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = INPUT_DIR / f"{timestamp}_{safe_stem}{suffix}"
    destination.write_bytes(uploaded_file.getbuffer())
    return destination


@st.cache_data(show_spinner=False)
def _load_video_bytes(path: str) -> bytes:
    return Path(path).read_bytes()


def _load_manifest(video_root: Path) -> List[Dict[str, Any]]:
    payload = _read_json(video_root / "manifest.json")
    if not isinstance(payload, list):
        return []
    return payload


def _list_output_names(output_base: Path) -> List[str]:
    if not output_base.exists():
        return []
    return sorted([path.name for path in output_base.iterdir() if path.is_dir()])


def _find_existing_video(video_name: str) -> Optional[Path]:
    if not video_name:
        return None
    for base_dir in (INPUT_DIR,):
        if not base_dir.exists():
            continue
        for ext in VIDEO_EXTENSIONS:
            candidate = base_dir / f"{video_name}{ext}"
            if candidate.exists():
                return candidate
        for path in base_dir.glob(f"{video_name}.*"):
            if path.suffix.lower() in VIDEO_EXTENSIONS:
                return path
    return None


def _has_pre_adk_outputs(video_root: Path) -> bool:
    return (
        (video_root / "stt.json").exists()
        and (video_root / "manifest.json").exists()
        and (video_root / "captures").exists()
    )


def _resolve_video_name(
    video_path: Optional[Path],
    video_root_value: Optional[str],
) -> Optional[str]:
    if video_path:
        return sanitize_video_name(video_path.stem)
    if video_root_value:
        return Path(video_root_value).name
    return None


def _render_final_summaries(video_root: Path, *, scroll_height: int = 600) -> None:
    summaries_dir = video_root / "fusion" / "outputs"
    summary_paths = sorted(summaries_dir.glob("final_summary_*.md")) if summaries_dir.exists() else []
    if not summary_paths:
        st.info("No final summaries found.")
        return
    scroll_container = st.container(height=scroll_height)
    with scroll_container:
        for path in summary_paths:
            content = _read_text(path)
            if not content:
                continue
            st.markdown(f"#### {path.name}")
            st.markdown(content)
            st.markdown("---")


def _render_segment_summaries(video_root: Path) -> None:
    output_dir = video_root / "fusion"
    candidates = [
        output_dir / "segment_summaries.md",
        output_dir / "segment_summaries_nl.md",
    ]
    shown = False
    for path in candidates:
        content = _read_text(path)
        if not content:
            continue
        shown = True
        with st.expander(path.name, expanded=False):
            st.markdown(content)
    if not shown:
        st.info("No segment summaries found.")


def _extract_manifest_timestamp_ms(item: Dict[str, Any]) -> Optional[int]:
    for key in ("timestamp_ms", "start_ms", "start"):
        if key not in item:
            continue
        try:
            return int(item[key])
        except (TypeError, ValueError):
            continue
    return None


def _render_captures(video_root: Path) -> None:
    manifest = _load_manifest(video_root)
    if not manifest:
        st.info("No capture manifest found.")
        return

    vlm_payload = _read_json(video_root / "vlm.json")
    vlm_lookup: Dict[int, str] = {}
    if isinstance(vlm_payload, dict):
        for item in vlm_payload.get("items", []):
            try:
                timestamp = int(item.get("timestamp_ms"))
            except (TypeError, ValueError):
                continue
            text = item.get("extracted_text")
            if isinstance(text, str) and text.strip():
                vlm_lookup[timestamp] = text.strip()

    manifest = sorted(
        manifest,
        key=lambda item: _extract_manifest_timestamp_ms(item) or 0,
    )
    max_items = st.slider(
        "Max captures",
        min_value=1,
        max_value=len(manifest),
        value=min(24, len(manifest)),
    )
    show_vlm = st.checkbox("Show VLM text above captures", value=True)
    show_full_vlm = False
    if show_vlm:
        show_full_vlm = st.checkbox("Show full VLM text (Markdown)", value=False)
    columns = st.slider("Columns", min_value=2, max_value=5, value=3)
    cols = st.columns(columns)

    for idx, item in enumerate(manifest[:max_items]):
        file_name = str(item.get("file_name", "")).strip()
        if not file_name:
            continue
        image_path = video_root / "captures" / file_name
        if not image_path.exists():
            continue
        timestamp_ms = _extract_manifest_timestamp_ms(item)
        caption = item.get("timestamp_human")
        if not caption and timestamp_ms is not None:
            caption = f"{timestamp_ms / 1000:.2f}s"
        if not caption:
            caption = file_name
        with cols[idx % columns]:
            if show_vlm:
                vlm_text = vlm_lookup.get(timestamp_ms) if timestamp_ms is not None else None
                if vlm_text:
                    if show_full_vlm:
                        st.markdown(vlm_text)
                    else:
                        preview = " ".join(vlm_text.split())
                        preview = preview[:200] + ("..." if len(preview) > 200 else "")
                        st.caption(preview)
                else:
                    st.caption("VLM: (no text)")
            st.image(str(image_path), caption=caption, width="stretch")

    graph_path = next(video_root.glob("*_scene_analysis.png"), None)
    if graph_path:
        st.markdown("---")
        st.markdown("#### Scene analysis")
        st.image(str(graph_path), width="stretch")


def main() -> None:
    st.set_page_config(page_title="Screentime Pipeline", layout="wide")
    st.title("Screentime Video Pipeline")
    st.toggle("Chat panel", value=True, key="show_chat")

    if "video_root" not in st.session_state:
        st.session_state.video_root = None
    if "run_meta" not in st.session_state:
        st.session_state.run_meta = None
    if "selected_video" not in st.session_state:
        st.session_state.selected_video = None
    if "uploaded_video_path" not in st.session_state:
        st.session_state.uploaded_video_path = None
    if "uploaded_video_signature" not in st.session_state:
        st.session_state.uploaded_video_signature = None
    if "pre_adk_status" not in st.session_state:
        st.session_state.pre_adk_status = "idle"
    if "pre_adk_error" not in st.session_state:
        st.session_state.pre_adk_error = None
    if "pre_adk_signature" not in st.session_state:
        st.session_state.pre_adk_signature = None
    if "pre_adk_result" not in st.session_state:
        st.session_state.pre_adk_result = None
    if "adk_session" not in st.session_state:
        st.session_state.adk_session = None
    if "adk_state_signature" not in st.session_state:
        st.session_state.adk_state_signature = None
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "adk_busy" not in st.session_state:
        st.session_state.adk_busy = False
    if "selected_output_name" not in st.session_state:
        st.session_state.selected_output_name = None
    if "preview_video_path" not in st.session_state:
        st.session_state.preview_video_path = None

    source = st.radio(
        "Video source",
        ["Upload", "Local path", "Existing output"],
        horizontal=True,
    )
    video_path: Optional[Path] = None

    if source == "Upload":
        uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "mkv", "avi"])
        if uploaded:
            signature = (uploaded.name, uploaded.size)
            if st.session_state.uploaded_video_signature != signature:
                saved_path = _save_upload(uploaded)
                st.session_state.uploaded_video_path = str(saved_path)
                st.session_state.uploaded_video_signature = signature
            if st.session_state.uploaded_video_path:
                video_path = Path(st.session_state.uploaded_video_path)
    elif source == "Local path":
        path_str = st.text_input("Video path", value="")
        if path_str:
            candidate = Path(path_str).expanduser().resolve()
            if candidate.exists():
                video_path = candidate
            else:
                st.error("Video path does not exist.")
    else:
        available_outputs = _list_output_names(ADK_OUTPUT_BASE)
        if not available_outputs:
            st.info("No outputs found yet. Upload a video to create outputs.")
        else:
            selected_output = st.selectbox(
                "Select existing output",
                options=[""] + available_outputs,
                index=0,
            )
            if st.button("Load selected output", disabled=not selected_output):
                candidate = ADK_OUTPUT_BASE / selected_output
                if candidate.exists():
                    st.session_state.video_root = str(candidate)
                    st.session_state.run_meta = _read_json(candidate / "pipeline_run.json")
                    st.session_state.pre_adk_status = "done"
                    st.session_state.pre_adk_error = None
                    st.session_state.pre_adk_signature = None
                    st.session_state.pre_adk_result = None
                    st.session_state.adk_session = None
                    st.session_state.adk_state_signature = None
                    st.session_state.chat_messages = []
                    st.session_state.selected_output_name = selected_output
                    preview_candidate = _find_existing_video(selected_output)
                    st.session_state.preview_video_path = (
                        str(preview_candidate) if preview_candidate else None
                    )
                else:
                    st.error("Selected output folder does not exist.")

    if video_path:
        if str(video_path) != st.session_state.selected_video:
            st.session_state.selected_video = str(video_path)
            st.session_state.video_root = None
            st.session_state.run_meta = None
            st.session_state.pre_adk_status = "idle"
            st.session_state.pre_adk_error = None
            st.session_state.pre_adk_signature = None
            st.session_state.pre_adk_result = None
            st.session_state.adk_session = None
            st.session_state.adk_state_signature = None
            st.session_state.chat_messages = []
        st.session_state.preview_video_path = str(video_path)

    with st.sidebar:
        st.header("Pipeline options")
        st.subheader("Pre-ADK")
        st.text_input("Output base", value=DEFAULT_OUTPUT_BASE_STR, disabled=True)
        st.caption("ADK output base is fixed to data/outputs.")
        stt_backend = st.selectbox("STT backend", ["clova"])
        parallel = st.checkbox("Parallel STT + capture", value=True)
        capture_threshold = st.number_input("Capture threshold", min_value=0.1, value=3.0, step=0.1)
        capture_dedupe_threshold = st.number_input(
            "Capture dedupe threshold",
            min_value=0.1,
            value=3.0,
            step=0.1,
        )
        capture_min_interval = st.number_input("Capture min interval (sec)", min_value=0.1, value=0.5, step=0.1)
        rerun_pre_adk = st.button("Rerun Pre-ADK", disabled=video_path is None)

        st.markdown("---")
        st.subheader("ADK options")
        force_preprocessing = st.checkbox("Force preprocessing (VLM/Sync)", value=False)
        max_reruns = st.number_input("Max reruns", min_value=0, value=2, step=1)
        vlm_batch_size: Optional[int] = None
        if st.checkbox("Set VLM batch size", value=False):
            vlm_batch_size = int(st.number_input("VLM batch size", min_value=1, value=2, step=1))
        vlm_concurrency = st.number_input("VLM concurrency", min_value=1, value=3, step=1)
        vlm_show_progress = st.checkbox("VLM show progress", value=True)
        judge_min_score = st.number_input("Judge min score", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
        if st.button("Reset ADK session", disabled=st.session_state.adk_session is None):
            st.session_state.adk_session = None
            st.session_state.adk_state_signature = None
            st.session_state.chat_messages = []

        st.markdown("---")
        st.subheader("Existing outputs")
        if st.button("Load outputs for selected video", disabled=video_path is None):
            if not video_path:
                st.error("Select a video first.")
            else:
                candidate = ADK_OUTPUT_BASE / sanitize_video_name(video_path.stem)
                if candidate.exists():
                    st.session_state.video_root = str(candidate)
                    st.session_state.run_meta = _read_json(candidate / "pipeline_run.json")
                    st.session_state.pre_adk_status = "done"
                    st.session_state.preview_video_path = str(video_path)
                else:
                    st.warning("No outputs found for selected video.")

        existing_output_path = st.text_input("Output folder path", value="")
        if st.button("Load output folder"):
            if not existing_output_path:
                st.error("Enter an output folder path.")
            else:
                candidate = Path(existing_output_path).expanduser().resolve()
                if candidate.exists():
                    st.session_state.video_root = str(candidate)
                    st.session_state.run_meta = _read_json(candidate / "pipeline_run.json")
                    if _has_pre_adk_outputs(candidate):
                        st.session_state.pre_adk_status = "done"
                    preview_candidate = _find_existing_video(candidate.name)
                    st.session_state.preview_video_path = (
                        str(preview_candidate) if preview_candidate else None
                    )
                else:
                    st.error("Output folder does not exist.")

    video_root_value = st.session_state.video_root
    video_name = _resolve_video_name(video_path, video_root_value)

    if video_path:
        pre_adk_signature: Tuple[str] = (str(video_path),)
        should_run_pre_adk = st.session_state.pre_adk_signature != pre_adk_signature
        if rerun_pre_adk:
            should_run_pre_adk = True

        if should_run_pre_adk:
            st.session_state.pre_adk_status = "running"
            st.session_state.pre_adk_error = None
            st.session_state.adk_session = None
            st.session_state.adk_state_signature = None
            st.session_state.chat_messages = []
            with st.spinner("Running Pre-ADK..."):
                try:
                    result = run_pre_adk_pipeline(
                        video_path=video_path,
                        output_base=ADK_OUTPUT_BASE,
                        stt_backend=stt_backend,
                        parallel=parallel,
                        capture_threshold=capture_threshold,
                        capture_dedupe_threshold=capture_dedupe_threshold,
                        capture_min_interval=capture_min_interval,
                        force=rerun_pre_adk,
                    )
                except Exception as exc:
                    st.session_state.pre_adk_status = "error"
                    st.session_state.pre_adk_error = str(exc)
                else:
                    st.session_state.pre_adk_status = "done"
                    st.session_state.pre_adk_signature = pre_adk_signature
                    st.session_state.pre_adk_result = result
                    st.session_state.video_root = result.get("video_root")
                    video_root_value = st.session_state.video_root

    if (
        st.session_state.pre_adk_status == "idle"
        and video_root_value
        and _has_pre_adk_outputs(Path(video_root_value))
    ):
        st.session_state.pre_adk_status = "done"

    if video_name and st.session_state.pre_adk_status == "done":
        adk_state = build_adk_state(
            video_name=video_name,
            force_preprocessing=force_preprocessing,
            max_reruns=int(max_reruns),
            vlm_batch_size=vlm_batch_size,
            vlm_concurrency=int(vlm_concurrency),
            vlm_show_progress=bool(vlm_show_progress),
            judge_min_score=float(judge_min_score),
        )
        adk_state_signature = (
            video_name,
            force_preprocessing,
            int(max_reruns),
            vlm_batch_size,
            int(vlm_concurrency),
            bool(vlm_show_progress),
            float(judge_min_score),
        )
        if st.session_state.adk_session is None:
            st.session_state.adk_session = start_adk_session(state=adk_state)
            st.session_state.adk_state_signature = adk_state_signature
    else:
        adk_state_signature = None

    if st.session_state.pre_adk_status == "running":
        st.info("Pre-ADK is running. This can take a while for long videos.")
    elif st.session_state.pre_adk_status == "error":
        st.error(f"Pre-ADK failed: {st.session_state.pre_adk_error}")
    elif st.session_state.pre_adk_status == "done":
        st.success("Pre-ADK completed.")

    if st.session_state.show_chat:
        main_col, chat_col = st.columns([2, 1], gap="large")
    else:
        main_col = st.container()
        chat_col = None

    with main_col:
        if video_path or video_root_value:
            st.markdown("---")
            st.markdown("### Video")
            preview_path_value = st.session_state.preview_video_path
            preview_path = Path(preview_path_value) if preview_path_value else None
            if preview_path:
                st.video(_load_video_bytes(str(preview_path)))
                st.caption(f"Selected: {preview_path}")
            else:
                st.info("No video selected.")

            if video_root_value:
                st.markdown("---")
                summary_height = st.slider(
                    "Summary height",
                    min_value=320,
                    max_value=900,
                    value=600,
                    step=20,
                )
                _render_final_summaries(
                    Path(video_root_value),
                    scroll_height=summary_height,
                )
            else:
                st.info("Run pipeline or load outputs to see summaries.")

        if video_root_value:
            video_root = Path(video_root_value)
            st.markdown(f"Outputs: `{video_root}`")
            tabs = st.tabs(["Segment summaries", "Captures"])
            with tabs[0]:
                _render_segment_summaries(video_root)
            with tabs[1]:
                _render_captures(video_root)

    if chat_col:
        with chat_col:
            st.markdown("### ADK Chat")
            if not video_name:
                st.info("Upload a video or load outputs to start.")
                return
            if st.session_state.pre_adk_status == "running":
                st.info("Pre-ADK is running. Chat will be ready after it finishes.")
                return
            if st.session_state.pre_adk_status == "error":
                st.error("Pre-ADK failed. Fix the error and rerun.")
                return
            if st.session_state.adk_session is None:
                st.info("ADK session is not ready yet.")
                return

            if (
                adk_state_signature
                and st.session_state.adk_state_signature
                and adk_state_signature != st.session_state.adk_state_signature
            ):
                st.warning("ADK settings changed. Reset session to apply them.")

            if st.session_state.adk_busy:
                st.info("ADK is running. Please wait for the current run to finish.")

            if st.button(
                "Run pipeline",
                disabled=st.session_state.adk_session is None or st.session_state.adk_busy,
            ):
                st.session_state.adk_busy = True
                message = f"{video_name}로 파이프라인 실행해줘"
                st.session_state.chat_messages.append({"role": "user", "content": message})
                try:
                    with st.spinner("Running ADK pipeline..."):
                        responses = send_adk_message(st.session_state.adk_session, message)
                    for response in responses:
                        st.session_state.chat_messages.append(
                            {
                                "role": "assistant",
                                "author": response.author,
                                "content": response.text,
                            }
                        )
                except Exception as exc:
                    st.session_state.chat_messages.append(
                        {
                            "role": "assistant",
                            "author": "system",
                            "content": f"ADK error: {exc}",
                        }
                    )
                finally:
                    st.session_state.adk_busy = False
                st.rerun()

            chat_container = st.container(height=520)
            with chat_container:
                for message in st.session_state.chat_messages:
                    role = message.get("role", "assistant")
                    with st.chat_message(role):
                        author = message.get("author")
                        if author:
                            st.caption(author)
                        st.markdown(message.get("content", ""))

            prompt = st.chat_input("Message ADK", disabled=st.session_state.adk_busy)
            if prompt and not st.session_state.adk_busy:
                st.session_state.adk_busy = True
                st.session_state.chat_messages.append({"role": "user", "content": prompt})
                try:
                    with st.spinner("Waiting for ADK..."):
                        responses = send_adk_message(st.session_state.adk_session, prompt)
                    for response in responses:
                        st.session_state.chat_messages.append(
                            {
                                "role": "assistant",
                                "author": response.author,
                                "content": response.text,
                            }
                        )
                except Exception as exc:
                    st.session_state.chat_messages.append(
                        {
                            "role": "assistant",
                            "author": "system",
                            "content": f"ADK error: {exc}",
                        }
                    )
                finally:
                    st.session_state.adk_busy = False
                st.rerun()


if __name__ == "__main__":
    main()
