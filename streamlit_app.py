from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from src.run_video_pipeline import run_pipeline, _sanitize_video_name

ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "data" / "uploads"
DEFAULT_OUTPUT_BASE = "data/outputs"


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
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    stem = Path(uploaded_file.name).stem or "video"
    safe_stem = _sanitize_video_name(stem)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = UPLOAD_DIR / f"{timestamp}_{safe_stem}{suffix}"
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

    manifest = sorted(manifest, key=lambda item: int(item.get("timestamp_ms", 0)))
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
        caption = item.get("timestamp_human") or file_name
        with cols[idx % columns]:
            if show_vlm:
                vlm_text = vlm_lookup.get(int(item.get("timestamp_ms", -1)))
                if vlm_text:
                    if show_full_vlm:
                        st.markdown(vlm_text)
                    else:
                        preview = " ".join(vlm_text.split())
                        preview = preview[:200] + ("…" if len(preview) > 200 else "")
                        st.caption(preview)
                else:
                    st.caption("VLM: (no text)")
            st.image(str(image_path), caption=caption, width="stretch")

    graph_path = next(video_root.glob("*_scene_analysis.png"), None)
    if graph_path:
        st.markdown("---")
        st.markdown("#### Scene analysis")
        st.image(str(graph_path), width="stretch")


def _render_run_meta(video_root: Path, run_meta: Optional[Dict[str, Any]]) -> None:
    if run_meta:
        st.json(run_meta)
        return

    payload = _read_json(video_root / "pipeline_run.json")
    if payload is None:
        st.info("No pipeline metadata found.")
        return
    st.json(payload)


def main() -> None:
    st.set_page_config(page_title="Screentime Pipeline", layout="wide")
    st.title("Screentime Video Pipeline")

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

    source = st.radio("Video source", ["Upload", "Local path"], horizontal=True)
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
    else:
        path_str = st.text_input("Video path", value="")
        if path_str:
            candidate = Path(path_str).expanduser().resolve()
            if candidate.exists():
                video_path = candidate
            else:
                st.error("Video path does not exist.")

    if video_path:
        if str(video_path) != st.session_state.selected_video:
            st.session_state.selected_video = str(video_path)
            st.session_state.video_root = None
            st.session_state.run_meta = None

    with st.sidebar:
        st.header("Pipeline options")
        output_base = st.text_input("Output base", value=DEFAULT_OUTPUT_BASE)
        output_base_path = Path(output_base)
        if not output_base_path.is_absolute():
            output_base_path = (ROOT / output_base_path).resolve()
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

        vlm_batch_size: Optional[int] = None
        if st.checkbox("Set VLM batch size", value=False):
            vlm_batch_size = int(st.number_input("VLM batch size", min_value=1, value=1, step=1))

        limit: Optional[int] = None
        if st.checkbox("Limit fusion segments", value=False):
            limit = int(st.number_input("Segment limit", min_value=1, value=10, step=1))

        dry_run = st.checkbox("Dry run (skip LLM summarizer)", value=False)

        st.markdown("---")
        st.subheader("Existing outputs")
        if st.button("Load outputs for selected video", disabled=video_path is None):
            if not video_path:
                st.error("Select a video first.")
            else:
                candidate = output_base_path / _sanitize_video_name(video_path.stem)
                if candidate.exists():
                    st.session_state.video_root = str(candidate)
                    st.session_state.run_meta = _read_json(candidate / "pipeline_run.json")
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
                else:
                    st.error("Output folder does not exist.")

    run_clicked = st.button("Run pipeline", type="primary", disabled=video_path is None)
    if run_clicked:
        if not video_path:
            st.error("Select a video first.")
        else:
            with st.spinner("Running pipeline..."):
                try:
                    video_root, run_meta = run_pipeline(
                        video=video_path,
                        output_base=output_base,
                        stt_backend=stt_backend,
                        parallel=parallel,
                        capture_threshold=capture_threshold,
                        capture_dedupe_threshold=capture_dedupe_threshold,
                        capture_min_interval=capture_min_interval,
                        capture_verbose=False,
                        vlm_batch_size=vlm_batch_size,
                        limit=limit,
                        dry_run=dry_run,
                    )
                except Exception as exc:
                    st.error(f"Pipeline failed: {exc}")
                else:
                    st.session_state.video_root = str(video_root)
                    st.session_state.run_meta = run_meta
                    st.success("Pipeline finished.")

    video_root_value = st.session_state.video_root
    if video_path or video_root_value:
        st.markdown("---")
        video_ratio = st.slider(
            "Layout ratio (video : summary)",
            min_value=1,
            max_value=5,
            value=3,
            step=1,
        )
        summary_ratio = 6 - video_ratio
        video_col, summary_col = st.columns([video_ratio, summary_ratio], gap="large")

        with video_col:
            st.markdown("### Video")
            if video_path:
                st.video(_load_video_bytes(str(video_path)))
                st.caption(f"Selected: {video_path}")
            else:
                st.info("No video selected.")

        with summary_col:
            if video_root_value:
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
        tabs = st.tabs(["Segment summaries", "Captures", "Run meta"])
        with tabs[0]:
            _render_segment_summaries(video_root)
        with tabs[1]:
            _render_captures(video_root)
        with tabs[2]:
            _render_run_meta(video_root, st.session_state.run_meta)


if __name__ == "__main__":
    main()
