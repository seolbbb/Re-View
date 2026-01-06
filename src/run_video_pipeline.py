"""동영상 1개 입력 → STT/Capture/VLM → Fusion 요약까지 end-to-end 실행."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.audio.speech_client import ClovaSpeechClient
from src.audio.preprocess_audio import preprocess_audio
from src.capture.video_processor import VideoProcessor
from src.fusion.config import load_config
from src.fusion.final_summary_composer import compose_final_summaries
from src.fusion.io_utils import ensure_output_root
from src.fusion.renderer import render_segment_summaries_md
from src.fusion.summarizer import run_summarizer
from src.fusion.sync_engine import run_sync_engine
from src.vlm.vlm_engine import OpenRouterVlmExtractor, write_vlm_raw_json
from src.vlm.vlm_fusion import convert_vlm_raw_to_fusion_vlm
from src.vlm.qwen3_detect import DEFAULT_DETECT_PROMPT, OpenRouterQwen3Detector

DETECT_SCHEMA_VERSION = 1


def _sanitize_video_name(stem: str) -> str:
    value = stem.strip()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^A-Za-z0-9가-힣._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._-")
    if not value:
        return "video"
    return value[:80]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timed(label: str, func, *args, **kwargs):
    started = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - started
    print(f"[TIME] {label}: {elapsed:.3f}s")
    return result, elapsed


def _generate_fusion_config(
    *,
    template_config: Path,
    output_config: Path,
    repo_root: Path,
    stt_json: Path,
    vlm_json: Path,
    manifest_json: Path,
    qwen3_detect_json: Optional[Path],
    output_root: Path,
) -> None:
    payload: Dict[str, Any]
    with template_config.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(repo_root)).replace("\\", "/")
        except ValueError:
            return str(p)

    payload["paths"] = {
        "stt_json": _rel(stt_json),
        "vlm_json": _rel(vlm_json),
        "captures_manifest_json": _rel(manifest_json),
        "qwen3_detect_json": _rel(qwen3_detect_json) if qwen3_detect_json else None,
        "output_root": _rel(output_root),
    }

    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _run_stt_clova(video_path: Path, output_stt_json: Path) -> None:
    client = ClovaSpeechClient()
    with tempfile.TemporaryDirectory(prefix="stt_preprocess_") as temp_dir:
        temp_wav = Path(temp_dir) / "stt_input.wav"
        preprocess_audio(video_path, temp_wav)
        client.transcribe(
            temp_wav,
            output_path=output_stt_json,
        )


def _run_capture(
    video_path: Path,
    captures_dir: Path,
    manifest_json: Path,
    *,
    threshold: float,
    min_interval: float,
    verbose: bool,
    video_name: str,
) -> List[Dict[str, Any]]:
    captures_dir.mkdir(parents=True, exist_ok=True)
    processor = VideoProcessor()
    metadata = processor.extract_keyframes(
        str(video_path),
        output_dir=str(captures_dir),
        threshold=threshold,
        min_interval=min_interval,
        verbose=verbose,
        video_name=video_name,
    )
    _write_json(manifest_json, metadata)
    return metadata


def _run_vlm_openrouter(
    *,
    captures_dir: Path,
    manifest_json: Path,
    video_name: str,
    output_base: Path,
    batch_size: Optional[int],
) -> None:
    extractor = OpenRouterVlmExtractor(video_name=video_name, output_root=output_base)
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size는 1 이상이어야 합니다.")

    image_paths = _collect_capture_images(manifest_json, captures_dir)
    if not image_paths:
        raise ValueError("VLM 입력 이미지가 없습니다(manifest.json을 확인하세요).")

    results = extractor.extract_features(image_paths, batch_size=batch_size)
    raw_path = extractor.get_output_path()
    write_vlm_raw_json(results, raw_path)

    convert_vlm_raw_to_fusion_vlm(
        manifest_json=manifest_json,
        vlm_raw_json=raw_path,
        output_vlm_json=raw_path.with_name("vlm.json"),
    )


def _collect_capture_images(manifest_json: Path, captures_dir: Path) -> List[str]:
    manifest_payload = json.loads(manifest_json.read_text(encoding="utf-8"))
    if not isinstance(manifest_payload, list):
        raise ValueError("manifest.json 형식이 올바르지 않습니다(배열이어야 함).")

    image_paths: List[str] = []
    for item in sorted(
        (x for x in manifest_payload if isinstance(x, dict)),
        key=lambda x: (int(x.get("timestamp_ms", 0)), str(x.get("file_name", ""))),
    ):
        file_name = str(item.get("file_name", "")).strip()
        if not file_name:
            continue
        image_paths.append(str(captures_dir / file_name))
    return image_paths


def _run_qwen3_detect_openrouter(
    *,
    captures_dir: Path,
    manifest_json: Path,
    output_root: Path,
    batch_size: Optional[int],
    prompt: str,
) -> Path:
    detector = OpenRouterQwen3Detector()
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size는 1 이상이어야 합니다.")

    image_paths = _collect_capture_images(manifest_json, captures_dir)
    if not image_paths:
        raise ValueError("qwen3 detect 입력 이미지가 없습니다(manifest.json을 확인하세요).")

    outputs = detector.extract_features(image_paths, prompt, batch_size=batch_size)
    raw_path = output_root / "qwen3_detect_raw.json"
    raw_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    return raw_path


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = [line.strip() for line in stripped.splitlines()]
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _load_manifest_map(manifest_json: Path) -> Dict[str, int]:
    payload = json.loads(manifest_json.read_text(encoding="utf-8"))
    mapping: Dict[str, int] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        file_name = str(item.get("file_name", "")).strip()
        if not file_name:
            continue
        try:
            timestamp_ms = int(item["timestamp_ms"])
        except (TypeError, ValueError, KeyError):
            continue
        mapping[file_name] = timestamp_ms
    return mapping


def _postprocess_qwen3_detect(
    raw_json: Path,
    manifest_json: Path,
    output_json: Path,
) -> None:
    payload = json.loads(raw_json.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("qwen3_detect.json 형식이 올바르지 않습니다(배열이어야 함).")

    manifest_map = _load_manifest_map(manifest_json)
    items: List[Dict[str, Any]] = []

    for entry in payload:
        if not isinstance(entry, dict):
            continue
        image_path = entry.get("image")
        if not isinstance(image_path, str) or not image_path.strip():
            continue
        file_name = Path(image_path).name
        timestamp_ms = manifest_map.get(file_name)
        if timestamp_ms is None:
            continue

        raw = entry.get("raw", "")
        cleaned = _strip_code_fence(str(raw))
        detections: List[Dict[str, Any]] = []
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                for det in parsed:
                    if not isinstance(det, dict):
                        continue
                    box = det.get("box") or det.get("bbox")
                    if not isinstance(box, list) or len(box) != 4:
                        continue
                    detections.append(
                        {
                            "label": det.get("label", ""),
                            "box": box,
                            "description": det.get("description", ""),
                        }
                    )
        except json.JSONDecodeError:
            detections = []

        items.append(
            {
                "timestamp_ms": timestamp_ms,
                "file_name": file_name,
                "image_path": image_path,
                "image_size": entry.get("image_size"),
                "coord_space": entry.get("coord_space"),
                "bbox_format": entry.get("bbox_format"),
                "detections": detections,
            }
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps({"schema_version": DETECT_SCHEMA_VERSION, "items": items}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _run_fusion_pipeline(config_path: Path, *, limit: Optional[int], dry_run: bool) -> Dict[str, float]:
    config = load_config(str(config_path))
    ensure_output_root(config.paths.output_root)

    timings: Dict[str, float] = {}

    _, timings["sync_engine_sec"] = _timed(
        "sync_engine",
        run_sync_engine,
        config,
        limit=limit,
        dry_run=False,
    )

    _, timings["llm_summarizer_sec"] = _timed(
        "summarizer(LLM)",
        run_summarizer,
        config,
        limit=limit,
        dry_run=dry_run,
    )

    output_dir = config.paths.output_root / "fusion"
    if not dry_run:
        _, timings["renderer_sec"] = _timed(
            "renderer",
            render_segment_summaries_md,
            summaries_jsonl=output_dir / "segment_summaries.jsonl",
            output_md=output_dir / "segment_summaries.md",
            include_sources=config.raw.render.include_sources,
            sources_jsonl=output_dir / "segments_units.jsonl",
            md_wrap_width=config.raw.render.md_wrap_width,
            limit=limit,
        )

        summaries, timings["final_summary_sec"] = _timed(
            "final_summary",
            compose_final_summaries,
            summaries_jsonl=output_dir / "segment_summaries.jsonl",
            max_chars=config.raw.final_summary.max_chars_per_format,
            include_timestamps=config.raw.final_summary.style.include_timestamps,
            limit=limit,
        )
        outputs_dir = output_dir / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        for fmt in config.raw.final_summary.generate_formats:
            if fmt in summaries:
                outputs_dir.joinpath(f"final_summary_{fmt}.md").write_text(
                    summaries[fmt], encoding="utf-8"
                )
    return timings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="비디오 1개 입력으로 end-to-end 요약 실행")
    parser.add_argument("--video", required=True, help="입력 비디오 파일 경로")
    parser.add_argument("--output-base", default="data/outputs", help="동영상별 outputs 베이스 디렉토리")
    parser.add_argument("--stt-backend", choices=["clova"], default="clova", help="STT 백엔드")
    parser.add_argument("--parallel", action=argparse.BooleanOptionalAction, default=True, help="STT+Capture 병렬 실행")
    parser.add_argument("--capture-threshold", type=float, default=11.0, help="장면 전환 감지 임계값")
    parser.add_argument("--capture-min-interval", type=float, default=0.5, help="캡처 최소 간격(초)")
    parser.add_argument("--capture-verbose", action="store_true", help="캡처 상세 로그 출력")
    parser.add_argument("--vlm-batch-size", type=int, default=None, help="VLM 배치 크기(미지정 시 전부 한 번에)")
    parser.add_argument("--limit", type=int, default=None, help="fusion 단계에서 처리할 segment 수 제한")
    parser.add_argument("--dry-run", action="store_true", help="summarizer LLM 미호출(출력 미생성)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {video_path}")

    repo_root = ROOT
    output_base = (repo_root / Path(args.output_base)).resolve()
    video_name = _sanitize_video_name(video_path.stem)
    video_root = output_base / video_name
    video_root.mkdir(parents=True, exist_ok=True)

    run_meta_path = video_root / "pipeline_run.json"
    run_meta: Dict[str, Any] = {
        "schema_version": 1,
        "video_path": str(video_path),
        "video_name": video_name,
        "output_base": str(output_base),
        "video_root": str(video_root),
        "started_at_utc": _utc_now_iso(),
        "args": vars(args),
        "durations_sec": {},
        "status": "running",
    }
    _write_json(run_meta_path, run_meta)

    total_started = time.perf_counter()
    try:
        stt_json = video_root / "stt.json"
        captures_dir = video_root / "captures"
        manifest_json = video_root / "manifest.json"

        stt_elapsed = 0.0
        capture_elapsed = 0.0

        if args.parallel:
            with ThreadPoolExecutor(max_workers=2) as executor:
                stt_future = executor.submit(_timed, "stt", _run_stt_clova, video_path, stt_json)
                capture_future = executor.submit(
                    _timed,
                    "capture",
                    _run_capture,
                    video_path,
                    captures_dir,
                    manifest_json,
                    threshold=args.capture_threshold,
                    min_interval=args.capture_min_interval,
                    verbose=args.capture_verbose,
                    video_name=video_name,
                )
                _, stt_elapsed = stt_future.result()
                _, capture_elapsed = capture_future.result()
        else:
            _, stt_elapsed = _timed("stt", _run_stt_clova, video_path, stt_json)
            _, capture_elapsed = _timed(
                "capture",
                _run_capture,
                video_path,
                captures_dir,
                manifest_json,
                threshold=args.capture_threshold,
                min_interval=args.capture_min_interval,
                verbose=args.capture_verbose,
                video_name=video_name,
            )

        _, vlm_elapsed = _timed(
            "vlm",
            _run_vlm_openrouter,
            captures_dir=captures_dir,
            manifest_json=manifest_json,
            video_name=video_name,
            output_base=output_base,
            batch_size=args.vlm_batch_size,
        )

        raw_detect_path = _run_qwen3_detect_openrouter(
            captures_dir=captures_dir,
            manifest_json=manifest_json,
            output_root=video_root,
            batch_size=args.vlm_batch_size,
            prompt=DEFAULT_DETECT_PROMPT,
        )
        qwen3_detect_clean = video_root / "qwen3_detect.json"
        _postprocess_qwen3_detect(raw_detect_path, manifest_json, qwen3_detect_clean)

        template_config = repo_root / "src" / "fusion" / "config.yaml"
        if not template_config.exists():
            raise FileNotFoundError(f"fusion config template을 찾을 수 없습니다: {template_config}")
        fusion_config_path = video_root / "config.yaml"
        _generate_fusion_config(
            template_config=template_config,
            output_config=fusion_config_path,
            repo_root=repo_root,
            stt_json=stt_json,
            vlm_json=video_root / "vlm.json",
            manifest_json=manifest_json,
            qwen3_detect_json=qwen3_detect_clean,
            output_root=video_root,
        )

        fusion_timings = _run_fusion_pipeline(fusion_config_path, limit=args.limit, dry_run=args.dry_run)
        total_elapsed = time.perf_counter() - total_started
        llm_elapsed = float(fusion_timings.get("llm_summarizer_sec", 0.0))

        run_meta["durations_sec"] = {
            "stt_sec": round(stt_elapsed, 6),
            "capture_sec": round(capture_elapsed, 6),
            "vlm_sec": round(vlm_elapsed, 6),
            "llm_sec": round(llm_elapsed, 6),
            "total_sec": round(total_elapsed, 6),
            **{k: round(v, 6) for k, v in fusion_timings.items()},
        }
        run_meta["ended_at_utc"] = _utc_now_iso()
        run_meta["status"] = "ok"
        _write_json(run_meta_path, run_meta)

        print(f"[OK] outputs: {video_root}")
    except Exception as exc:
        run_meta["ended_at_utc"] = _utc_now_iso()
        run_meta["status"] = "error"
        run_meta["error"] = str(exc)
        _write_json(run_meta_path, run_meta)
        raise


if __name__ == "__main__":
    main()
