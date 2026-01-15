"""파이프라인 단계별 실행 함수를 모은 모듈."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.audio.stt_router import STTRouter
from src.capture.process_content import process_single_video_capture
from src.fusion.config import load_config
from src.fusion.io_utils import ensure_output_root, write_json
from src.fusion.renderer import compose_final_summaries, render_segment_summaries_md
from src.fusion.summarizer import run_summarizer
from src.fusion.sync_engine import run_sync_engine
from src.judge.judge import run_judge
from src.pipeline.benchmark import BenchmarkTimer
from src.vlm.vlm_engine import OpenRouterVlmExtractor, write_vlm_raw_json
from src.vlm.vlm_fusion import convert_vlm_raw_to_fusion_vlm


def generate_fusion_config(
    *,
    template_config: Path,
    output_config: Path,
    repo_root: Path,
    stt_json: Path,
    vlm_json: Path,
    manifest_json: Optional[Path],
    output_root: Path,
) -> None:
    """Fusion settings.yaml을 템플릿에서 생성한다."""
    with template_config.open("r", encoding="utf-8") as handle:
        payload: Dict[str, Any] = yaml.safe_load(handle)

    def _rel(path: Path) -> str:
        try:
            return str(path.relative_to(repo_root)).replace("\\", "/")
        except ValueError:
            return str(path)

    paths_payload: Dict[str, Any] = {
        "stt_json": _rel(stt_json),
        "vlm_json": _rel(vlm_json),
        "output_root": _rel(output_root),
    }
    if manifest_json is not None:
        paths_payload["captures_manifest_json"] = _rel(manifest_json)

    payload["paths"] = paths_payload

    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def run_stt(video_path: Path, output_stt_json: Path, *, backend: str) -> None:
    """음성 인식을 실행해 stt.json을 생성한다."""
    router = STTRouter(provider=backend)
    audio_output_path = output_stt_json.with_name(f"{video_path.stem}.wav")
    router.transcribe_media(
        video_path,
        provider=backend,
        audio_output_path=audio_output_path,
        mono_method="auto",
        output_path=output_stt_json,
    )


def run_capture(
    video_path: Path,
    output_base: Path,
    *,
    threshold: float,
    dedupe_threshold: float,
    min_interval: float,
    verbose: bool,
    video_name: str,
) -> List[Dict[str, Any]]:
    """슬라이드 캡처를 실행하고 메타데이터 목록을 반환한다."""
    metadata = process_single_video_capture(
        str(video_path),
        str(output_base),
        scene_threshold=threshold,
        dedupe_threshold=dedupe_threshold,
        min_interval=min_interval,
    )
    return metadata


def run_vlm_openrouter(
    *,
    captures_dir: Path,
    manifest_json: Path,
    video_name: str,
    output_base: Path,
    batch_size: Optional[int],
    concurrency: int,
    show_progress: bool,
) -> int:
    """이미지 정보를 추출해 vlm.json을 만들고 처리 개수를 반환한다."""
    extractor = OpenRouterVlmExtractor(video_name=video_name, output_root=output_base)
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size는 1 이상이어야 합니다.")

    manifest_payload = json.loads(manifest_json.read_text(encoding="utf-8"))
    if not isinstance(manifest_payload, list):
        raise ValueError("manifest.json 형식이 올바르지 않습니다(배열이어야 함).")

    image_paths: List[str] = []
    for item in sorted(
        (x for x in manifest_payload if isinstance(x, dict)),
        key=lambda x: (int(x.get("timestamp_ms", x.get("start_ms", 0))), str(x.get("file_name", ""))),
    ):
        file_name = str(item.get("file_name", "")).strip()
        if not file_name:
            continue
        image_paths.append(str(captures_dir / file_name))

    if not image_paths:
        raise ValueError("VLM 입력 이미지가 없습니다(manifest.json을 확인하세요).")

    results = extractor.extract_features(
        image_paths,
        batch_size=batch_size,
        show_progress=show_progress,
        concurrency=concurrency,
    )
    raw_path = extractor.get_output_path()
    write_vlm_raw_json(results, raw_path)

    convert_vlm_raw_to_fusion_vlm(
        manifest_json=manifest_json,
        vlm_raw_json=raw_path,
        output_vlm_json=raw_path.with_name("vlm.json"),
    )
    raw_path.unlink(missing_ok=True)

    return len(image_paths)


def _filter_manifest_by_time_range(
    manifest_payload: List[Dict[str, Any]],
    start_ms: int,
    end_ms: int,
) -> List[Dict[str, Any]]:
    """manifest에서 특정 시간 범위의 항목만 필터링한다."""
    filtered = []
    for item in manifest_payload:
        timestamp_ms = item.get("timestamp_ms", item.get("start_ms", 0))
        if timestamp_ms is None:
            continue
        timestamp_ms = int(timestamp_ms)
        if start_ms <= timestamp_ms < end_ms:
            filtered.append(item)
    return filtered


def run_vlm_for_batch(
    *,
    captures_dir: Path,
    manifest_json: Path,
    video_name: str,
    output_dir: Path,
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
    batch_manifest: Optional[List[Dict[str, Any]]] = None,
    batch_size: Optional[int] = None,
    concurrency: int = 1,
    show_progress: bool = False,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """배치 범위만 VLM 처리해 batch 단위의 vlm.json을 생성한다."""
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = OpenRouterVlmExtractor(video_name=video_name, output_root=output_dir)
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size는 1 이상이어야 합니다.")

    manifest_payload = json.loads(manifest_json.read_text(encoding="utf-8"))
    if not isinstance(manifest_payload, list):
        raise ValueError("manifest.json 형식이 올바르지 않습니다(배열이어야 함).")

    if batch_manifest is not None:
        filtered_manifest_items = batch_manifest
    elif start_idx is not None and end_idx is not None:
        sorted_manifest = sorted(
            (x for x in manifest_payload if isinstance(x, dict)),
            key=lambda x: (int(x.get("timestamp_ms", x.get("start_ms", 0))), str(x.get("file_name", ""))),
        )
        filtered_manifest_items = sorted_manifest[start_idx:end_idx]
    elif start_ms is not None and end_ms is not None:
        filtered_manifest_items = _filter_manifest_by_time_range(
            manifest_payload,
            start_ms,
            end_ms,
        )
    else:
        raise ValueError("batch_manifest, start_idx/end_idx, 또는 start_ms/end_ms 중 하나가 필요합니다.")

    image_paths: List[str] = []
    for item in sorted(
        (x for x in filtered_manifest_items if isinstance(x, dict)),
        key=lambda x: (int(x.get("timestamp_ms", x.get("start_ms", 0))), str(x.get("file_name", ""))),
    ):
        file_name = str(item.get("file_name", "")).strip()
        if not file_name:
            continue
        image_paths.append(str(captures_dir / file_name))

    if not image_paths:
        empty_vlm = {"items": [], "duration_ms": 0}
        vlm_json_path = output_dir / "vlm.json"
        vlm_json_path.write_text(json.dumps(empty_vlm, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "vlm_raw_json": "",
            "vlm_json": str(vlm_json_path),
            "image_count": 0,
        }

    results = extractor.extract_features(
        image_paths,
        batch_size=batch_size,
        show_progress=show_progress,
        concurrency=concurrency,
    )

    raw_path = output_dir / "vlm_raw.json"
    write_vlm_raw_json(results, raw_path)

    temp_manifest_path = output_dir / "manifest_temp.json"
    temp_manifest_path.write_text(
        json.dumps(filtered_manifest_items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    vlm_json_path = output_dir / "vlm.json"
    convert_vlm_raw_to_fusion_vlm(
        manifest_json=temp_manifest_path,
        vlm_raw_json=raw_path,
        output_vlm_json=vlm_json_path,
    )
    temp_manifest_path.unlink(missing_ok=True)

    return {
        "vlm_raw_json": str(raw_path),
        "vlm_json": str(vlm_json_path),
        "image_count": len(image_paths),
    }


def run_fusion_pipeline(
    config_path: Path,
    *,
    limit: Optional[int],
    timer: BenchmarkTimer,
) -> Dict[str, Any]:
    """동기화부터 최종 요약까지 묶어서 실행하고 통계를 반환한다."""
    config = load_config(str(config_path))
    ensure_output_root(config.paths.output_root)

    fusion_info: Dict[str, Any] = {
        "segment_count": 0,
        "timings": {},
    }

    _, sync_elapsed = timer.time_stage(
        "fusion.sync_engine",
        run_sync_engine,
        config,
        limit=limit,
    )
    fusion_info["timings"]["sync_engine_sec"] = sync_elapsed

    _, llm_elapsed = timer.time_stage(
        "fusion.llm_summarizer",
        run_summarizer,
        config,
        limit=limit,
    )
    fusion_info["timings"]["llm_summarizer_sec"] = llm_elapsed

    output_dir = config.paths.output_root / "fusion"

    _, render_elapsed = timer.time_stage(
        "fusion.renderer",
        render_segment_summaries_md,
        summaries_jsonl=output_dir / "segment_summaries.jsonl",
        output_md=output_dir / "segment_summaries.md",
        include_sources=config.raw.render.include_sources,
        sources_jsonl=output_dir / "segments_units.jsonl",
        md_wrap_width=config.raw.render.md_wrap_width,
        limit=limit,
    )
    fusion_info["timings"]["renderer_sec"] = render_elapsed

    summaries, final_elapsed = timer.time_stage(
        "fusion.final_summary",
        compose_final_summaries,
        summaries_jsonl=output_dir / "segment_summaries.jsonl",
        max_chars=config.raw.final_summary.max_chars_per_format,
        include_timestamps=config.raw.final_summary.style.include_timestamps,
        limit=limit,
    )
    fusion_info["timings"]["final_summary_sec"] = final_elapsed

    outputs_dir = output_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    for fmt in config.raw.final_summary.generate_formats:
        if fmt in summaries:
            outputs_dir.joinpath(f"final_summary_{fmt}.md").write_text(
                summaries[fmt], encoding="utf-8"
            )

    judge_output_dir = output_dir / "judge"
    judge_output_dir.mkdir(parents=True, exist_ok=True)
    judge_result, judge_elapsed = timer.time_stage(
        "fusion.judge",
        run_judge,
        config=config,
        segments_units_path=output_dir / "segments_units.jsonl",
        segment_summaries_path=output_dir / "segment_summaries.jsonl",
        output_report_path=judge_output_dir / "judge_report.json",
        output_segments_path=judge_output_dir / "judge_segment_reports.jsonl",
        batch_size=config.judge.batch_size,
        workers=config.judge.workers,
        json_repair_attempts=config.judge.json_repair_attempts,
        limit=limit,
        write_outputs=True,
        verbose=config.judge.verbose,
    )
    fusion_info["timings"]["judge_sec"] = judge_elapsed

    report = judge_result.get("report", {})
    segment_reports = judge_result.get("segment_reports", []) or []
    final_score = float(report.get("scores", {}).get("final", 0.0))
    min_score = float(config.judge.min_score)
    passed = final_score >= min_score
    feedback = [
        {"segment_id": int(item.get("segment_id")), "feedback": str(item.get("feedback", "")).strip()}
        for item in segment_reports
        if item.get("segment_id") is not None
    ]
    payload: Dict[str, Any] = {
        "model": str(report.get("meta", {}).get("model", "")),
        "pass": passed,
        "final_score": final_score,
        "min_score": min_score,
        "prompt_version": str(report.get("meta", {}).get("prompt_version", "")),
        "generated_at_utc": str(report.get("meta", {}).get("generated_at_utc", "")),
        "feedback": feedback,
    }
    if config.judge.include_segments:
        payload["segments"] = [
            {
                "segment_id": int(item.get("segment_id")),
                "scores": item.get("scores", {}),
            }
            for item in segment_reports
            if item.get("segment_id") is not None
        ]
    write_json(output_dir / "judge.json", payload)

    segments_file = output_dir / "segment_summaries.jsonl"
    if segments_file.exists():
        fusion_info["segment_count"] = sum(1 for _ in segments_file.open(encoding="utf-8"))

    return fusion_info


def run_batch_fusion_pipeline(
    *,
    video_root: Path,
    captures_dir: Path,
    manifest_json: Path,
    stt_json: Path,
    video_name: str,
    batch_size: int,
    timer: BenchmarkTimer,
    vlm_batch_size: Optional[int],
    vlm_concurrency: int,
    vlm_show_progress: bool,
    limit: Optional[int],
    repo_root: Path,
) -> Dict[str, Any]:
    """배치 단위로 동기화와 요약을 반복 실행한다."""
    from src.fusion.summarizer import run_batch_summarizer
    from src.fusion.sync_engine import run_batch_sync_engine

    manifest_payload = json.loads(manifest_json.read_text(encoding="utf-8"))
    sorted_manifest = sorted(
        (x for x in manifest_payload if isinstance(x, dict)),
        key=lambda x: (int(x.get("timestamp_ms", x.get("start_ms", 0))), str(x.get("file_name", ""))),
    )

    total_captures = len(sorted_manifest)
    total_batches = max(1, math.ceil(total_captures / batch_size))

    print(
        f"\n📦 Pipeline batches: {total_captures} images across {total_batches} groups "
        f"(group size: {batch_size})"
    )

    batch_ranges = []
    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_captures)
        start_ms = int(
            sorted_manifest[start_idx].get("start_ms", sorted_manifest[start_idx].get("timestamp_ms", 0))
        )
        end_ms = int(
            sorted_manifest[end_idx - 1].get(
                "end_ms", sorted_manifest[end_idx - 1].get("timestamp_ms", 0) + 1000
            )
        )
        batch_ranges.append(
            {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "capture_count": end_idx - start_idx,
            }
        )

    batches_dir = video_root / "batches"
    batches_dir.mkdir(parents=True, exist_ok=True)
    fusion_dir = video_root / "fusion"
    fusion_dir.mkdir(parents=True, exist_ok=True)

    template_config = repo_root / "config" / "fusion" / "settings.yaml"
    fusion_config_path = video_root / "config.yaml"

    fusion_info: Dict[str, Any] = {
        "segment_count": 0,
        "batch_count": total_batches,
        "batch_results": [],
        "timings": {},
    }

    cumulative_segment_count = 0
    previous_context = ""

    accumulated_summaries_path = fusion_dir / "segment_summaries.jsonl"
    if accumulated_summaries_path.exists():
        accumulated_summaries_path.unlink()

    for batch_idx, batch_info in enumerate(batch_ranges):
        if batch_idx > 0:
            print("\n⏳ Waiting 5s to avoid API rate limiting...")
            time.sleep(5)

        print(f"\n{'-'*50}")
        print(f"🔄 Pipeline batch {batch_idx + 1}/{total_batches} in progress...")
        print(f"   Capture range: {batch_info['start_idx']} ~ {batch_info['end_idx'] - 1}")

        batch_dir = batches_dir / f"batch_{batch_idx}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        batch_manifest = sorted_manifest[batch_info["start_idx"] : batch_info["end_idx"]]

        timer.time_stage(
            f"pipeline_batch_{batch_idx + 1}.vlm",
            run_vlm_for_batch,
            captures_dir=captures_dir,
            manifest_json=manifest_json,
            video_name=video_name,
            output_dir=batch_dir,
            batch_manifest=batch_manifest,
            batch_size=vlm_batch_size,
            concurrency=vlm_concurrency,
            show_progress=vlm_show_progress,
        )

        if not fusion_config_path.exists():
            generate_fusion_config(
                template_config=template_config,
                output_config=fusion_config_path,
                repo_root=repo_root,
                stt_json=stt_json,
                vlm_json=batch_dir / "vlm.json",
                manifest_json=manifest_json,
                output_root=video_root,
            )

        sync_result, _ = timer.time_stage(
            f"pipeline_batch_{batch_idx + 1}.sync",
            run_batch_sync_engine,
            stt_json=stt_json,
            vlm_json=batch_dir / "vlm.json",
            manifest_json=manifest_json,
            output_dir=batch_dir,
            time_range=(batch_info["start_ms"], batch_info["end_ms"]),
            sync_config={
                "min_segment_sec": 15,
                "max_segment_sec": 120,
                "max_transcript_chars": 1000,
                "silence_gap_ms": 500,
                "max_visual_items": 10,
                "max_visual_chars": 3000,
                "dedup_similarity_threshold": 0.9,
            },
            segment_id_offset=cumulative_segment_count,
        )

        new_segment_count = sync_result.get("segments_count", 0)
        cumulative_segment_count += new_segment_count

        batch_segments_path = batch_dir / "segments_units.jsonl"
        batch_summaries_path = batch_dir / "segment_summaries.jsonl"

        config = load_config(str(fusion_config_path))
        summarize_result, _ = timer.time_stage(
            f"pipeline_batch_{batch_idx + 1}.summarize",
            run_batch_summarizer,
            segments_units_jsonl=batch_segments_path,
            output_dir=batch_dir,
            config=config,
            previous_context=previous_context,
            limit=limit,
        )

        new_context = summarize_result.get("context", "")
        if new_context:
            previous_context = new_context[:500]

        if batch_summaries_path.exists():
            with batch_summaries_path.open("r", encoding="utf-8") as handle:
                batch_content = handle.read()
            with accumulated_summaries_path.open("a", encoding="utf-8") as handle:
                handle.write(batch_content)

        batch_judge_dir = batch_dir / "judge"
        batch_judge_dir.mkdir(parents=True, exist_ok=True)
        judge_result, _ = timer.time_stage(
            f"pipeline_batch_{batch_idx + 1}.judge",
            run_judge,
            config=config,
            segments_units_path=batch_segments_path,
            segment_summaries_path=batch_summaries_path,
            output_report_path=batch_judge_dir / "judge_report.json",
            output_segments_path=batch_judge_dir / "judge_segment_reports.jsonl",
            batch_size=config.judge.batch_size,
            workers=config.judge.workers,
            json_repair_attempts=config.judge.json_repair_attempts,
            limit=limit,
            verbose=config.judge.verbose,
            write_outputs=True,
        )

        report = judge_result.get("report", {})
        segment_reports = judge_result.get("segment_reports", []) or []
        final_score = float(report.get("scores", {}).get("final", 0.0))
        min_score = float(config.judge.min_score)
        passed = final_score >= min_score
        feedback = [
            {"segment_id": int(item.get("segment_id")), "feedback": str(item.get("feedback", "")).strip()}
            for item in segment_reports
            if item.get("segment_id") is not None
        ]
        payload = {
            "model": str(report.get("meta", {}).get("model", "")),
            "pass": passed,
            "final_score": final_score,
            "min_score": min_score,
            "prompt_version": str(report.get("meta", {}).get("prompt_version", "")),
            "generated_at_utc": str(report.get("meta", {}).get("generated_at_utc", "")),
            "feedback": feedback,
        }
        if config.judge.include_segments:
            payload["segments"] = [
                {
                    "segment_id": int(item.get("segment_id")),
                    "scores": item.get("scores", {}),
                }
                for item in segment_reports
                if item.get("segment_id") is not None
            ]
        batch_judge_path = batch_dir / "judge.json"
        write_json(batch_judge_path, payload)
        print(
            f"  📊 Pipeline batch {batch_idx + 1} Judge: {'PASS' if passed else 'FAIL'} "
            f"(score: {final_score:.1f})"
        )

        fusion_info["batch_results"].append(
            {
                "batch_index": batch_idx,
                "capture_range": [batch_info["start_idx"], batch_info["end_idx"]],
                "segments_count": new_segment_count,
            }
        )

        print(
            f"  ✅ Pipeline batch {batch_idx + 1} complete "
            f"(segments: {new_segment_count})"
        )

    fusion_info["segment_count"] = cumulative_segment_count

    if accumulated_summaries_path.exists():
        config = load_config(str(fusion_config_path))
        _, render_elapsed = timer.time_stage(
            "fusion.renderer",
            render_segment_summaries_md,
            summaries_jsonl=accumulated_summaries_path,
            output_md=fusion_dir / "segment_summaries.md",
            include_sources=config.raw.render.include_sources,
            sources_jsonl=fusion_dir / "segments_units.jsonl"
            if (fusion_dir / "segments_units.jsonl").exists()
            else None,
            md_wrap_width=config.raw.render.md_wrap_width,
            limit=limit,
        )
        fusion_info["timings"]["renderer_sec"] = render_elapsed

    return fusion_info
