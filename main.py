"""Pipeline orchestrator for Lecture-Note-AI."""

import os
from pathlib import Path

from dotenv import load_dotenv

from src.audio.speech_client import ClovaSpeechClient
from src.capture.video_processor import SlideExtractor
from src.fusion.data_fuser import ContextAligner
from src.fusion.llm_generator import NoteGenerator
from src.ocr.ocr_engine import OpenRouterOcrExtractor


def run_pipeline() -> None:
    """Execute the end-to-end note generation pipeline."""
    load_dotenv()

    video_path = os.getenv("VIDEO_PATH", "data/input/sample.mp4")

    slide_extractor = SlideExtractor()
    speech_client = ClovaSpeechClient()
    ocr_client = OpenRouterOcrExtractor()
    aligner = ContextAligner()
    note_generator = NoteGenerator()

    slides = slide_extractor.extract(video_path)
    audio_segments = speech_client.transcribe(video_path)
    ocr_results = ocr_client.extract_features([slide.image_path for slide in slides])
    fused_contexts = aligner.align(slides, audio_segments, ocr_results)

    notes_md = note_generator.generate_note(fused_contexts)

    output_path = Path("data/output/notes.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(notes_md, encoding="utf-8")
    print(f"[OK] Notes written to {output_path}")


def main() -> None:
    print("이전 파이프라인 엔트리포인트는 제거되었습니다. src/fusion/README.md를 참고하세요.")


if __name__ == "__main__":
    main()
