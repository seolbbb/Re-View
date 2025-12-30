"""
Alignment utilities that tie slides, audio, and OCR results together.

Inputs:
    - slides: List[SlideData]
    - audio: List[AudioSegment]
    - ocr: List[OcrResult]
Outputs:
    - List[FusedContext]
"""

import math
from typing import Dict, List

from src.common.schemas import AudioSegment, FusedContext, OcrResult, SlideData


class ContextAligner:
    """Aligns audio segments and OCR detections to their nearest slide."""

    def align(
        self,
        slides: List[SlideData],
        audio: List[AudioSegment],
        ocr: List[OcrResult],
    ) -> List[FusedContext]:
        """
        Map transcript segments to slide windows and attach OCR hints.

        Returns:
            List[FusedContext]: Fusion of slide, audio script, and OCR data.
        """
        slides_sorted = sorted(slides, key=lambda s: s.timestamp)
        audio_sorted = sorted(audio, key=lambda a: a.start)

        ocr_lookup: Dict[str, List] = {item.image_path: item.raw_results for item in ocr}

        fused: List[FusedContext] = []
        for idx, slide in enumerate(slides_sorted):
            next_timestamp = slides_sorted[idx + 1].timestamp if idx + 1 < len(slides_sorted) else math.inf
            relevant_audio = [
                segment.text
                for segment in audio_sorted
                if slide.timestamp <= segment.start < next_timestamp
            ]
            script = " ".join(relevant_audio).strip()
            ocr_data = ocr_lookup.get(slide.image_path, [])

            fused.append(FusedContext(slide=slide, script=script, ocr_data=ocr_data))

        return fused
