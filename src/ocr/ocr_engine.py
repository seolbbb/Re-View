"""
PaddleOCR-based extraction with bounding boxes retained.

Inputs:
    - image_paths: List of slide image paths.
Outputs:
    - List[OcrResult]: Structured OCR detections per image.
"""

from typing import List

from paddleocr import PaddleOCR

from src.common.schemas import OcrBox, OcrResult


class PaddleHintExtractor:
    """Extract OCR hints and bounding boxes for downstream alignment."""

    def __init__(self) -> None:
        # Angle classification improves rotated text handling; korean supports multilingual detection.
        self.ocr = PaddleOCR(use_angle_cls=True, lang="korean")

    def extract_features(self, image_paths: List[str]) -> List[OcrResult]:
        """
        Run OCR over provided images and keep bounding boxes + raw text.

        Returns:
            List[OcrResult]: Structured OCR results per image.
        """
        results: List[OcrResult] = []

        for image_path in image_paths:
            raw_output = self.ocr.ocr(image_path, cls=True)
            detections: List[OcrBox] = []

            for line in raw_output:
                for bbox, (text, _score) in line:
                    # Preserve raw math-like strings; Gemini will reformat to LaTeX when needed.
                    formatted_bbox = [[float(x), float(y)] for x, y in bbox]
                    detections.append(OcrBox(text=text, bbox=formatted_bbox))

            results.append(OcrResult(image_path=image_path, raw_results=detections))

        return results
