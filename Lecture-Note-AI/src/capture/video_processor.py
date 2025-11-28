"""
Slide extraction using histogram differences.

Inputs:
    - video_path: Path to an input video file.
Outputs:
    - List[SlideData]: Timestamped slide metadata for downstream processing.
"""

from pathlib import Path
from typing import List

import cv2
import numpy as np

from src.common.schemas import SlideData


class SlideExtractor:
    """Detects keyframes (slide changes) via histogram comparison."""

    def __init__(
        self,
        output_dir: Path | str = Path("data/output/images"),
        hist_threshold: float = 0.7,
        frame_interval: int = 30,
    ) -> None:
        """
        Args:
            output_dir: Directory where extracted slide images will be saved.
            hist_threshold: Correlation threshold to decide a new slide (lower => more sensitive).
            frame_interval: Process every Nth frame to balance speed vs accuracy.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.hist_threshold = hist_threshold
        self.frame_interval = frame_interval

    def _compute_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Compute a normalized color histogram for a frame."""
        hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
        return cv2.normalize(hist, hist).flatten()

    def extract(self, video_path: str) -> List[SlideData]:
        """
        Extract slide keyframes from a video.

        Returns:
            List[SlideData]: Sorted by timestamp.
        """
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        slides: List[SlideData] = []
        previous_hist: np.ndarray | None = None
        frame_idx = 0

        while True:
            has_frame, frame = capture.read()
            if not has_frame:
                break

            if frame_idx % self.frame_interval != 0:
                frame_idx += 1
                continue

            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            histogram = self._compute_histogram(frame)

            similarity = (
                cv2.compareHist(previous_hist, histogram, cv2.HISTCMP_CORREL)
                if previous_hist is not None
                else 0
            )

            if previous_hist is None or similarity < self.hist_threshold:
                filename = self.output_dir / f"slide_{len(slides):04d}.jpg"
                cv2.imwrite(str(filename), frame)
                slides.append(SlideData(timestamp=timestamp, image_path=str(filename)))
                previous_hist = histogram

            frame_idx += 1

        capture.release()
        return slides
