"""Shared data contracts enforced by Pydantic across all components."""

from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SlideData(BaseModel):
    """Represents a key slide frame extracted from a video."""

    model_config = ConfigDict(extra="forbid")

    timestamp: float = Field(..., ge=0, description="Time in seconds when the slide appears.")
    image_path: str = Field(..., description="Path to the saved slide image.")


class AudioSegment(BaseModel):
    """Represents a transcribed chunk of audio."""

    model_config = ConfigDict(extra="forbid")

    start: float = Field(..., ge=0, description="Start time (seconds) of the transcript segment.")
    end: float = Field(..., ge=0, description="End time (seconds) of the transcript segment.")
    text: str = Field(..., min_length=1, description="Recognized transcript text.")

    @model_validator(mode="after")
    def validate_times(self) -> "AudioSegment":
        """Ensure end time is not before start time."""
        if self.end < self.start:
            raise ValueError("end must be greater than or equal to start")
        return self


class OcrBox(BaseModel):
    """Represents a single OCR detection bounding box and its text."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1, description="Detected text snippet.")
    bbox: List[List[float]] = Field(
        ...,
        description="Polygon of the detected region as [[x1, y1], [x2, y2], ...].",
    )

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, value: List[List[float]]) -> List[List[float]]:
        """Ensure the bounding box is a list of coordinate pairs."""
        if not value:
            raise ValueError("bbox cannot be empty")
        for point in value:
            if len(point) != 2:
                raise ValueError("each bbox point must contain exactly two values [x, y]")
        return value


class OcrResult(BaseModel):
    """All OCR detections tied to a single image."""

    model_config = ConfigDict(extra="forbid")

    image_path: str = Field(..., description="Image that produced these OCR results.")
    raw_results: List[OcrBox] = Field(default_factory=list, description="Structured OCR detections.")


class FusedContext(BaseModel):
    """Fusion of slide, audio script, and OCR context for LLM consumption."""

    model_config = ConfigDict(extra="forbid")

    slide: SlideData
    script: str = Field("", description="Aligned transcript text for the slide.")
    ocr_data: List[OcrBox] = Field(default_factory=list, description="OCR detections for the slide.")
