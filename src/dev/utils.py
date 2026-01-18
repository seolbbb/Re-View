# -*- coding: utf-8 -*-
"""
Common utilities for dev scripts.
Includes context management (paths, config) and experiment logging.
"""

import sys
import csv
import shutil
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional

# Add repo root to path if not present
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.fusion.config import load_config, ConfigBundle


class DevContext:
    """Manages paths and configuration for a specific video/experiment run."""

    def __init__(self, video_name: str, output_base: str = "data/outputs"):
        self.root = ROOT
        self.video_name = video_name
        self.output_base = self.root / output_base
        self.video_dir = self.output_base / video_name
        self.fusion_dir = self.video_dir / "fusion"
        
        # Standard input paths
        self.paths = {
            "stt": self.video_dir / "stt.json",
            "vlm": self.video_dir / "vlm.json",
            "manifest": self.video_dir / "capture.json",
            "config": self.video_dir / "config.yaml",
            "segments_units": self.fusion_dir / "segments_units.jsonl",
            "segment_summaries": self.fusion_dir / "segment_summaries.jsonl",
            "token_usage": self.fusion_dir / "token_usage.json",
        }

    def exists(self) -> bool:
        """Check if video directory exists."""
        return self.video_dir.exists()

    def load_config(self) -> ConfigBundle:
        """Load config.yaml from video directory."""
        if not self.paths["config"].exists():
            raise FileNotFoundError(f"Config file not found: {self.paths['config']}")
        return load_config(str(self.paths["config"]))

    def setup_batch_dirs(self) -> Path:
        """Create and return batches directory."""
        batches_dir = self.video_dir / "batches"
        batches_dir.mkdir(parents=True, exist_ok=True)
        return batches_dir

    def cleanup_previous_fusion_outputs(self):
        """Remove previous summary and judge output files."""
        if self.paths["segment_summaries"].exists():
            self.paths["segment_summaries"].unlink()
        
        judge_dir = self.fusion_dir / "judge"
        if judge_dir.exists():
            shutil.rmtree(judge_dir)
        judge_dir.mkdir(parents=True, exist_ok=True)


class ExperimentLogger:
    """Handles logging of experiment results to CSV."""

    def __init__(self, csv_path: Optional[Path] = None):
        if csv_path is None:
            self.csv_path = ROOT / "src" / "dev" / "experiment_summary.csv"
        else:
            self.csv_path = csv_path

    def log(
        self,
        prompt_version: str,
        scores: Dict[str, float],
        input_tokens: int,
        elapsed_sec: float,
        note: str,
        model: str = "",
        temperature: float = 0.0,
        segments_count: int = 0,
        workers: int = 1,
        segment_indices: str = "",
        command: str = "",
        component: str = "pipeline",
        # Module timings
        stt_sec: float = 0.0,
        capture_sec: float = 0.0,
        vlm_sec: float = 0.0,
        summarizer_sec: float = 0.0,
        judge_elapsed_sec: float = 0.0,
    ):
        """Append a log entry to the CSV file."""
        file_exists = self.csv_path.exists()
        
        headers = [
            "timestamp", "component", "prompt_version", "model", "temperature", 
            "final_score", "groundedness", "compliance", "note_quality", "multimodal_use",
            "input_tokens", "elapsed_sec", 
            "stt_sec", "capture_sec", "vlm_sec", "summarizer_sec", "judge_elapsed_sec",
            "segments_count", "workers", "segment_indices", "command", "note"
        ]
        
        with self.csv_path.open("a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            
            writer.writerow({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "component": component,
                "prompt_version": prompt_version,
                "model": model,
                "temperature": temperature,
                "final_score": scores.get("final", 0),
                "groundedness": scores.get("groundedness", 0),
                "compliance": scores.get("compliance", 0),
                "note_quality": scores.get("note_quality", 0),
                "multimodal_use": scores.get("multimodal_use", 0),
                "input_tokens": input_tokens,
                "elapsed_sec": elapsed_sec,
                "stt_sec": stt_sec,
                "capture_sec": capture_sec,
                "vlm_sec": vlm_sec,
                "summarizer_sec": summarizer_sec,
                "judge_elapsed_sec": judge_elapsed_sec,
                "segments_count": segments_count,
                "workers": workers,
                "segment_indices": segment_indices,
                "command": command,
                "note": note
            })

