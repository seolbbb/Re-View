
import sys
from pathlib import Path

# Add repo root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.pipeline.stages import run_fusion_pipeline
from src.pipeline.benchmark import BenchmarkTimer
from src.fusion.summarizer import run_summarizer
from src.fusion.config import load_config
from src.fusion.sync_engine import run_sync_engine

def main():
    video_root = ROOT / "data/outputs/sample4"
    config_path = video_root / "config.yaml"
    
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return

    config_bundle = load_config(config_path)

    # Override prompt version for verification
    # Changed from direct assignment to using an overrides dictionary
    overrides = {
        "fusion": {
            "summarizer": {
                "limit": 1, # This limit is for the config, not the run_summarizer call below
                "prompt_version": "v2"  # Changed from opt_v7 to v2
            }
        }
    }
    # Assuming config_bundle has a method to apply overrides, or we apply it manually
    # For now, let's manually apply the prompt_version override as per the original structure
    config_bundle.raw.summarizer.prompt_version = overrides["fusion"]["summarizer"]["prompt_version"]
    print(f"Running with prompt version: {config_bundle.raw.summarizer.prompt_version}")

    print("Running sync engine...")
    run_sync_engine(config_bundle)

    print("Running summarizer...")
    run_summarizer(config_bundle)  # Full run
    timer = BenchmarkTimer()
    timer.start_total()
    
    info = run_fusion_pipeline(config_path, timer=timer)
    
    timer.end_total()
    print("Fusion pipeline completed successfully.")
    print(f"Segment count: {info.get('segment_count')}")

if __name__ == "__main__":
    main()
