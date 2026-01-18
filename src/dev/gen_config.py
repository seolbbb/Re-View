import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.pipeline.stages import generate_fusion_config

video_name = "sample4"
output_root = ROOT / "data" / "outputs" / video_name

generate_fusion_config(
    template_config=ROOT / "config" / "fusion" / "settings.yaml",
    output_config=output_root / "config.yaml",
    repo_root=ROOT,
    stt_json=output_root / "stt.json",
    vlm_json=output_root / "vlm.json",
    manifest_json=output_root / "capture.json",
    output_root=output_root,
)
print(f"Generated config.yaml at: {output_root / 'config.yaml'}")
