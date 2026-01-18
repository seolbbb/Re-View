# -*- coding: utf-8 -*-
"""JSONL Summary to Markdown converter."""

import json
import argparse
from pathlib import Path

def convert_jsonl_to_md(jsonl_path: Path, output_path: Path):
    """Convert segment_summaries.jsonl to human-readable Markdown."""
    if not jsonl_path.exists():
        print(f"[ERROR] File not found: {jsonl_path}")
        return

    print(f"[INFO] Reading: {jsonl_path}")

    with jsonl_path.open("r", encoding="utf-8") as f:
        lines = [l for l in f.readlines() if l.strip()]

    with output_path.open("w", encoding="utf-8") as out:
        # Header
        out.write(f"# Segment Summaries Report\n\n")
        out.write(f"| Property | Value |\n")
        out.write(f"|:---|:---|\n")
        out.write(f"| Source File | `{jsonl_path.name}` |\n")
        out.write(f"| Total Segments | {len(lines)} |\n\n")

        for line in lines:
            try:
                data = json.loads(line)
                seg_id = data.get("segment_id", "Unknown")
                start_ms = data.get("start_ms", 0)
                end_ms = data.get("end_ms", 0)
                summary = data.get("summary", {})
                
                # Segment Header with timestamp
                start_sec = start_ms // 1000
                end_sec = end_ms // 1000
                start_str = f"{start_sec//60:02d}:{start_sec%60:02d}"
                end_str = f"{end_sec//60:02d}:{end_sec%60:02d}"
                
                out.write(f"### Segment {seg_id} ({start_str}-{end_str})\n")
                
                # Format 1 (ReViewFeature): source_refs with stt_ids/vlm_ids
                # Format 2 (ReView): transcript_units/visual_units with unit_id
                source_refs = data.get("source_refs", {})
                valid_stt = set(source_refs.get("stt_ids", []))
                valid_vlm = set(source_refs.get("vlm_ids", []))
                
                # If source_refs is empty, extract from transcript_units/visual_units
                if not valid_stt and not valid_vlm:
                    transcript_units = data.get("transcript_units", [])
                    visual_units = data.get("visual_units", [])
                    valid_stt = set(u.get("unit_id", "") for u in transcript_units if u.get("unit_id"))
                    valid_vlm = set(u.get("unit_id", "") for u in visual_units if u.get("unit_id"))
                
                def validate_and_format(refs, context_str=""):
                    if not refs:
                        return ""
                    
                    t_refs = []
                    v_refs = []
                    
                    for r in refs:
                        # Check strictly against valid input IDs
                        if r in valid_stt:
                            t_refs.append(r)
                        elif r in valid_vlm:
                            v_refs.append(r)
                        else:
                            # Heuristic check
                            if r.startswith("t") or r.startswith("stt"):
                                t_refs.append(r)
                            elif r.startswith("v") or r.startswith("vlm"):
                                v_refs.append(r)
                            else:
                                if r not in valid_stt and r not in valid_vlm:
                                    # print(f"[WARN] Segment {seg_id} [{context_str}]: ID '{r}' not found in source_refs!")
                                    pass

                    parts = []
                    t_refs_str = str(t_refs).replace("'", "")
                    v_refs_str = str(v_refs).replace("'", "")
                    parts.append(f"text_ids : {t_refs_str}")
                    parts.append(f"vlm_ids : {v_refs_str}")
                    return ", ".join(parts)

                # aggregation
                items_by_source = {
                    "direct": [],
                    "background": [],
                    "inferred": []
                }

                # Helper to add item
                def add_item(s_type, text, refs, context_str):
                    if s_type not in items_by_source:
                        s_type = "direct" 
                    items_by_source[s_type].append({
                        "text": text,
                        "refs": refs,
                        "context": context_str
                    })

                # 1. Collect Bullets
                bullets = summary.get("bullets", [])
                for i, b in enumerate(bullets, 1):
                    claim = b.get("claim", "")
                    s_type = b.get("source_type", "direct")
                    notes = b.get("notes", "")
                    refs = b.get("evidence_refs", [])
                    
                    text = f"({seg_id}-{i}) {claim}"
                    if notes:
                        text += f"\n    - notes: {notes}"
                    add_item(s_type, text, refs, f"Summary-{i}")

                # 2. Collect Definitions
                definitions = summary.get("definitions", [])
                for d in definitions:
                    term = d.get("term", "")
                    defin = d.get("definition", "")
                    s_type = d.get("source_type", "background")
                    refs = d.get("evidence_refs", [])
                    
                    text = f"**{term}**: {defin}"
                    add_item(s_type, text, refs, f"Definition-{term}")

                # 3. Collect Explanations
                explanations = summary.get("explanations", [])
                for e in explanations:
                    point = e.get("point", "")
                    s_type = e.get("source_type", "inferred")
                    refs = e.get("evidence_refs", [])
                    
                    add_item(s_type, point, refs, "Explanation")

                # Print Groups
                headers_map = {
                    "direct": "핵심 내용 (Direct / Recall)",
                    "inferred": "심화/추론 (Inferred / Logic)",
                    "background": "배경 지식 (Background)"
                }
                
                display_order = ["direct", "background", "inferred"]

                for key in display_order:
                    items = items_by_source.get(key, [])
                    if not items:
                        continue
                        
                    header_title = headers_map.get(key, key.capitalize())
                    out.write(f"- [{key}] {header_title}\n")
                    
                    for item in items:
                        out.write(f"  - {item['text']}\n")
                        ev_str = validate_and_format(item['refs'], item['context'])
                        if ev_str:
                            out.write(f"    - evidence: {ev_str}\n")
                out.write("\n")

            except json.JSONDecodeError:
                print(f"[WARN] Failed to parse line: {line[:50]}...")
                continue
    
    print(f"[SUCCESS] Markdown report generated: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSONL summaries to Markdown")
    parser.add_argument("file", help="Path to JSONL file")
    args = parser.parse_args()
    
    input_path = Path(args.file)
    output_path = input_path.with_suffix('.md')
    
    convert_jsonl_to_md(input_path, output_path)

