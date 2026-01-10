
import argparse
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys

# Add src to sys.path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from src.fusion.config import ConfigBundle, load_config
from src.fusion.io_utils import read_jsonl, print_jsonl_head, ensure_output_root, update_token_usage
from src.fusion.summarizer import (
    _init_gemini_client,
    _build_batch_prompt,
    _build_response_schema,
    _run_with_retries,
    _parse_json_response,
    _validate_summary_payload,
    _normalize_evidence_refs,
    _normalize_confidence,
    _normalize_source_type,
    GeminiClientBundle
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_batch(
    segments_batch: List[Dict[str, Any]],
    client_bundle: GeminiClientBundle,
    config: ConfigBundle,
    claim_max_chars: int,
    bullets_min: int,
    bullets_max: int
) -> Dict[str, Any]:
    """Process a single batch of segments: build prompt, call LLM, parse response."""
    prompt = _build_batch_prompt(segments_batch, claim_max_chars, bullets_min, bullets_max)
    
    # Measure input tokens
    input_tokens = 0
    try:
        client = client_bundle.client
        token_result = client.models.count_tokens(
            model=client_bundle.model,
            contents=prompt
        )
        input_tokens = token_result.total_tokens
    except Exception as exc:
        logger.warning(f"Failed to count tokens: {exc}")

    response_schema = _build_response_schema()
    
    # Call LLM
    llm_text = _run_with_retries(
        client_bundle,
        prompt,
        response_schema,
        config.raw.summarizer.temperature,
        config.raw.llm_gemini.response_mime_type,
        config.raw.llm_gemini.timeout_sec,
        config.raw.llm_gemini.max_retries,
        config.raw.llm_gemini.backoff_sec,
    )

    # Parse and Validate with Repair Logic
    attempts = config.raw.summarizer.json_repair_attempts
    # Import internal repair function
    from src.fusion.summarizer import _repair_prompt

    valid_payload = None
    current_text = llm_text
    last_error: Optional[Exception] = None
    
    for attempt_idx in range(attempts + 1):
        try:
            payload = _parse_json_response(current_text)
            if not isinstance(payload, list):
                raise ValueError("Response is not a JSON array")
            
            # Create a map to validate against segments_batch
            payload_map = {}
            for item in payload:
                if not isinstance(item, dict): continue
                if "segment_id" in item and "summary" in item:
                    try:
                        sid = int(item["segment_id"])
                        payload_map[sid] = item["summary"]
                    except:
                        pass
            
            # Validate
            batch_results = []
            for seg in segments_batch:
                sid = seg["segment_id"]
                if sid not in payload_map:
                    # In a strict pipeline we might error, or try to salvage.
                    # For benchmark, let's treat as error to force retry.
                    raise ValueError(f"Missing summary for segment {sid}")
                
                raw_summary = payload_map[sid]
                validated = _validate_summary_payload(
                    raw_summary, sid, claim_max_chars, bullets_min, bullets_max
                )
                batch_results.append({
                    "segment_id": sid,
                    "summary": validated
                })
            
            if len(batch_results) != len(segments_batch):
                 raise ValueError("Incomplete batch results")
            
            valid_payload = batch_results
            break 

        except Exception as e:
            last_error = e
            if attempt_idx < attempts:
                logger.info(f"JSON Parse/Validate failed: {e}. Attempting repair {attempt_idx+1}/{attempts}")
                repair_prompt = _repair_prompt(current_text, bullets_min, bullets_max, claim_max_chars)
                current_text = _run_with_retries(
                    client_bundle,
                    repair_prompt,
                    response_schema, 
                    config.raw.summarizer.temperature,
                    config.raw.llm_gemini.response_mime_type,
                    config.raw.llm_gemini.timeout_sec,
                    config.raw.llm_gemini.max_retries,
                    config.raw.llm_gemini.backoff_sec,
                )
            else:
                 logger.error(f"Final repair attempt failed: {e}")

    if valid_payload is None:
        raise last_error or RuntimeError("Failed to generate valid summaries after retries")

    # Measure output tokens
    output_tokens = 0
    try:
         token_result_out = client.models.count_tokens(
            model=client_bundle.model,
            contents=str(valid_payload)
        )
         output_tokens = token_result_out.total_tokens
    except:
        pass
        
    return {
        "results": valid_payload,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens
    }


def chunk_list(data, chunk_size):
    """Yield successive chunks from data."""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

def main():
    parser = argparse.ArgumentParser(description="Benchmark Summarizer Batch Sizes")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--input", required=True, help="Path to input segments_units.jsonl")
    parser.add_argument("--batch-sizes", default="1,2,4,8", help="Comma-separated batch sizes")
    parser.add_argument("--output-dir", required=True, help="Directory to save results")
    parser.add_argument("--max-workers", type=int, default=4, help="Max concurrent workers (default: 4)")
    parser.add_argument("--request-interval", type=float, default=1.0, help="Min interval between requests in seconds")
    args = parser.parse_args()

    config_path = Path(args.config)
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    batch_sizes = [int(s.strip()) for s in args.batch_sizes.split(",")]

    # Load Config
    config = load_config(config_path)
    
    # Load Segments
    segments = list(read_jsonl(input_path))
    logger.info(f"Loaded {len(segments)} segments from {input_path}")
    
    ensure_output_root(output_dir)
    
    # Init Gemini Client
    client_bundle = _init_gemini_client(config)
    
    # Summarizer settings
    bullets_min = config.raw.summarizer.bullets_per_segment_min
    bullets_max = config.raw.summarizer.bullets_per_segment_max
    claim_max_chars = config.raw.summarizer.claim_max_chars

    report = {
        "total_segments": len(segments),
        "results": []
    }

    for bs in batch_sizes:
        logger.info(f"--- Running Benchmark for Batch Size: {bs} ---")
        
        # Split into chunks
        chunks = list(chunk_list(segments, bs))
        logger.info(f"Split into {len(chunks)} chunks")

        start_time = time.perf_counter()
        
        chunk_results = []
        total_input_tokens = 0
        total_output_tokens = 0
        
        # Parallel Execution using ThreadPoolExecutor
        import concurrent.futures
        
        requests_interval = args.request_interval
        max_workers = min(len(chunks), args.max_workers)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {}
            for chunk in chunks:
                future = executor.submit(
                    process_batch,
                    chunk,
                    client_bundle,
                    config,
                    claim_max_chars,
                    bullets_min,
                    bullets_max
                )
                future_to_chunk[future] = chunk
                time.sleep(requests_interval)  # Throttle requests
            
            for future in concurrent.futures.as_completed(future_to_chunk):
                try:
                    res = future.result()
                    chunk_results.extend(res["results"])
                    total_input_tokens += res["input_tokens"]
                    total_output_tokens += res["output_tokens"]
                except Exception as exc:
                    logger.error(f"Chunk processing failed: {exc}")
        
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        
        # Sort results by segment_id
        chunk_results.sort(key=lambda x: x["segment_id"])
        
        # Save output
        out_file = output_dir / f"segment_summaries_batch{bs}.jsonl"
        with out_file.open("w", encoding="utf-8") as f:
            for item in chunk_results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        logger.info(f"Batch Size {bs}: {elapsed:.2f}s, Tokens: {total_input_tokens}/{total_output_tokens}")
        
        report["results"].append({
            "batch_size": bs,
            "num_requests": len(chunks),
            "latency_sec": elapsed,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "latency_per_segment": elapsed / len(segments)
        })

    # Save Report
    report_file = output_dir / "benchmark_report.json"
    with report_file.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Benchmark finished. Report saved to {report_file}")
    
    # Print summary table
    print("\nBenchmark Summary:")
    print(f"{'Batch Size':<12} {'Latency (s)':<12} {'Input Tok':<12} {'Output Tok':<12}")
    print("-" * 50)
    for res in report["results"]:
        print(f"{res['batch_size']:<12} {res['latency_sec']:<12.2f} {res['input_tokens']:<12} {res['output_tokens']:<12}")

if __name__ == "__main__":
    main()
