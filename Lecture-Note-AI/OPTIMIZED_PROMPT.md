# Optimized Prompt for Lecture Note AI Update

## Objective

Update the `Lecture-Note-AI` system to process ClovaSpeech JSON outputs and capture video keyframes using the existing "previous method" (scene change detection + human removal).

## Inputs

- **Video Directory**: `src/data/input` (Contains `.mp4` files)
- **JSON Directory**: `src/data/output` (Contains ClovaSpeech `.json` files)

## Requirements

### 1. JSON Processing Module

- **Goal**: Parse the ClovaSpeech JSON file to make it readable.
- **Logic**:
  - Read the JSON file.
  - Extract the `segments` list.
  - For each segment, extract the `start` time (convert to `MM:SS` format) and `text`.
  - Save the result as a text file (e.g., `[MM:SS] Text content...`) or a simplified JSON in the `output` directory.
  - **Filename**: Same basename as the JSON file but with `_readable.txt` or `_parsed.json`.

### 2. Video Capture Module

- **Goal**: Capture keyframes from the video when the screen changes.
- **Logic**:
  - Reuse the existing `VideoProcessor` class in `src/capture/video_processor.py`.
  - Iterate through video files in `src/data/input`.
  - Apply `extract_keyframes` with appropriate parameters (threshold, min_interval).
  - Save captured frames to `src/data/output/{video_name}_frames`.

### 3. Orchestration

- Create a main script (e.g., `src/process_content.py`) that:
  1. Scans `src/data/input` for videos.
  2. Scans `src/data/output` for corresponding JSON files.
  3. Runs the JSON parser.
  4. Runs the Video Processor.

## Deliverables

- `src/data/json_parser.py`: New module for JSON handling.
- `src/process_content.py`: Main script to run the workflow.
- Updated `src/capture/video_processor.py` (if minor tweaks are needed, otherwise keep as is).
