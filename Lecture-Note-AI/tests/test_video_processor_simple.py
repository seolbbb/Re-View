import cv2
import numpy as np
import os
import shutil
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.capture.video_processor import VideoProcessor

def create_test_video(filename):
    """Create a video with distinct scene changes using random noise"""
    width, height = 640, 360
    fps = 30
    duration_per_scene = 1 # seconds
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    # Scene 1: Noise A
    np.random.seed(42)
    frame1 = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    for _ in range(fps * duration_per_scene):
        out.write(frame1)
        
    # Scene 2: Noise B
    np.random.seed(43)
    frame2 = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    for _ in range(fps * duration_per_scene):
        out.write(frame2)
        
    # Scene 3: Noise C
    np.random.seed(44)
    frame3 = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    for _ in range(fps * duration_per_scene):
        out.write(frame3)
        
    out.release()
    print(f"Created test video: {filename}")

def test_extraction():
    video_path = "test_video.mp4"
    output_dir = "test_output"
    
    # Cleanup previous run
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        
    create_test_video(video_path)
    
    processor = VideoProcessor()
    # Use small min_interval to allow capturing all transitions
    keyframes = processor.extract_keyframes(video_path, output_dir, threshold=10, min_interval=0.5, verbose=True)
    
    print(f"Captured {len(keyframes)} keyframes.")
    
    # Expected: 
    # 1. First frame (Black)
    # 2. Transition to White (approx 1.0s)
    # 3. Transition to Black (approx 2.0s)
    # Total 3 frames expected.
    
    if len(keyframes) >= 3:
        print("✅ Test Passed: Captured expected scene changes.")
    else:
        print(f"❌ Test Failed: Expected at least 3 frames, got {len(keyframes)}.")
        for k in keyframes:
            print(f" - {k['timestamp']:.2f}s")

    # Cleanup
    if os.path.exists(video_path):
        os.remove(video_path)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

if __name__ == "__main__":
    test_extraction()
