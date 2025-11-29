import cv2
import os

def mask_video(input_path, output_path, box_x=400, box_y=300, box_size=50):
    if not os.path.exists(input_path):
        print(f"❌ Input video not found: {input_path}")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("❌ Failed to open video.")
        return

    # 원본 속성 가져오기
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"🎬 Processing: {input_path}")
    print(f"ℹ️ Info: {width}x{height}, {fps} fps, {total_frames} frames")
    print(f"🖌 Masking: Box at ({box_x}, {box_y}), Size {box_size}x{box_size}")

    # VideoWriter 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 검은색 박스 그리기
        # (x, y) ~ (x+w, y+h)
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_size, box_y + box_size), (0, 0, 0), -1)

        out.write(frame)
        frame_count += 1

        if frame_count % 100 == 0:
            print(f"⏳ Processed {frame_count}/{total_frames} frames...", end='\r')

    cap.release()
    out.release()
    print(f"\n✅ Done! Saved to: {output_path}")

if __name__ == "__main__":
    input_file = r"C:\Users\irubw\geminiProject\screentime_mvp\screentime_MVP\dirty_ex2.mp4"
    output_file = r"C:\Users\irubw\geminiProject\screentime_mvp\screentime_MVP\dirty_ex2_masked.mp4"
    
    mask_video(input_file, output_file)
