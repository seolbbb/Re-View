import whisper
import os
import json
import subprocess
import torch
from datetime import datetime

class AudioProcessor:
    def __init__(self, model_size='base'):
        """
        Whisper 모델 초기화
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Loading Whisper model ('{model_size}') on {self.device}...")
        try:
            self.model = whisper.load_model(model_size, device=self.device)
            print("✅ Model loaded successfully.")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise

    def extract_audio(self, video_path, output_audio_path="temp_audio.wav"):
        """
        비디오에서 오디오 추출 (ffmpeg 사용)
        """
        print(f"🔊 Extracting audio from {video_path}...")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # ffmpeg 명령어로 오디오 추출 (overwrite, quiet, audio codec pcm_s16le, ar 16000, ac 1)
        command = [
            "ffmpeg", "-y", 
            "-i", video_path, 
            "-vn", # Video disable
            "-acodec", "pcm_s16le", 
            "-ar", "16000", 
            "-ac", "1", 
            output_audio_path
        ]
        
        try:
            # subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # 디버깅을 위해 stderr는 출력하도록 변경 가능
            subprocess.run(command, check=True, stderr=subprocess.PIPE)
            print(f"✅ Audio extracted to {output_audio_path}")
            return output_audio_path
        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg failed: {e}")
            raise RuntimeError("FFmpeg is required. Please install it and add to PATH.")
        except FileNotFoundError:
            print("❌ FFmpeg not found.")
            raise RuntimeError("FFmpeg executable not found. Please install FFmpeg.")

    def transcribe(self, audio_path, language='ko'):
        """
        오디오 파일을 텍스트로 변환 (STT)
        """
        print(f"📝 Transcribing audio ({language})...")
        
        try:
            # Whisper 실행
            result = self.model.transcribe(audio_path, language=language)
            
            segments = []
            for seg in result['segments']:
                segments.append({
                    'start': seg['start'],
                    'end': seg['end'],
                    'text': seg['text'].strip()
                })
            
            print(f"✅ Transcription complete. ({len(segments)} segments)")
            return segments
            
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            raise

    def process_video(self, video_path, save_json=True):
        """
        비디오 -> 오디오 -> STT 전체 파이프라인
        """
        temp_audio = "temp_audio_for_stt.wav"
        try:
            # 1. 오디오 추출
            self.extract_audio(video_path, temp_audio)
            
            # 2. STT 변환
            transcript = self.transcribe(temp_audio)
            
            # 3. 결과 저장 (옵션)
            if save_json:
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                json_path = f"{base_name}_stt_result.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(transcript, f, ensure_ascii=False, indent=2)
                print(f"💾 Result saved to {json_path}")
            
            return transcript
            
        finally:
            # 임시 파일 삭제
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
                print("🧹 Temporary audio file removed.")

if __name__ == "__main__":
    # 테스트 실행
    # 경로 수정: 상대 경로 사용
    video_file = os.path.join("data", "input", "1배속.mp4")
    
    if os.path.exists(video_file):
        processor = AudioProcessor(model_size='base')
        result = processor.process_video(video_file)
        
        # 결과 일부 출력
        print("\n--- Transcription Preview ---")
        for item in result[:5]:
            print(f"[{item['start']:.2f}s ~ {item['end']:.2f}s] {item['text']}")
    else:
        print(f"⚠ Test video file not found: {video_file}")
        print("Please check the path.")
