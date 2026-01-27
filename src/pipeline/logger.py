"""
[Intent]
전처리 파이프라인의 핵심 로깅 시스템을 제공하는 모듈입니다. 
오디오 추출, 캡처, STT와 같이 동시에 실행되는 여러 컴포넌트의 상태를 
동일한 라인에 컬럼(Column) 형태로 평면화하여 실시간 진행 상황을 직관적으로 보여줍니다.

[Usage]
- run_preprocess_pipeline.py의 병렬 실행 구간에서 각 컴포넌트의 상태 변화를 기록할 때 활용됩니다.
- src.pipeline.stages 블록 등 각 단계별 실행 함수 내에서 상태 정보를 전파하기 위해 사용됩니다.

[Usage Method]
- pipeline_logger.log("Audio", "Extracting...") 와 같이 호출하여 상태를 실시간 업데이트합니다.
- 멀티스레딩 환경에서도 안전하게 출력될 수 있도록 스레드 락(Lock)이 적용되어 있습니다.
"""

import sys
import threading
from typing import Dict, Optional


class ColumnLogger:
    """
    [Class Purpose]
    병렬로 진행되는 여러 작업의 상태를 단일 라인 로그로 출력하는 컬럼형 로거 클래스입니다.
    이전 출력과 동일한 "DONE" 메시지가 반복되는 것을 방지하여 로그의 깔끔함을 유지합니다.
    """

    def __init__(self):
        """
        [Method Purpose]
        로거의 초기 상태를 설정합니다.
        
        [Details]
        - self.lock: 스레드 안전한 출력을 위한 Lock 객체
        - self.states: 각 컴포넌트(Audio, Capture, STT)의 현재 상태 메시지 저장소
        - self.last_printed: DONE 메시지의 중복 출력을 방지하기 위한 마지막 출력 기록
        """
        self.lock = threading.Lock()
        # 기본 컴포넌트 상태 정의
        self.states = {
            "Audio": "WAITING",
            "Capture": "WAITING",
            "STT": "WAITING"
        }
        # 완료(DONE) 상태 중복 출력 방지용 기록
        self.last_printed: Dict[str, Optional[str]] = {
            "Audio": None,
            "Capture": None,
            "STT": None
        }
        # 로그 출력 시 가시성을 위한 컬럼 폭 설정
        self.max_content_width = 45

    def log(self, component: str, message: str) -> None:
        """
        [Usage File] run_preprocess_pipeline.py, stages.py, hybrid_extractor.py 등 전 영역
        [Purpose] 특정 컴포넌트의 상태를 업데이트하고 전체 컴포넌트의 합산 상태를 화면에 출력합니다.
        [Connection] 표준 출력(stdout)을 통한 터미널 인터페이스 제공
        
        [Args]
        - component (str): 상태를 업데이트할 컴포넌트 명칭 ("Audio", "Capture", "STT")
        - message (str): 표시할 상태 메시지 (예: "Extracting...", "DONE (Task: 1.5s)")
        
        [Internal Logic]
        1. 스레드 Lock을 획득하여 공유 상태 접근의 안전성을 보장합니다.
        2. 입력된 컴포넌트의 상태를 업데이트합니다.
        3. 각 컴포넌트 순회 시, 이미 완료(DONE)되어 출력된 적이 있는 메시지는 중복 출력을 피하기 위해 생략 조건을 체크합니다.
        4. 내용이 너무 길면 지정된 폭에 맞춰 생략 기호(...) 처리를 수행합니다.
        5. 유효한 출력 내용이 존재하는 경우에만 최종 합산 라인을 출력합니다.
        """
        with self.lock:
            # 상태 정보 갱신
            if component in self.states:
                self.states[component] = message
            
            line_parts = []
            
            # 정의된 순서대로 컬럼 생성
            for name in ["Audio", "Capture", "STT"]:
                curr_msg = self.states.get(name, "")
                last_msg = self.last_printed.get(name)
                
                # 중복된 DONE 메시지는 출력하지 않음
                if curr_msg.startswith("DONE") and last_msg == curr_msg:
                    continue
                
                # 신규 DONE 메시지는 기록에 저장
                if curr_msg.startswith("DONE"):
                    self.last_printed[name] = curr_msg
                
                # 표시용 텍스트 가공 (너무 길면 잘라내기)
                if len(curr_msg) > self.max_content_width:
                     disp_msg = curr_msg[:self.max_content_width - 3] + "..."
                else:
                     disp_msg = curr_msg
                
                column_msg = f"[{name}] {disp_msg}"
                line_parts.append(column_msg)
            
            # 출력할 유효 내용이 있는 경우에만 실행
            if line_parts:
                final_line = "    ".join(line_parts)
                print(final_line)
                sys.stdout.flush()


# [Singleton] 시스템 전체에서 공유되는 로거 인스턴스
pipeline_logger = ColumnLogger()
