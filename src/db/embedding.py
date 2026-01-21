"""OpenRouter 기반 임베딩 생성 모듈.

Qwen3-Embedding-8B 모델을 사용하여 텍스트를 1024차원 벡터로 변환합니다.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log

logger = logging.getLogger(__name__)


# .env 파일 로드
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()

# OpenRouter 설정
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"
DEFAULT_DIMENSIONS = 1024


def _load_openrouter_key() -> str:
    """환경변수에서 OpenRouter API 키를 로드한다."""
    # 1. OPENROUTER_API_KEYS (콤마 구분)
    keys_env = os.getenv("OPENROUTER_API_KEYS", "")
    if keys_env:
        for item in re.split(r"[,\s]+", keys_env):
            if item.strip():
                return item.strip()
    
    # 2. OPENROUTER_API_KEY_1, _2, ...
    index = 1
    while True:
        key = os.getenv(f"OPENROUTER_API_KEY_{index}")
        if not key:
            break
        return key.strip()
    
    # 3. OPENROUTER_API_KEY
    single_key = os.getenv("OPENROUTER_API_KEY")
    if single_key:
        return single_key.strip()
    
    raise ValueError("OPENROUTER_API_KEY 환경변수가 설정되지 않았습니다.")


def get_embedding_client() -> OpenAI:
    """OpenRouter 클라이언트 생성."""
    api_key = _load_openrouter_key()
    
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def generate_embedding(
    text: str,
    model: str = DEFAULT_EMBEDDING_MODEL,
    dimensions: int = DEFAULT_DIMENSIONS,
) -> List[float]:
    """단일 텍스트의 임베딩 벡터 생성.
    
    Args:
        text: 임베딩할 텍스트
        model: 임베딩 모델 (기본: qwen/qwen3-embedding-8b)
        dimensions: 출력 차원 (기본: 1024, MRL 지원으로 32~4096 가능)
        
    Returns:
        1024차원 임베딩 벡터
        
    Raises:
        Exception: 3회 재시도 후에도 API 호출 실패 시
        
    Example:
        >>> embedding = generate_embedding("ELBO는 Evidence Lower Bound입니다.")
        >>> len(embedding)
        1024
    """
    if not text or not text.strip():
        logger.debug("빈 텍스트 - 제로 벡터 반환")
        return [0.0] * dimensions
    
    client = get_embedding_client()
    
    response = client.embeddings.create(
        model=model,
        input=text,
        dimensions=dimensions,
    )
    
    return response.data[0].embedding


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def generate_embeddings_batch(
    texts: List[str],
    model: str = DEFAULT_EMBEDDING_MODEL,
    dimensions: int = DEFAULT_DIMENSIONS,
) -> List[List[float]]:
    """여러 텍스트의 임베딩 벡터 일괄 생성.
    
    Args:
        texts: 임베딩할 텍스트 리스트
        model: 임베딩 모델
        dimensions: 출력 차원
        
    Returns:
        임베딩 벡터 리스트
        
    Raises:
        Exception: 3회 재시도 후에도 API 호출 실패 시
    """
    if not texts:
        return []
    
    # 빈 텍스트 필터링 및 인덱스 추적
    valid_indices = []
    valid_texts = []
    for i, text in enumerate(texts):
        if text and text.strip():
            valid_indices.append(i)
            valid_texts.append(text)
    
    if not valid_texts:
        logger.debug(f"유효한 텍스트 없음 - {len(texts)}개 제로 벡터 반환")
        return [[0.0] * dimensions for _ in texts]
    
    logger.info(f"임베딩 배치 생성: {len(valid_texts)}/{len(texts)}개 텍스트")
    client = get_embedding_client()
    
    response = client.embeddings.create(
        model=model,
        input=valid_texts,
        dimensions=dimensions,
    )
    
    # 결과를 원래 인덱스에 맞게 재배치
    results = [[0.0] * dimensions for _ in texts]
    for i, embedding_data in enumerate(response.data):
        original_index = valid_indices[i]
        results[original_index] = embedding_data.embedding
    
    logger.debug(f"임베딩 생성 완료: {len(valid_texts)}개")
    return results
