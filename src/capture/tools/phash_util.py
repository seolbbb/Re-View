"""
[Intent]
이미지의 시각적 특징을 압축하여 해시값으로 변환하고, 두 해시값 간의 유사도를 비교하는 유틸리티 모듈입니다.
주로 중복 이미지 감지를 위한 Perceptual Hash(pHash) 연산을 수행합니다.

[Usage]
- deduplicator.py: 중복 검사 시 이미지의 지문을 생성하고 비교하기 위해 사용됩니다.
"""

import cv2
import numpy as np

def compute_phash(image: np.ndarray) -> str:
    """
    [Usage File] src/capture/tools/deduplicator.py
    
    [Purpose]
    - 입력 이미지의 Perceptual Hash(pHash)를 계산합니다.
    - 이미지를 32x32로 축소하고 DCT(이산 코사인 변환)를 수행하여 저주파 성분의 평균값을 기준으로 해시를 생성합니다.
    
    [Args]
    - image (np.ndarray): 입력 BGR 이미지

    [Returns]
    - str: 64비트 해시 값을 표현하는 16진수 문자열
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32))
    dct = cv2.dct(np.float32(resized))
    dct_low_freq = dct[0:8, 0:8]
    avg = dct_low_freq.mean()
    diff = dct_low_freq > avg
    
    bits = diff.flatten()
    decimal_val = 0
    for i, bit in enumerate(bits):
        if bit:
            decimal_val += 2 ** i
            
    return hex(decimal_val)[2:]

def hamming_distance(hash1: str, hash2: str) -> int:
    """
    [Usage File] src/capture/tools/deduplicator.py
    
    [Purpose]
    - 두 16진수 해시 문자열 간의 해밍 거리(Hamming Distance)를 계산합니다.
    - 해밍 거리는 서로 다른 비트의 개수를 의미하며, 값이 작을수록 두 이미지가 유사함을 나타냅니다.
    
    [Args]
    - hash1 (str): 첫 번째 해시 문자열
    - hash2 (str): 두 번째 해시 문자열

    [Returns]
    - int: 서로 다른 비트의 개수 (0 ~ 64)
    """
    try:
        val1 = int(hash1, 16)
        val2 = int(hash2, 16)
        xor_val = val1 ^ val2
        return bin(xor_val).count('1')
    except ValueError:
        return 64
