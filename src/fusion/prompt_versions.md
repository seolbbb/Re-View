# Summarizer Prompt Versions

프롬프트 버전은 `config/fusion/prompts.yaml`에서 관리합니다.

- 기본 버전은 `config/fusion/config.yaml`의 `summarizer.prompt_version`으로 결정됩니다.
- 템플릿 치환 토큰은 `{{CLAIM_MAX_CHARS}}`, `{{BULLETS_MIN}}`,
  `{{BULLETS_MAX}}`, `{{SEGMENTS_TEXT}}` 등을 사용할 수 있습니다.
