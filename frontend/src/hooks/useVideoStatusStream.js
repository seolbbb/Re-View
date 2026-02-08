import { useEffect, useRef, useState, useCallback } from 'react';
import { BASE_URL } from '../api/client';
import { supabase } from '../lib/supabase';

/**
 * SSE를 통해 비디오 상태 및 요약을 실시간으로 스트리밍합니다.
 *
 * @param {string} videoId - 비디오 ID
 * @param {Object} options
 * @param {boolean} options.enabled - 스트리밍 활성화 여부 (기본 true)
 * @param {Function} options.onStatus - 상태 변경 시 콜백 (data) => void
 * @param {Function} options.onSummaries - 새 요약 추가 시 콜백 (data) => void
 * @param {Function} options.onDone - 처리 완료/실패 시 콜백 (data) => void
 * @param {Function} options.onError - 에러 발생 시 콜백 (error) => void
 *
 * @returns {{ status, summaries, error, isConnected, reconnect }}
 */
export default function useVideoStatusStream(
  videoId,
  { enabled = true, onStatus, onSummaries, onDone, onError } = {}
) {
  const [status, setStatus] = useState(null);
  const [summaries, setSummaries] = useState(null);
  const [error, setError] = useState(null);
  const [isConnected, setIsConnected] = useState(false);

  const controllerRef = useRef(null);
  const retryCountRef = useRef(0);
  const maxRetries = 5;
  const stoppedRef = useRef(false);

  // 콜백 refs (의존성 문제 방지)
  const onStatusRef = useRef(onStatus);
  const onSummariesRef = useRef(onSummaries);
  const onDoneRef = useRef(onDone);
  const onErrorRef = useRef(onError);

  useEffect(() => {
    onStatusRef.current = onStatus;
    onSummariesRef.current = onSummaries;
    onDoneRef.current = onDone;
    onErrorRef.current = onError;
  }, [onStatus, onSummaries, onDone, onError]);

  const handleEventBlock = useCallback((block) => {
    if (!block) return;
    const lines = block.split('\n');
    let event = 'message';
    const dataLines = [];

    for (const line of lines) {
      if (!line || line.startsWith(':')) continue;
      if (line.startsWith('event:')) {
        event = line.slice(6).trim() || 'message';
        continue;
      }
      if (line.startsWith('data:')) {
        dataLines.push(line.slice(5).trim());
      }
    }

    const dataText = dataLines.join('\n');
    if (!dataText) return;

    let data = dataText;
    try {
      data = JSON.parse(dataText);
    } catch {
      // keep as plain text
    }

    if (event === 'status' && data) {
      setStatus(data);
      if (onStatusRef.current) onStatusRef.current(data);
      return;
    }

    if (event === 'summaries' && data) {
      setSummaries(data);
      if (onSummariesRef.current) onSummariesRef.current(data);
      return;
    }

    if (event === 'done' && data) {
      if (onDoneRef.current) onDoneRef.current(data);
      return;
    }

    if (event === 'error' && data) {
      const errorMsg = data.error || 'Stream error';
      setError(new Error(errorMsg));
      if (onErrorRef.current) onErrorRef.current(new Error(errorMsg));
    }
  }, []);

  const connect = useCallback(async () => {
    if (!videoId || stoppedRef.current) return;

    // 이전 연결 정리
    if (controllerRef.current) {
      controllerRef.current.abort();
    }

    const controller = new AbortController();
    controllerRef.current = controller;

    try {
      const headers = { Accept: 'text/event-stream' };
      if (supabase) {
        try {
          const { data: { session } } = await supabase.auth.getSession();
          if (session?.access_token) {
            headers.Authorization = `Bearer ${session.access_token}`;
          }
        } catch {
          // ignore auth header errors
        }
      }

      const res = await fetch(`${BASE_URL}/videos/${videoId}/status/stream`, {
        method: 'GET',
        headers,
        signal: controller.signal,
      });

      if (!res.ok) {
        // If the video was deleted, stop retrying and let the caller decide how to react.
        if (res.status === 404) {
          const err = new Error('Video not found');
          err.status = 404;
          stoppedRef.current = true;
          setIsConnected(false);
          setError(err);
          if (onErrorRef.current) onErrorRef.current(err);
          return;
        }
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }

      setIsConnected(true);
      setError(null);
      retryCountRef.current = 0;

      const reader = res.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';
      let isDone = false;

      while (!isDone && !stoppedRef.current) {
        const { value, done } = await reader.read();
        isDone = done;

        if (value) {
          buffer += decoder.decode(value, { stream: !isDone });
        }

        const parts = buffer.replace(/\r\n/g, '\n').split('\n\n');
        buffer = parts.pop() || '';

        for (const part of parts) {
          handleEventBlock(part.trim());
        }
      }

      // 남은 버퍼 처리
      if (buffer.trim()) {
        handleEventBlock(buffer.trim());
      }

      setIsConnected(false);
    } catch (err) {
      if (err.name === 'AbortError') {
        setIsConnected(false);
        return;
      }

      setIsConnected(false);
      setError(err);
      if (onErrorRef.current) onErrorRef.current(err);

      // 지수 백오프 재연결
      if (retryCountRef.current < maxRetries && !stoppedRef.current) {
        const delay = Math.min(1000 * Math.pow(2, retryCountRef.current), 30000);
        retryCountRef.current += 1;
        setTimeout(() => {
          if (!stoppedRef.current) connect();
        }, delay);
      }
    }
  }, [videoId, handleEventBlock]);

  const reconnect = useCallback(() => {
    stoppedRef.current = false;
    retryCountRef.current = 0;
    setError(null);
    connect();
  }, [connect]);

  useEffect(() => {
    stoppedRef.current = false;

    if (!enabled || !videoId) {
      // 비활성화 시 연결 종료
      if (controllerRef.current) {
        controllerRef.current.abort();
        controllerRef.current = null;
      }
      setIsConnected(false);
      return;
    }

    connect();

    return () => {
      stoppedRef.current = true;
      if (controllerRef.current) {
        controllerRef.current.abort();
        controllerRef.current = null;
      }
      setIsConnected(false);
    };
  }, [videoId, enabled, connect]);

  return { status, summaries, error, isConnected, reconnect };
}
