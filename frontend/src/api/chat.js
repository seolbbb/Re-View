import { post, BASE_URL, ApiError } from './client';

export function sendChatMessage({ videoId, message, sessionId }) {
  return post('/api/chat', {
    video_id: videoId,
    message,
    session_id: sessionId || undefined,
  });
}

export function streamChatMessage({
  videoId,
  message,
  sessionId,
  onChunk,
  onSessionId,
  onDone,
  onError,
}) {
  const controller = new AbortController();
  let doneCalled = false;

  const emitDone = (payload) => {
    if (doneCalled) return;
    doneCalled = true;
    if (onDone) onDone(payload);
  };

  const handleEventBlock = (block) => {
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

    if (event === 'session' && data && data.session_id && onSessionId) {
      onSessionId(data.session_id);
      return;
    }
    if (event === 'message' && data && onChunk) {
      onChunk(data.text || '', Boolean(data.is_final));
      return;
    }
    if (event === 'done') {
      emitDone(data);
      return;
    }
    if (event === 'error' && onError) {
      const messageText = data && data.error ? data.error : 'Streaming error';
      onError(new Error(messageText));
    }
  };

  const start = async () => {
    try {
      const res = await fetch(`${BASE_URL}/api/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_id: videoId,
          message,
          session_id: sessionId || undefined,
        }),
        signal: controller.signal,
      });

      if (!res.ok) {
        let messageText = res.statusText;
        try {
          const body = await res.json();
          messageText = body.detail || body.message || messageText;
        } catch {
          // ignore parse errors
        }
        throw new ApiError(res.status, messageText);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';
      let isDone = false;

      while (!isDone) {
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

      if (buffer.trim()) {
        handleEventBlock(buffer.trim());
      }

      emitDone();
    } catch (err) {
      if (err && err.name === 'AbortError') return;
      if (onError) onError(err);
    }
  };

  start();
  return controller;
}
