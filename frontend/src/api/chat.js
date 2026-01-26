import { post } from './client';

export function sendChatMessage({ videoId, message, sessionId }) {
  return post('/api/chat', {
    video_id: videoId,
    message,
    session_id: sessionId || undefined,
  });
}
