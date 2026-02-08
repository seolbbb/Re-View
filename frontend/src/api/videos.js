import { get, post, del, BASE_URL } from './client';

// ---------------------------------------------------------------------------
// Signed URL 기반 3단계 업로드 (FastAPI 경유)
// ---------------------------------------------------------------------------

export async function initUpload(file) {
  return post('/api/videos/upload/init', {
    filename: file.name,
    content_type: file.type || 'video/mp4',
  });
}

export async function uploadToStorage(uploadUrl, file) {
  const res = await fetch(uploadUrl, {
    method: 'PUT',
    headers: { 'Content-Type': file.type || 'video/mp4' },
    body: file,
  });
  if (!res.ok) {
    throw new Error('Storage upload failed');
  }
}

export async function completeUpload(videoId, storageKey, pipelineMode = 'async') {
  return post('/api/videos/upload/complete', {
    video_id: videoId,
    storage_key: storageKey,
    pipeline_mode: pipelineMode,
  });
}

export async function uploadVideo(file, pipelineMode = 'async') {
  const { video_id, video_name, upload_url, storage_key } = await initUpload(file);
  await uploadToStorage(upload_url, file);
  await completeUpload(video_id, storage_key, pipelineMode);
  invalidateVideosCache();
  return { video_id, video_name, status: 'PREPROCESSING' };
}

// ---------------------------------------------------------------------------
// 조회 API (FastAPI 경유 — 썸네일 URL 포함, sessionStorage 캐시)
// ---------------------------------------------------------------------------

const VIDEOS_CACHE_KEY = 'videos_cache';
const VIDEOS_CACHE_TTL = 60_000; // 1분

export async function listVideos() {
  // sessionStorage 캐시 확인
  try {
    const cached = sessionStorage.getItem(VIDEOS_CACHE_KEY);
    if (cached) {
      const { data, ts } = JSON.parse(cached);
      if (Date.now() - ts < VIDEOS_CACHE_TTL) {
        return data;
      }
    }
  } catch { /* ignore */ }

  const data = await get('/api/videos');

  // 캐시 저장
  try {
    sessionStorage.setItem(VIDEOS_CACHE_KEY, JSON.stringify({ data, ts: Date.now() }));
  } catch { /* ignore */ }

  return data;
}

export function invalidateVideosCache() {
  try { sessionStorage.removeItem(VIDEOS_CACHE_KEY); } catch { /* ignore */ }
}

export async function getVideoSummary(videoId) {
  return get(`/videos/${videoId}/summary`);
}

export async function getVideoSummaries(videoId) {
  // 백엔드 API 사용 (segments 테이블과 적절히 JOIN 처리됨)
  return get(`/videos/${videoId}/summaries`);
}

// ---------------------------------------------------------------------------
// 상태 조회 및 스트리밍 (FastAPI 유지)
// ---------------------------------------------------------------------------

export function getVideoStatus(videoId) {
  return get(`/videos/${videoId}/status`);
}

export function getVideoProgress(videoId) {
  return get(`/videos/${videoId}/progress`);
}

export function restartProcessing(videoId) {
  // Ensure reruns publish results back into DB so the UI can read summaries.
  return post('/process', { video_id: videoId, sync_to_db: true });
}

// ---------------------------------------------------------------------------
// Delete
// ---------------------------------------------------------------------------

export async function deleteVideo(videoId) {
  const result = await del(`/api/videos/${videoId}`);
  invalidateVideosCache();
  return result;
}

// ---------------------------------------------------------------------------
// Media Ticket (for <video>/<img> src without Authorization header)
// ---------------------------------------------------------------------------

export function getMediaTicket() {
  return post('/api/media/ticket', {});
}

// ---------------------------------------------------------------------------
// Media URLs
// ---------------------------------------------------------------------------

export function getVideoStreamUrl(videoId, ticket) {
  if (!videoId) return null;
  const q = ticket ? `?ticket=${encodeURIComponent(ticket)}` : '';
  return `${BASE_URL}/api/videos/${videoId}/stream${q}`;
}

export function getThumbnailUrl(videoId, ticket) {
  if (!videoId) return null;
  const q = ticket ? `?ticket=${encodeURIComponent(ticket)}` : '';
  return `${BASE_URL}/api/videos/${videoId}/thumbnail${q}`;
}
