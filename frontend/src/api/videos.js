import { get, post, del, BASE_URL } from './client';

// ---------------------------------------------------------------------------
// Signed URL 기반 3단계 업로드
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

export async function completeUpload(videoId, storageKey) {
  return post('/api/videos/upload/complete', {
    video_id: videoId,
    storage_key: storageKey,
  });
}

export async function uploadVideo(file) {
  const { video_id, video_name, upload_url, storage_key } = await initUpload(file);
  await uploadToStorage(upload_url, file);
  await completeUpload(video_id, storage_key);
  return { video_id, video_name, status: 'PREPROCESSING' };
}

// ---------------------------------------------------------------------------
// 조회 API
// ---------------------------------------------------------------------------

export function listVideos() {
  return get('/api/videos');
}

export function getVideoStatus(videoId) {
  return get(`/videos/${videoId}/status`);
}

export function getVideoProgress(videoId) {
  return get(`/videos/${videoId}/progress`);
}

export function getVideoSummary(videoId) {
  return get(`/videos/${videoId}/summary`);
}

export function getVideoSummaries(videoId) {
  return get(`/videos/${videoId}/summaries`);
}

export function getVideoStreamUrl(videoId) {
  return `${BASE_URL}/api/videos/${videoId}/stream`;
}

export function getThumbnailUrl(videoId) {
  return `${BASE_URL}/api/videos/${videoId}/thumbnail`;
}

export function restartProcessing(videoId) {
  return post('/process', { video_id: videoId });
}

export function deleteVideo(videoId) {
  return del(`/api/videos/${videoId}`);
}
