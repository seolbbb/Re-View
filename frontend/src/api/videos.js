import { get, postForm } from './client';

export function uploadVideo(file) {
  const form = new FormData();
  form.append('file', file);
  return postForm('/api/videos/upload', form);
}

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
  return `/api/videos/${videoId}/stream`;
}

export function getThumbnailUrl(videoId) {
  return `/api/videos/${videoId}/thumbnail`;
}
