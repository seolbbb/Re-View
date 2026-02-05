import { get, post, BASE_URL } from './client';
import { supabase } from '../lib/supabase';

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
// 조회 API (Supabase 직접 쿼리 - RLS 적용)
// ---------------------------------------------------------------------------

export async function listVideos() {
  if (!supabase) {
    // Supabase 미설정 시 기존 FastAPI 사용
    return get('/api/videos');
  }

  // 현재 로그인한 사용자 세션 확인
  const { data: { session } } = await supabase.auth.getSession();

  if (!session?.user) {
    return { videos: [] };
  }

  const { data, error } = await supabase
    .from('videos')
    .select('*')
    .eq('user_id', session.user.id)  // 로그인한 유저의 영상만 필터링
    .order('created_at', { ascending: false });

  if (error) throw error;
  return { videos: data };
}

export async function getVideoSummary(videoId) {
  if (!supabase) {
    return get(`/videos/${videoId}/summary`);
  }

  const { data, error } = await supabase
    .from('summary_results')
    .select('*')
    .eq('video_id', videoId)
    .single();

  if (error) throw error;
  return data;
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

export function getVideoStreamUrl(videoId) {
  return `${BASE_URL}/api/videos/${videoId}/stream`;
}

export function getThumbnailUrl(videoId) {
  return `${BASE_URL}/api/videos/${videoId}/thumbnail`;
}

export function restartProcessing(videoId) {
  return post('/process', { video_id: videoId });
}
