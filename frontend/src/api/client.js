import { supabase } from '../lib/supabase';

export const BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
    this.name = 'ApiError';
  }
}

async function getAuthHeaders() {
  if (!supabase) return {};
  try {
    const { data: { session } } = await supabase.auth.getSession();
    if (session?.access_token) {
      return { Authorization: `Bearer ${session.access_token}` };
    }
  } catch {
    // ignore auth header errors
  }
  return {};
}

async function request(url, options = {}) {
  const { headers: optHeaders, ...restOptions } = options;
  const authHeaders = await getAuthHeaders();
  const res = await fetch(BASE_URL + url, {
    ...restOptions,
    headers: { 'Content-Type': 'application/json', ...optHeaders, ...authHeaders },
  });

  // 401이면 세션 갱신 후 1회 재시도
  if (res.status === 401 && supabase) {
    const { data, error } = await supabase.auth.refreshSession();
    if (!error && data.session?.access_token) {
      const retryHeaders = { Authorization: `Bearer ${data.session.access_token}` };
      const retryRes = await fetch(BASE_URL + url, {
        ...restOptions,
        headers: { 'Content-Type': 'application/json', ...optHeaders, ...retryHeaders },
      });
      if (retryRes.status === 204) return null;
      if (retryRes.ok) return retryRes.json();
      // 재시도도 실패하면 아래 에러 처리로
      let message = retryRes.statusText;
      try {
        const body = await retryRes.json();
        message = body.detail || body.message || message;
      } catch { /* ignore */ }
      throw new ApiError(retryRes.status, message);
    }
  }

  if (!res.ok) {
    let message = res.statusText;
    try {
      const body = await res.json();
      message = body.detail || body.message || message;
    } catch {
      // ignore parse errors
    }
    throw new ApiError(res.status, message);
  }

  if (res.status === 204) return null;
  return res.json();
}

export function get(url) {
  return request(url, { method: 'GET' });
}

export function post(url, body) {
  return request(url, {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

export function del(url) {
  return request(url, { method: 'DELETE' });
}

export async function postForm(url, formData) {
  const authHeaders = await getAuthHeaders();
  let res = await fetch(BASE_URL + url, { method: 'POST', body: formData, headers: { ...authHeaders } });

  // 401이면 세션 갱신 후 1회 재시도
  if (res.status === 401 && supabase) {
    const { data, error } = await supabase.auth.refreshSession();
    if (!error && data.session?.access_token) {
      const retryHeaders = { Authorization: `Bearer ${data.session.access_token}` };
      res = await fetch(BASE_URL + url, { method: 'POST', body: formData, headers: { ...authHeaders, ...retryHeaders } });
    }
  }

  if (!res.ok) {
    let message = res.statusText;
    try {
      const body = await res.json();
      message = body.detail || body.message || message;
    } catch { /* ignore */ }
    throw new ApiError(res.status, message);
  }
  return res.json();
}
