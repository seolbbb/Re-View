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
  const authHeaders = await getAuthHeaders();
  const res = await fetch(BASE_URL + url, {
    headers: { 'Content-Type': 'application/json', ...authHeaders, ...options.headers },
    ...options,
  });

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

export function postForm(url, formData) {
  return getAuthHeaders().then((authHeaders) =>
    fetch(BASE_URL + url, { method: 'POST', body: formData, headers: { ...authHeaders } })
  ).then(async (res) => {
    if (!res.ok) {
      let message = res.statusText;
      try {
        const body = await res.json();
        message = body.detail || body.message || message;
      } catch {
        // ignore
      }
      throw new ApiError(res.status, message);
    }
    return res.json();
  });
}
