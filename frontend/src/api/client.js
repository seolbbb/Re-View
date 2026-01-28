export const BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
    this.name = 'ApiError';
  }
}

async function request(url, options = {}) {
  const res = await fetch(BASE_URL + url, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
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

export function postForm(url, formData) {
  return fetch(BASE_URL + url, { method: 'POST', body: formData }).then(async (res) => {
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
