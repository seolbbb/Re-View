import { useEffect, useRef, useState } from 'react';

/**
 * 주기적으로 fetchFn을 호출하여 데이터를 폴링합니다.
 *
 * @param {Function} fetchFn - 호출할 async 함수
 * @param {Object} options
 * @param {number}  options.interval  - 폴링 간격 (ms, 기본 3000)
 * @param {boolean} options.enabled   - 폴링 활성화 여부 (기본 true)
 * @param {Function} options.until    - (data) => boolean, true 반환 시 폴링 중지
 */
export default function usePolling(fetchFn, { interval = 3000, enabled = true, until } = {}) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const stopped = useRef(false);

  useEffect(() => {
    stopped.current = false;
    if (!enabled) return;

    let timer;

    async function poll() {
      if (stopped.current) return;
      setLoading(true);
      try {
        const result = await fetchFn();
        if (stopped.current) return;
        setData(result);
        setError(null);
        if (until && until(result)) {
          stopped.current = true;
          return;
        }
      } catch (err) {
        if (!stopped.current) setError(err);
      } finally {
        if (!stopped.current) setLoading(false);
      }
      if (!stopped.current) {
        timer = setTimeout(poll, interval);
      }
    }

    poll();

    return () => {
      stopped.current = true;
      clearTimeout(timer);
    };
  }, [fetchFn, interval, enabled, until]);

  return { data, error, loading };
}
