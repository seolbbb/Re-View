import { useEffect, useMemo, useState } from 'react';
import { useLocation, useParams, Link } from 'react-router-dom';
import { getVideoStatus, getVideoSummaries } from '../api/videos';
import MarkdownRenderer from '../components/MarkdownRenderer';
import './SummaryPrintPage.css';

function formatMs(ms) {
  if (ms == null) return '--:--';
  const totalSec = Math.floor(ms / 1000);
  const m = Math.floor(totalSec / 60);
  const s = totalSec % 60;
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

function getBulletText(b) {
  if (typeof b === 'string') return b;
  const prefix = b?.bullet_id ? `(${b.bullet_id}) ` : '';
  return prefix + (b?.claim || b?.text || JSON.stringify(b));
}

function SummaryPrintPage() {
  const { id: videoId } = useParams();
  const location = useLocation();
  const [videoName, setVideoName] = useState('');
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const autoprint = useMemo(() => {
    const qs = new URLSearchParams(location.search);
    return qs.get('autoprint') === '1';
  }, [location.search]);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      if (!videoId) return;
      setLoading(true);
      setError(null);
      try {
        const [status, summaries] = await Promise.all([
          getVideoStatus(videoId),
          getVideoSummaries(videoId),
        ]);
        if (cancelled) return;
        setVideoName(status?.video_name || videoId);
        setItems(summaries?.items || []);
      } catch (e) {
        if (cancelled) return;
        setError(e);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [videoId]);

  useEffect(() => {
    if (!autoprint) return;
    if (loading) return;
    if (error) return;
    if (!items || items.length === 0) return;

    let cancelled = false;
    async function triggerPrint() {
      try {
        // Wait for fonts (KaTeX + Korean glyphs) to be ready before printing.
        if (document?.fonts?.ready) await document.fonts.ready;
      } catch {
        // ignore
      }

      // Allow a couple of frames so KaTeX/markdown layout settles.
      await new Promise((r) => requestAnimationFrame(() => r()));
      await new Promise((r) => requestAnimationFrame(() => r()));
      if (cancelled) return;
      window.print();
    }
    triggerPrint();
    return () => {
      cancelled = true;
    };
  }, [autoprint, loading, error, items]);

  const generatedAt = useMemo(() => new Date().toLocaleString(), []);

  return (
    <div className="summary-print-root">
      <div className="summary-print-toolbar no-print">
        <div className="summary-print-toolbar-inner">
          <div className="summary-print-muted">
            <strong style={{ color: '#111827' }}>Print View</strong>
          </div>
          <div style={{ display: 'flex', gap: 10 }}>
            <button className="summary-print-btn" onClick={() => window.print()}>
              Print / Save as PDF
            </button>
            <Link className="summary-print-btn" to={`/analysis/${videoId}`}>
              Back
            </Link>
          </div>
        </div>
      </div>

      <div className="summary-print-container">
        <h1 style={{ fontSize: 26, fontWeight: 900, letterSpacing: '-0.02em', lineHeight: 1.2 }}>
          {videoName || 'Summary'}
        </h1>
        <div style={{ marginTop: 10, display: 'flex', flexWrap: 'wrap', gap: 14 }} className="summary-print-muted">
          <div><strong style={{ color: '#374151' }}>Generated</strong>: {generatedAt}</div>
          <div><strong style={{ color: '#374151' }}>Segments</strong>: {items.length}</div>
        </div>
        <div style={{ height: 1, background: '#e5e7eb', marginTop: 14 }} />

        {loading && (
          <div style={{ marginTop: 18 }} className="summary-print-muted">Loading…</div>
        )}

        {!loading && error && (
          <div style={{ marginTop: 18, color: '#b91c1c', fontSize: 12 }}>
            Failed to load summary data.
          </div>
        )}

        {!loading && !error && items.length === 0 && (
          <div style={{ marginTop: 18 }} className="summary-print-muted">요약 데이터가 없습니다.</div>
        )}

        {!loading && !error && items.map((item, index) => {
          const segIdx = item.segment_index ?? index + 1;
          const timeRange = `${formatMs(item.start_ms)}–${formatMs(item.end_ms)}`;
          const summary = item.summary || {};
          const bullets = summary.bullets || [];
          const definitions = summary.definitions || [];
          const explanations = summary.explanations || [];
          const openQuestions = summary.open_questions || [];

          return (
            <div key={item.summary_id || index} className="summary-segment-card">
              <div style={{ fontSize: 15, fontWeight: 900, letterSpacing: '-0.01em' }}>
                Segment {segIdx} ({timeRange})
              </div>

              {bullets.length > 0 && (
                <>
                  <div className="summary-section-title">요약</div>
                  <ul className="summary-list">
                    {bullets.map((b, i) => (
                      <li key={i}>
                        <MarkdownRenderer>{getBulletText(b)}</MarkdownRenderer>
                      </li>
                    ))}
                  </ul>
                </>
              )}

              {definitions.length > 0 && (
                <>
                  <div className="summary-section-title">정의</div>
                  <ul className="summary-list">
                    {definitions.map((d, i) => (
                      <li key={i} className="markdown-inline">
                        {typeof d === 'string' ? (
                          <MarkdownRenderer>{d}</MarkdownRenderer>
                        ) : (
                          <>
                            <MarkdownRenderer inline className="font-semibold">{d.term}</MarkdownRenderer>
                            {': '}
                            <MarkdownRenderer inline>{d.definition}</MarkdownRenderer>
                          </>
                        )}
                      </li>
                    ))}
                  </ul>
                </>
              )}

              {explanations.length > 0 && (
                <>
                  <div className="summary-section-title">해설</div>
                  <ul className="summary-list">
                    {explanations.map((e, i) => (
                      <li key={i}>
                        <MarkdownRenderer>{typeof e === 'string' ? e : (e.point || e.text || JSON.stringify(e))}</MarkdownRenderer>
                      </li>
                    ))}
                  </ul>
                </>
              )}

              {openQuestions.length > 0 && (
                <>
                  <div className="summary-section-title">열린 질문</div>
                  <ul className="summary-list" style={{ fontStyle: 'italic' }}>
                    {openQuestions.map((q, i) => (
                      <li key={i}>
                        <MarkdownRenderer>{typeof q === 'string' ? q : (q.question || q.text || JSON.stringify(q))}</MarkdownRenderer>
                      </li>
                    ))}
                  </ul>
                </>
              )}

              {bullets.length === 0 && definitions.length === 0 && explanations.length === 0 && openQuestions.length === 0 && (
                <div style={{ marginTop: 10 }} className="summary-print-muted">요약 데이터 준비 중…</div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default SummaryPrintPage;
