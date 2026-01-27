import React, { useState, useEffect, useCallback, useRef } from 'react';
import { StickyNote, RefreshCw, Pencil, Minimize2, Maximize2, Loader2, AlertTriangle, RotateCcw } from 'lucide-react';
import { getVideoSummaries, getVideoStatus, restartProcessing } from '../api/videos';
import usePolling from '../hooks/usePolling';

function formatMs(ms) {
    if (ms == null) return '--:--';
    const totalSec = Math.floor(ms / 1000);
    const m = Math.floor(totalSec / 60);
    const s = totalSec % 60;
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

function SummaryPanel({ isExpanded, onToggleExpand, videoId, onSeekTo, currentTimeMs }) {
    const [items, setItems] = useState([]);
    const [initialLoading, setInitialLoading] = useState(true);
    const [videoStatus, setVideoStatus] = useState(null);
    const [progressInfo, setProgressInfo] = useState({ current: 0, total: 0, percent: 0, stage: '' });
    const prevCountRef = useRef(0);
    const [errorMessage, setErrorMessage] = useState(null);
    const [isRestarting, setIsRestarting] = useState(false);
    const [pollingKey, setPollingKey] = useState(0);

    // 비디오 상태 확인: 처리 중인지 판단
    const isProcessing = videoStatus && !['DONE', 'FAILED'].includes((videoStatus).toUpperCase());

    const fetchStatus = useCallback(() => {
        if (!videoId) return Promise.resolve(null);
        return getVideoStatus(videoId);
    }, [videoId, pollingKey]);

    const fetchSummaries = useCallback(() => {
        if (!videoId) return Promise.resolve(null);
        return getVideoSummaries(videoId);
    }, [videoId, pollingKey]);

    // 상태 폴링: DONE/FAILED까지
    const { data: statusData } = usePolling(fetchStatus, {
        interval: 3000,
        enabled: !!videoId,
        until: useCallback((d) => {
            if (!d) return false;
            const vs = (d.video_status || '').toUpperCase();
            return vs === 'DONE' || vs === 'FAILED';
        }, []),
    });

    // 요약 폴링: 처리 중일 때만 (배치 완료될 때마다 새 데이터 반영)
    const { data: summaryData } = usePolling(fetchSummaries, {
        interval: 4000,
        enabled: !!videoId && isProcessing,
    });

    // 상태 데이터 반영
    useEffect(() => {
        if (!statusData) return;
        setVideoStatus(statusData.video_status);
        setErrorMessage(statusData.error_message || null);
        const job = statusData.processing_job;
        if (job) {
            const current = job.progress_current || 0;
            const total = job.progress_total || 1;
            const jobStatus = (job.status || '').toUpperCase();

            // Stage-aware progress: each stage adds partial progress within the current batch
            const stageWeight = { VLM_RUNNING: 0.3, SUMMARY_RUNNING: 0.6, JUDGE_RUNNING: 0.9 };
            const basePct = total > 0 ? (current / total) * 100 : 0;
            const batchWidth = total > 0 ? 100 / total : 0;
            const stageBonus = (stageWeight[jobStatus] || 0) * batchWidth;
            const percent = jobStatus === 'DONE' ? 100
                : Math.min(Math.round(basePct + stageBonus), 99);

            setProgressInfo({
                current,
                total,
                percent,
                stage: jobStatus,
            });
        }
    }, [statusData]);

    // 초기 요약 로드
    useEffect(() => {
        if (!videoId) return;
        setInitialLoading(true);
        getVideoSummaries(videoId)
            .then((data) => {
                const newItems = data.items || [];
                setItems(newItems);
                prevCountRef.current = newItems.length;
            })
            .catch(() => { })
            .finally(() => setInitialLoading(false));
    }, [videoId]);

    // 폴링 결과 반영 (새 배치가 추가되면 자동 업데이트)
    useEffect(() => {
        if (!summaryData?.items) return;
        const newItems = summaryData.items;
        if (newItems.length !== prevCountRef.current) {
            setItems(newItems);
            prevCountRef.current = newItems.length;
        }
    }, [summaryData]);

    // 처리 완료 시 최종 요약 한 번 더 로드
    useEffect(() => {
        if (!videoId) return;
        if (videoStatus && (videoStatus.toUpperCase() === 'DONE')) {
            getVideoSummaries(videoId)
                .then((data) => setItems(data.items || []))
                .catch(() => { });
        }
    }, [videoStatus, videoId]);

    const stageLabel = (() => {
        const s = (progressInfo.stage || videoStatus || '').toUpperCase();
        const batchPrefix = progressInfo.total > 1
            ? `배치 ${Math.min(progressInfo.current + 1, progressInfo.total)}/${progressInfo.total} · `
            : '';
        if (s === 'VLM_RUNNING') return `${batchPrefix}AI 분석 진행 중...`;
        if (s === 'SUMMARY_RUNNING') return `${batchPrefix}요약 생성 중...`;
        if (s === 'JUDGE_RUNNING') return `${batchPrefix}품질 평가 중...`;
        if (s === 'PREPROCESSING' || s === 'PREPROCESS_DONE') return '전처리 완료, 분석 대기 중...';
        return '영상 분석 중...';
    })();

    const ProcessingStatusBar = () => {
        if (!isProcessing) return null;
        return (
            <div className="bg-surface rounded-lg p-4 border border-[var(--border-color)] flex flex-col gap-3 shadow-sm mb-4">
                <div className="flex gap-6 justify-between items-center">
                    <div className="flex items-center gap-2">
                        <RefreshCw className="w-5 h-5 text-primary animate-spin" />
                        <p className="text-[var(--text-primary)] text-sm font-medium">{stageLabel}</p>
                    </div>
                    <div className="flex items-center gap-2">
                        {progressInfo.total > 0 && (
                            <span className="text-gray-400 text-xs">
                                {progressInfo.current}/{progressInfo.total} batches
                            </span>
                        )}
                        <p className="text-gray-400 text-xs font-mono">{progressInfo.percent}%</p>
                    </div>
                </div>
                <div className="h-1.5 w-full bg-surface-highlight rounded-full overflow-hidden">
                    <div
                        className="h-full bg-primary rounded-full relative overflow-hidden transition-all duration-700"
                        style={{ width: `${progressInfo.percent}%` }}
                    >
                        <div className="absolute inset-0 bg-white/20 w-full animate-[shimmer_2s_infinite] translate-x-[-100%]"></div>
                    </div>
                </div>
            </div>
        );
    };

    const CompletedBanner = () => {
        if (!videoStatus || videoStatus.toUpperCase() !== 'DONE') return null;
        return (
            <div className="bg-green-500/10 rounded-lg p-3 border border-green-500/20 flex items-center gap-2 mb-4">
                <span className="text-green-400 text-sm font-medium">분석 완료 - {items.length}개 세그먼트</span>
            </div>
        );
    };

    const FailedBanner = () => {
        if (!videoStatus || videoStatus.toUpperCase() !== 'FAILED') return null;

        const handleRestart = async () => {
            setIsRestarting(true);
            try {
                await restartProcessing(videoId);
                setVideoStatus(null);
                setErrorMessage(null);
                setProgressInfo({ current: 0, total: 0, percent: 0, stage: '' });
                setPollingKey(k => k + 1);
            } catch (err) {
                setErrorMessage(`재시작 실패: ${err.message || '알 수 없는 오류'}`);
            } finally {
                setIsRestarting(false);
            }
        };

        return (
            <div className="bg-red-500/10 rounded-lg p-4 border border-red-500/20 flex flex-col gap-3 mb-4">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <AlertTriangle className="w-5 h-5 text-red-400" />
                        <span className="text-red-400 text-sm font-medium">분석 실패</span>
                    </div>
                    <button
                        onClick={handleRestart}
                        disabled={isRestarting}
                        className="flex items-center gap-1.5 px-3 py-1.5 bg-red-500/20 hover:bg-red-500/30 text-red-400 text-xs font-medium rounded-lg border border-red-500/30 transition-colors disabled:opacity-50"
                    >
                        <RotateCcw className={`w-3.5 h-3.5 ${isRestarting ? 'animate-spin' : ''}`} />
                        {isRestarting ? '재시작 중...' : '요약 재시작'}
                    </button>
                </div>
                {errorMessage && (
                    <p className="text-red-400/70 text-xs">{errorMessage}</p>
                )}
            </div>
        );
    };

    const renderBulletText = (b) => {
        if (typeof b === 'string') return b;
        const prefix = b.bullet_id ? `(${b.bullet_id}) ` : '';
        return prefix + (b.claim || b.text || JSON.stringify(b));
    };

    const renderSummaryItem = (item, index) => {
        const summaryContent = item.summary || {};
        const segIdx = item.segment_index ?? index + 1;
        const timeRange = `${formatMs(item.start_ms)}–${formatMs(item.end_ms)}`;
        const title = `Segment ${segIdx} (${timeRange})`;

        const bullets = summaryContent.bullets || [];
        const definitions = summaryContent.definitions || [];
        const explanations = summaryContent.explanations || [];
        const openQuestions = summaryContent.open_questions || [];
        const isNew = index >= prevCountRef.current - 1 && isProcessing;
        const hasSections = bullets.length > 0 || definitions.length > 0 || explanations.length > 0;

        const isActive = currentTimeMs != null &&
            item.start_ms != null && item.end_ms != null &&
            currentTimeMs >= item.start_ms && currentTimeMs < item.end_ms;

        const handleCardClick = () => {
            if (!isExpanded && onSeekTo && item.start_ms != null) {
                onSeekTo(item.start_ms);
            }
        };

        return (
            <div
                key={item.summary_id || index}
                className={`group flex flex-col md:flex-row gap-2 md:gap-8 border-l-2 ${
                    isActive
                        ? 'border-l-primary bg-primary/10'
                        : 'border-l-transparent'
                } ${isExpanded ? 'p-5 rounded-xl hover:bg-surface/30' : 'p-3 rounded-lg hover:bg-[var(--bg-hover)] cursor-pointer'
                    } ${isNew ? 'animate-fade-in' : ''} transition-colors`}
                onClick={handleCardClick}
            >
                <div className="md:w-24 shrink-0 flex md:justify-end">
                    <button
                        onClick={(e) => {
                            e.stopPropagation();
                            if (onSeekTo && item.start_ms != null) onSeekTo(item.start_ms);
                        }}
                        className={`font-mono text-sm bg-surface/50 px-2 py-0.5 rounded border transition-colors h-fit cursor-pointer ${
                            isActive
                                ? 'text-primary border-primary/50 bg-primary/10'
                                : 'text-gray-400 border-[var(--border-color)] group-hover:border-primary/50 hover:text-primary hover:bg-primary/10'
                        }`}
                        title={`${formatMs(item.start_ms)} 으로 이동`}
                    >
                        {formatMs(item.start_ms)}
                    </button>
                </div>
                <div className="flex-1">
                    <h4 className={`text-[var(--text-primary)] font-semibold ${isExpanded ? 'mb-3 text-lg' : 'mb-1'}`}>{title}</h4>

                    {!hasSections && (
                        <p className="text-[var(--text-secondary)] text-sm italic">요약 데이터 준비 중...</p>
                    )}

                    {/* 요약 (Bullets) */}
                    {bullets.length > 0 && (
                        <div className={isExpanded ? 'mb-4' : 'mb-1'}>
                            {isExpanded && (
                                <h5 className="text-primary text-xs font-bold uppercase tracking-wider mb-1.5">요약</h5>
                            )}
                            <ul className={`list-disc list-outside ml-4 text-[var(--text-secondary)] ${isExpanded ? 'text-base space-y-1.5' : 'text-sm space-y-0.5'} leading-relaxed`}>
                                {bullets.map((b, i) => (
                                    <li key={i}>{renderBulletText(b)}</li>
                                ))}
                            </ul>
                        </div>
                    )}

                    {/* 정의 (Definitions) — expanded only */}
                    {isExpanded && definitions.length > 0 && (
                        <div className="mb-4">
                            <h5 className="text-primary text-xs font-bold uppercase tracking-wider mb-1.5">정의</h5>
                            <ul className="list-disc list-outside ml-4 text-[var(--text-secondary)] text-base space-y-1.5 leading-relaxed">
                                {definitions.map((d, i) => (
                                    <li key={i}>
                                        <span className="font-semibold text-[var(--text-primary)]">{d.term}</span>
                                        {': '}
                                        {d.definition}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}

                    {/* 해설 (Explanations) — expanded only */}
                    {isExpanded && explanations.length > 0 && (
                        <div className="mb-4">
                            <h5 className="text-primary text-xs font-bold uppercase tracking-wider mb-1.5">해설</h5>
                            <ul className="list-disc list-outside ml-4 text-[var(--text-secondary)] text-base space-y-1.5 leading-relaxed">
                                {explanations.map((e, i) => (
                                    <li key={i}>{e.point || e.text || JSON.stringify(e)}</li>
                                ))}
                            </ul>
                        </div>
                    )}

                    {/* 미해결 질문 (Open Questions) — expanded only */}
                    {isExpanded && openQuestions.length > 0 && (
                        <div className="mb-2">
                            <h5 className="text-primary text-xs font-bold uppercase tracking-wider mb-1.5">미해결 질문</h5>
                            <ul className="list-disc list-outside ml-4 text-[var(--text-secondary)] text-base space-y-1.5 leading-relaxed italic">
                                {openQuestions.map((q, i) => (
                                    <li key={i}>{q.question || q.text || JSON.stringify(q)}</li>
                                ))}
                            </ul>
                        </div>
                    )}
                </div>
            </div>
        );
    };

    return (
        <div className={`flex flex-col ${isExpanded ? 'items-center p-6 lg:p-10 overflow-y-auto custom-scrollbar absolute inset-0' : 'gap-4 pb-12 max-w-4xl'}`}>
            <div className={`w-full ${isExpanded ? 'max-w-4xl flex flex-col gap-6 pb-24' : ''}`}>
                {/* Header */}
                {isExpanded ? (
                    <div className="flex items-center justify-between border-b border-[var(--border-color)] pb-4 sticky top-0 bg-[var(--bg-primary)]/95 backdrop-blur z-40 py-2">
                        <div className="flex items-center gap-3">
                            <StickyNote className="w-7 h-7 text-primary" />
                            <div className="flex flex-col">
                                <h3 className="text-[var(--text-primary)] text-2xl font-bold tracking-tight">Real-time Summary</h3>
                                <p className="text-gray-400 text-xs">
                                    {items.length > 0 ? `${items.length} segments` : isProcessing ? 'Generating...' : 'No data'}
                                </p>
                            </div>
                        </div>
                        <div className="flex items-center gap-3">
                            <button className="text-gray-400 hover:text-[var(--text-primary)] text-xs flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-[var(--border-color)] hover:bg-surface-highlight transition-colors bg-surface">
                                <Pencil className="w-4 h-4" />
                                Edit Notes
                            </button>
                            <button
                                onClick={onToggleExpand}
                                className="text-white bg-primary hover:bg-[var(--accent-coral-dark)] text-xs flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-transparent transition-colors shadow-lg shadow-primary/20"
                            >
                                <Minimize2 className="w-4 h-4" />
                                Collapse View
                            </button>
                        </div>
                    </div>
                ) : (
                    <>
                        <ProcessingStatusBar />
                        <CompletedBanner />
                        <FailedBanner />
                        <div className="flex items-center justify-between border-b border-[var(--border-color)] pb-2">
                            <div className="flex items-center gap-2">
                                <StickyNote className="w-6 h-6 text-primary" />
                                <h3 className="text-[var(--text-primary)] text-xl font-bold tracking-tight">Real-time Summary</h3>
                                {items.length > 0 && (
                                    <span className="text-gray-400 text-xs ml-1">({items.length})</span>
                                )}
                            </div>
                            <button
                                onClick={onToggleExpand}
                                className="text-gray-400 hover:text-[var(--text-primary)] text-xs flex items-center gap-1 transition-colors"
                            >
                                <Maximize2 className="w-4 h-4" />
                                Expand
                            </button>
                        </div>
                    </>
                )}

                {isExpanded && (
                    <>
                        <ProcessingStatusBar />
                        <CompletedBanner />
                        <FailedBanner />
                    </>
                )}

                {/* Timeline */}
                <div className={`flex flex-col gap-1 relative pl-2 ${isExpanded ? 'pt-4' : ''}`}>
                    <div className="absolute left-[8.5rem] top-4 bottom-4 w-px bg-border-color hidden md:block"></div>

                    {initialLoading && items.length === 0 && (
                        <div className="flex justify-center py-8">
                            <Loader2 className="w-6 h-6 animate-spin text-primary" />
                        </div>
                    )}

                    {!initialLoading && items.length === 0 && (
                        <div className="py-8 text-center text-gray-400 text-sm">
                            {isProcessing ? '요약이 생성되면 여기에 실시간으로 표시됩니다...' : '요약 데이터가 없습니다.'}
                        </div>
                    )}

                    {items.map((item, index) => renderSummaryItem(item, index))}

                    {/* Skeleton: 다음 배치 대기 */}
                    {isProcessing && (
                        <div className={`group flex flex-col md:flex-row gap-2 md:gap-8 ${isExpanded ? 'p-4 rounded-xl opacity-40' : 'p-3 rounded-lg opacity-40'} transition-colors`}>
                            <div className="md:w-24 shrink-0 flex md:justify-end">
                                <span className="font-mono text-sm text-gray-400 bg-surface/50 px-2 py-0.5 rounded border border-[var(--border-color)] h-fit">--:--</span>
                            </div>
                            <div className="flex-1">
                                <div className="h-4 w-3/4 bg-[var(--border-color)] rounded animate-pulse mt-1"></div>
                                <div className="h-4 w-1/2 bg-[var(--border-color)] rounded animate-pulse mt-2"></div>
                                {isExpanded && <div className="h-4 w-5/6 bg-[var(--border-color)] rounded animate-pulse mt-2"></div>}
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

export default SummaryPanel;
