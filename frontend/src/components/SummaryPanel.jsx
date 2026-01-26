import React, { useState, useEffect, useCallback } from 'react';
import { StickyNote, RefreshCw, Pencil, Minimize2, Maximize2, Loader2 } from 'lucide-react';
import { getVideoSummaries, getVideoStatus } from '../api/videos';
import usePolling from '../hooks/usePolling';

function formatMs(ms) {
    if (ms == null) return '--:--';
    const totalSec = Math.floor(ms / 1000);
    const m = Math.floor(totalSec / 60);
    const s = totalSec % 60;
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

function SummaryPanel({ isExpanded, onToggleExpand, videoId }) {
    const [items, setItems] = useState([]);
    const [loading, setLoading] = useState(true);
    const [processingStatus, setProcessingStatus] = useState(null);
    const [progressPercent, setProgressPercent] = useState(0);

    const fetchSummaries = useCallback(() => {
        if (!videoId) return Promise.resolve(null);
        return getVideoSummaries(videoId);
    }, [videoId]);

    const fetchStatus = useCallback(() => {
        if (!videoId) return Promise.resolve(null);
        return getVideoStatus(videoId);
    }, [videoId]);

    // Check if processing is still running
    const isProcessing = processingStatus && !['DONE', 'FAILED'].includes((processingStatus.video_status || '').toUpperCase());

    const { data: statusData } = usePolling(fetchStatus, {
        interval: 3000,
        enabled: !!videoId,
        until: useCallback((d) => {
            if (!d) return false;
            const vs = (d.video_status || '').toUpperCase();
            return vs === 'DONE' || vs === 'FAILED';
        }, []),
    });

    const { data: summaryData } = usePolling(fetchSummaries, {
        interval: 5000,
        enabled: !!videoId && isProcessing,
        until: useCallback(() => false, []),
    });

    // Initial fetch
    useEffect(() => {
        if (!videoId) return;
        setLoading(true);
        getVideoSummaries(videoId)
            .then((data) => {
                setItems(data.items || []);
            })
            .catch(() => {})
            .finally(() => setLoading(false));
    }, [videoId]);

    // Update from polling
    useEffect(() => {
        if (summaryData?.items) {
            setItems(summaryData.items);
        }
    }, [summaryData]);

    useEffect(() => {
        if (statusData) {
            setProcessingStatus(statusData);
            const job = statusData.processing_job;
            if (job && job.progress_total > 0) {
                setProgressPercent(Math.round((job.progress_current / job.progress_total) * 100));
            }
            if (statusData.video_status === 'DONE') {
                setProgressPercent(100);
            }
        }
    }, [statusData]);

    const ProcessingStatusBar = () => {
        if (!isProcessing) return null;
        const jobStatus = processingStatus?.processing_job?.status || processingStatus?.video_status || '';
        return (
            <div className="bg-surface rounded-lg p-4 border border-[var(--border-color)] flex flex-col gap-3 shadow-sm mb-4">
                <div className="flex gap-6 justify-between items-center">
                    <div className="flex items-center gap-2">
                        <RefreshCw className="w-5 h-5 text-primary animate-spin" />
                        <p className="text-[var(--text-primary)] text-sm font-medium">
                            {jobStatus === 'VLM_RUNNING' ? 'AI 분석 진행 중...' :
                             jobStatus === 'SUMMARY_RUNNING' ? '요약 생성 중...' :
                             'Analyzing video content & generating notes...'}
                        </p>
                    </div>
                    <p className="text-gray-400 text-xs font-mono">{progressPercent}%</p>
                </div>
                <div className="h-1.5 w-full bg-surface-highlight rounded-full overflow-hidden">
                    <div className="h-full bg-primary rounded-full relative overflow-hidden transition-all duration-500" style={{ width: `${progressPercent}%` }}>
                        <div className="absolute inset-0 bg-white/20 w-full animate-[shimmer_2s_infinite] translate-x-[-100%]"></div>
                    </div>
                </div>
            </div>
        );
    };

    const renderSummaryItem = (item, index) => {
        const summaryContent = item.summary || {};
        const title = summaryContent.title || summaryContent.heading || `Segment ${item.segment_index ?? index + 1}`;
        const bullets = summaryContent.bullets || summaryContent.points || [];
        const body = summaryContent.body || summaryContent.text || '';

        return (
            <div
                key={item.summary_id || index}
                className={`group flex flex-col md:flex-row gap-2 md:gap-8 ${isExpanded ? 'p-4 rounded-xl hover:bg-surface/30' : 'p-3 rounded-lg hover:bg-[var(--bg-hover)]'} transition-colors cursor-pointer`}
            >
                <div className="md:w-24 shrink-0 flex md:justify-end">
                    <span className="font-mono text-sm text-gray-400 bg-surface/50 px-2 py-0.5 rounded border border-[var(--border-color)] group-hover:border-primary/50 transition-colors h-fit">
                        {formatMs(item.start_ms)}
                    </span>
                </div>
                <div className="flex-1">
                    <h4 className={`text-[var(--text-primary)] font-medium ${isExpanded ? 'mb-2 text-lg' : 'mb-1'}`}>{title}</h4>
                    {bullets.length > 0 ? (
                        <ul className={`list-disc list-outside ml-4 text-[var(--text-secondary)] ${isExpanded ? 'text-base space-y-2' : 'text-sm space-y-1'} leading-relaxed`}>
                            {bullets.map((b, i) => (
                                <li key={i}>{typeof b === 'string' ? b : b.text || JSON.stringify(b)}</li>
                            ))}
                        </ul>
                    ) : body ? (
                        <p className={`text-[var(--text-secondary)] ${isExpanded ? 'text-base' : 'text-sm'} leading-relaxed`}>{body}</p>
                    ) : (
                        <p className="text-[var(--text-secondary)] text-sm italic">Summary data available (raw JSON)</p>
                    )}
                </div>
            </div>
        );
    };

    return (
        <div className={`flex flex-col ${isExpanded ? 'items-center p-6 lg:p-10 overflow-y-auto custom-scrollbar absolute inset-0' : 'gap-4 pb-12 max-w-4xl'}`}>
            <div className={`w-full ${isExpanded ? 'max-w-4xl flex flex-col gap-6 pb-24' : ''}`}>
                {/* Header Section */}
                {isExpanded ? (
                    <div className="flex items-center justify-between border-b border-[var(--border-color)] pb-4 sticky top-0 bg-[var(--bg-primary)]/95 backdrop-blur z-40 py-2">
                        <div className="flex items-center gap-3">
                            <StickyNote className="w-7 h-7 text-primary" />
                            <div className="flex flex-col">
                                <h3 className="text-[var(--text-primary)] text-2xl font-bold tracking-tight">Real-time Summary</h3>
                                <p className="text-gray-400 text-xs">
                                    {items.length > 0 ? `${items.length} segments` : 'Generating detailed notes from video transcript...'}
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
                        <div className="flex items-center justify-between border-b border-[var(--border-color)] pb-2">
                            <div className="flex items-center gap-2">
                                <StickyNote className="w-6 h-6 text-primary" />
                                <h3 className="text-[var(--text-primary)] text-xl font-bold tracking-tight">Real-time Summary</h3>
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

                {isExpanded && <ProcessingStatusBar />}

                {/* Timeline Content */}
                <div className={`flex flex-col gap-1 relative pl-2 ${isExpanded ? 'pt-4' : ''}`}>
                    <div className="absolute left-[8.5rem] top-4 bottom-4 w-px bg-border-color hidden md:block"></div>

                    {loading && items.length === 0 && (
                        <div className="flex justify-center py-8">
                            <Loader2 className="w-6 h-6 animate-spin text-primary" />
                        </div>
                    )}

                    {!loading && items.length === 0 && (
                        <div className="py-8 text-center text-gray-400 text-sm">
                            {isProcessing ? '요약이 생성되면 여기에 표시됩니다...' : '요약 데이터가 없습니다.'}
                        </div>
                    )}

                    {items.map((item, index) => renderSummaryItem(item, index))}

                    {/* Skeleton items while processing */}
                    {isProcessing && items.length > 0 && (
                        <div className={`group flex flex-col md:flex-row gap-2 md:gap-8 ${isExpanded ? 'p-4 rounded-xl opacity-50' : 'p-3 rounded-lg opacity-50'} transition-colors cursor-pointer`}>
                            <div className="md:w-24 shrink-0 flex md:justify-end">
                                <span className="font-mono text-sm text-gray-400 bg-surface/50 px-2 py-0.5 rounded border border-[var(--border-color)] h-fit">--:--</span>
                            </div>
                            <div className="flex-1">
                                <div className="h-4 w-3/4 bg-[var(--border-color)] rounded animate-pulse mt-1"></div>
                                <div className="h-4 w-1/2 bg-[var(--border-color)] rounded animate-pulse mt-2"></div>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

export default SummaryPanel;
