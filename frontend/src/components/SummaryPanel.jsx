import React, { useState, useEffect, useCallback, useRef } from 'react';
import { StickyNote, RefreshCw, Minimize2, Maximize2, Loader2, AlertTriangle, RotateCcw, Clock, Bot } from 'lucide-react';
import { getVideoSummaries, restartProcessing } from '../api/videos';
import useVideoStatusStream from '../hooks/useVideoStatusStream';
import MarkdownRenderer from './MarkdownRenderer';

function formatMs(ms) {
    if (ms == null) return '--:--';
    const totalSec = Math.floor(ms / 1000);
    const m = Math.floor(totalSec / 60);
    const s = totalSec % 60;
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

function SummaryPanel({ isExpanded, onToggleExpand, videoId, onSeekTo, currentTimeMs, chatbotOpen, onAskChatBot }) {
    const [items, setItems] = useState([]);
    const [initialLoading, setInitialLoading] = useState(true);
    const [videoStatus, setVideoStatus] = useState(null);
    const [progressInfo, setProgressInfo] = useState({ current: 0, total: 0, percent: 0, stage: '' });
    const prevCountRef = useRef(0);
    const [errorMessage, setErrorMessage] = useState(null);
    const [isRestarting, setIsRestarting] = useState(false);
    const [streamKey, setStreamKey] = useState(0);
    const prevBatchRef = useRef(-1);
    const staleFlagRef = useRef(false);
    const [preprocessInfo, setPreprocessInfo] = useState({ status: null });
    const [pipelineMode, setPipelineMode] = useState('sequential');

    const isProcessing = videoStatus && !['DONE', 'FAILED'].includes((videoStatus).toUpperCase());

    const handleStatus = useCallback((statusData) => {
        if (!statusData) return;
        setVideoStatus(statusData.video_status);
        setErrorMessage(statusData.error_message || null);

        const ppJob = statusData.preprocess_job;
        if (ppJob) setPreprocessInfo({ status: (ppJob.status || '').toUpperCase() });
        setPipelineMode(statusData.pipeline_mode || 'sequential');

        const job = statusData.processing_job;
        if (job) {
            const current = job.progress_current || 0;
            const total = job.progress_total || 0;
            const jobStatus = (job.status || '').toUpperCase();

            let percent;
            if (jobStatus === 'DONE') {
                percent = 100;
            } else if (total > 0) {
                const basePct = (current / total) * 100;
                const batchWidth = 100 / total;
                const stageWeight = { VLM_RUNNING: 0.3, SUMMARY_RUNNING: 0.6, JUDGE_RUNNING: 0.9 };

                if (prevBatchRef.current >= 0 && current !== prevBatchRef.current) {
                    staleFlagRef.current = true;
                }
                prevBatchRef.current = current;
                if (staleFlagRef.current && jobStatus !== 'JUDGE_RUNNING') {
                    staleFlagRef.current = false;
                }
                const stageBonus = staleFlagRef.current ? 0 : (stageWeight[jobStatus] || 0) * batchWidth;
                percent = Math.min(Math.round(basePct + stageBonus), 99);
            } else if (current > 0) {
                percent = Math.min(Math.round(95 * (1 - 1 / (1 + current * 0.3))), 95);
            } else {
                percent = 0;
            }

            setProgressInfo({ current, total, percent, stage: jobStatus });
        }
    }, []);

    const handleSummaries = useCallback((summaryData) => {
        if (!summaryData?.items) return;
        const newItems = summaryData.items;
        if (newItems.length !== prevCountRef.current) {
            setItems(newItems);
            prevCountRef.current = newItems.length;
        }
    }, []);

    const handleDone = useCallback((data) => {
        if (data?.video_status) {
            setVideoStatus(data.video_status);
        }
    }, []);

    const { reconnect } = useVideoStatusStream(videoId, {
        enabled: !!videoId,
        onStatus: handleStatus,
        onSummaries: handleSummaries,
        onDone: handleDone,
    });

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
    }, [videoId, streamKey]);

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
        if (s === 'PROCESSING' && progressInfo.total === 0 && progressInfo.current === 0) return '분석 시작 중...';
        if (s === 'VLM_RUNNING') return 'AI 분석 진행 중...';
        if (s === 'SUMMARY_RUNNING') return '요약 생성 중...';
        if (s === 'JUDGE_RUNNING') return '품질 평가 중...';
        if (s === 'PREPROCESSING' || s === 'PREPROCESS_DONE') return '전처리 완료, 분석 대기 중...';
        return '영상 분석 중...';
    })();



    const getBulletText = (b) => {
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
            if (onSeekTo && item.start_ms != null) {
                onSeekTo(item.start_ms);
            }
        };

        if (!isExpanded) {
            return (
                <div
                    key={item.summary_id || index}
                    className={`group flex flex-col border border-[var(--border-color)] ${isActive
                        ? 'bg-primary/10 border-primary shadow-[0_0_15px_rgba(224,126,99,0.1)] ring-1 ring-primary/20'
                        : 'bg-white/[0.02] dark:bg-white/[0.03] hover:bg-white/[0.05] dark:hover:bg-white/[0.06] shadow-sm'
                        } py-2.5 rounded-xl cursor-pointer ${isNew ? 'animate-fade-in' : ''} transition-all duration-300`}
                    onClick={handleCardClick}
                >
                    <div className="flex flex-row items-start relative">
                        {/* Simple View: Left Side Seg Info */}
                        <div className="hidden lg:flex w-[60px] self-stretch shrink-0 items-center justify-center select-none border-r border-[var(--border-color)]/30">
                            <div className="flex flex-col items-center justify-center leading-none">
                                <span className="text-[var(--text-primary)] font-bold text-[13px]">Seg {segIdx}</span>
                                <div className="h-[2px]" />
                                <span className="text-[var(--text-secondary)] text-[10px] font-medium opacity-70">
                                    {item.start_ms != null ? formatMs(item.start_ms) : '00:00'}
                                </span>
                            </div>
                        </div>

                        <div className="flex-1 min-w-0 px-4">
                            {/* Mobile: Center Title */}
                            <div className="lg:hidden flex items-center justify-center h-6 mb-1">
                                <h4 className="text-[var(--text-primary)] font-semibold truncate text-center text-[15px]">
                                    {title}
                                </h4>
                            </div>

                            {!hasSections && <p className="text-[var(--text-secondary)] text-[13px] italic">요약 데이터 준비 중...</p>}

                            {bullets.length > 0 && (
                                <div className="mb-0">
                                    <ul className="list-disc list-inside text-[var(--text-secondary)] text-sm md:text-[15px] space-y-1 leading-relaxed">
                                        {bullets.map((b, i) => (
                                            <li key={i} className="markdown-inline">
                                                <MarkdownRenderer>{getBulletText(b)}</MarkdownRenderer>
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            )}
                        </div>

                        {/* Action Buttons */}
                        <div className="absolute right-2 top-0 flex items-center h-full opacity-0 group-hover:opacity-100 transition-opacity">
                            <button
                                onClick={(e) => {
                                    e.stopPropagation();
                                    onAskChatBot?.({ segIdx, timeRange, content: item.summary });
                                }}
                                className="w-8 h-8 flex items-center justify-center rounded-full hover:bg-white/10 text-[var(--text-secondary)] hover:text-blue-400 transition-colors"
                                title="원본 분석 답변"
                            >
                                <Clock size={16} />
                            </button>
                        </div>
                    </div>
                </div>
            );
        }

        /* Detail View: Card-based multi-row layout */
        return (
            <div
                key={item.summary_id || index}
                className={`group flex flex-col border border-[var(--border-color)] ${isActive
                    ? 'bg-primary/10 border-primary shadow-[0_0_20px_rgba(224,126,99,0.15)] ring-1 ring-primary/20'
                    : 'bg-white/[0.02] dark:bg-white/[0.03] hover:bg-white/[0.05] dark:hover:bg-white/[0.06] shadow-sm'
                    } p-4 sm:p-5 rounded-2xl cursor-pointer ${isNew ? 'animate-fade-in' : ''} transition-all duration-300 relative`}
                onClick={handleCardClick}
            >
                {/* Float Clock Button - Top Right */}


                {/* 1. Top Header Row Info */}
                <div className="flex flex-row items-center border-b border-[var(--border-color)]/30 pb-3 mb-4">
                    <div className="w-[60px] shrink-0 border-r border-[var(--border-color)]/30 mr-4 self-stretch flex items-center justify-center">
                        <div className="flex flex-col items-center justify-center h-full">
                            <span className="text-[var(--text-primary)] font-bold text-[15px]">Seg {segIdx}</span>
                        </div>
                    </div>
                    <div className="flex-1 min-w-0 flex items-center justify-between">
                        <h4 className="text-[var(--text-primary)] font-bold text-lg flex items-center gap-2">
                            <span className="text-[var(--text-secondary)] text-[20px] font-medium opacity-70 h-[20px] flex items-center">{timeRange}</span>
                            <button
                                onClick={(e) => {
                                    e.stopPropagation();
                                    onAskChatBot?.({ segIdx, timeRange, content: item.summary });
                                }}
                                className="flex items-center justify-center w-8 h-8 rounded-full hover:bg-white/10 text-orange-500 transition-colors ml-1"
                                title="원본 분석 답변"
                            >
                                <Bot size={24} />
                            </button>
                        </h4>
                    </div>
                </div>

                {/* 2. Summary Section */}
                {bullets.length > 0 && (
                    <div className="flex flex-row items-start">
                        <div className="hidden lg:block w-[60px] shrink-0 h-4 border-r border-[var(--border-color)]/30 mr-4" />
                        <div className="flex-1 min-w-0">
                            <ul className="list-disc list-inside text-[var(--text-secondary)] text-sm md:text-[15px] space-y-2 leading-relaxed">
                                {bullets.map((b, i) => (
                                    <li key={i} className="markdown-inline">
                                        <MarkdownRenderer>{getBulletText(b)}</MarkdownRenderer>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    </div>
                )}

                {/* 3. Definitions Section */}
                {definitions.length > 0 && (
                    <>
                        {/* Definitions Top Spacer: 12px height */}
                        <div className="flex flex-row items-center mt-8">
                            <div className="hidden lg:block w-[60px] shrink-0" />
                            <div className="flex-1 min-w-0 h-3" />
                        </div>
                        {/* Definitions Content Area: Label on the left, centered vertically */}
                        <div className="flex flex-row items-stretch mt-1">
                            <div className="hidden lg:flex w-[60px] shrink-0 items-center justify-center">
                                <span className="text-primary text-[11px] font-bold uppercase tracking-wider h-4 flex items-center justify-center">정의</span>
                            </div>
                            <div className="flex-1 min-w-0">
                                <ul className="list-disc list-inside ml-4 text-[var(--text-secondary)] text-sm md:text-base space-y-1.5 leading-relaxed">
                                    {definitions.map((d, i) => (
                                        <li key={i} className="markdown-inline">
                                            <MarkdownRenderer inline className="font-semibold text-[var(--text-primary)]">{d.term}</MarkdownRenderer>
                                            {': '}
                                            <MarkdownRenderer inline>{d.definition}</MarkdownRenderer>
                                        </li>
                                    ))}
                                </ul>
                            </div>
                            <div className="w-[2px] shrink-0" />
                        </div>
                    </>
                )}

                {/* 4. Explanations Section */}
                {explanations.length > 0 && (
                    <>
                        {/* Explanations Top Spacer: 12px height */}
                        <div className="flex flex-row items-center mt-8">
                            <div className="hidden lg:block w-[60px] shrink-0" />
                            <div className="flex-1 min-w-0 h-3" />
                        </div>
                        {/* Explanations Content Area: Label on the left, centered vertically */}
                        <div className="flex flex-row items-stretch mt-1">
                            <div className="hidden lg:flex w-[60px] shrink-0 items-center justify-center">
                                <span className="text-primary text-[11px] font-bold uppercase tracking-wider h-4 flex items-center justify-center">해설</span>
                            </div>
                            <div className="flex-1 min-w-0">
                                <ul className="list-disc list-inside ml-4 text-[var(--text-secondary)] text-sm md:text-base space-y-1.5 leading-relaxed">
                                    {explanations.map((e, i) => (
                                        <li key={i} className="markdown-inline">
                                            <MarkdownRenderer>{e.point || e.text || JSON.stringify(e)}</MarkdownRenderer>
                                        </li>
                                    ))}
                                </ul>
                            </div>
                            <div className="w-[2px] shrink-0" />
                        </div>
                    </>
                )}

                {/* 5. Open Questions Section */}
                {openQuestions.length > 0 && (
                    <>
                        {/* Open Questions Top Spacer: 12px height */}
                        <div className="flex flex-row items-center mt-8">
                            <div className="hidden lg:block w-[60px] shrink-0" />
                            <div className="flex-1 min-w-0 h-3" />
                        </div>
                        {/* Open Questions Content Area: Label on the left, centered vertically */}
                        <div className="flex flex-row items-stretch mt-1 mb-2">
                            <div className="hidden lg:flex w-[60px] shrink-0 items-center justify-center">
                                <span className="text-primary text-[11px] font-bold uppercase tracking-wider text-center leading-tight h-4 flex items-center justify-center">열린 질문</span>
                            </div>
                            <div className="flex-1 min-w-0">
                                <ul className="list-disc list-inside ml-4 text-[var(--text-secondary)] text-sm md:text-base space-y-1.5 leading-relaxed italic">
                                    {openQuestions.map((q, i) => (
                                        <li key={i} className="markdown-inline">
                                            <MarkdownRenderer>{q.question || q.text || JSON.stringify(q)}</MarkdownRenderer>
                                        </li>
                                    ))}
                                </ul>
                            </div>
                            <div className="w-[2px] shrink-0" />
                        </div>
                    </>
                )}
            </div>
        );
    };

    return (
        <div className={`flex flex-col transition-all duration-300 ${isExpanded
            ? 'items-center py-6 lg:py-10 w-full'
            : `gap-0 ${chatbotOpen ? 'pb-[70dvh] lg:pb-24' : 'pb-[70dvh] lg:pb-24'} w-full`}`}>
            {/* 1. Header Section Area (Grouped Spacers + Header) */}
            {!isExpanded ? (
                <div className="flex-none">
                    {/* Visual Spacer: Header Top h-3 (12px) */}
                    <div className="h-3" />

                    <div className="w-full">
                        <div className="relative flex items-center justify-between h-6 box-content">
                            <div className="w-1" /> {/* Far Side Spacer (4px) */}
                            {/* Left: Status Badge Wrapper */}
                            <div className="flex-1 flex items-center z-10">
                                {videoStatus?.toUpperCase() === 'DONE' && (
                                    <div className="px-1 flex items-center h-6 gap-1 sm:gap-1.5 animate-fade-in transition-all">
                                        <span className="text-green-500 text-[10px] sm:text-xs font-semibold whitespace-nowrap">
                                            분석 완료
                                        </span>
                                    </div>
                                )}
                                {isProcessing && (
                                    <div className="px-1 flex items-center h-6 gap-1 sm:gap-1.5 animate-pulse transition-all">
                                        <span className="text-primary text-[10px] sm:text-xs font-semibold whitespace-nowrap">
                                            <span className="sm:hidden">분석 중</span>
                                            <span className="hidden sm:inline">{stageLabel}</span>
                                        </span>
                                    </div>
                                )}
                            </div>

                            {/* Center: Title (Icon + Text) - Absolute Center */}
                            <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 flex items-center gap-1.5 sm:gap-2.5 whitespace-nowrap h-6 transition-all">
                                <StickyNote className="w-4 h-4 text-primary shrink-0" />
                                <h3 className="text-[var(--text-primary)] text-sm sm:text-lg font-bold tracking-tight m-0 leading-none">
                                    Real-time<span className="hidden sm:inline"> Summary</span>
                                </h3>
                                {items.length > 0 && (
                                    <span className="text-gray-400 text-[10px] sm:text-xs ml-0.5 leading-none">({items.length})</span>
                                )}
                            </div>

                            {/* Right: Toggle Button Wrapper */}
                            <div className="flex-1 flex justify-end z-10">
                                <button
                                    onClick={onToggleExpand}
                                    className="text-gray-400 hover:text-[var(--text-primary)] text-xs flex items-center gap-1 sm:gap-2 px-2.5 sm:px-4 h-6 rounded-xl hover:bg-surface-highlight transition-all"
                                >
                                    <Maximize2 className="w-4 h-4 shrink-0" />
                                    <span className="flex flex-col items-center leading-tight text-center">
                                        <span className="font-semibold whitespace-nowrap text-[10px] sm:text-xs">상세보기</span>
                                        <span className="text-[9px] text-gray-500 hidden md:block">정의 · 해설 · 열린 질문</span>
                                    </span>
                                </button>
                            </div>
                            <div className="w-1" /> {/* Far Side Spacer (4px) */}
                        </div>
                    </div>

                    {/* Visual Spacer: Header Bottom h-3 (12px) */}
                    <div className="h-3" />
                </div>
            ) : (
                /* Expanded Header Area */
                <div className="w-full max-w-[1024px] mb-6 sticky top-0 z-40 bg-[var(--bg-primary)]/95 backdrop-blur border-b border-black/5 dark:border-white/10">
                    <div className="h-3" /> {/* Header Top Spacer (12px) */}
                    <div className="relative flex items-center justify-between h-6 box-content">
                        <div className="w-1" /> {/* Far Side Spacer (4px) */}
                        {/* Left: Status Badge Wrapper */}
                        <div className="flex-1 flex items-center z-10">
                            {videoStatus?.toUpperCase() === 'DONE' && (
                                <div className="px-1 flex items-center h-6 gap-1 sm:gap-1.5 animate-fade-in transition-all">
                                    <span className="text-green-500 text-[10px] sm:text-xs font-semibold whitespace-nowrap">
                                        분석 완료
                                    </span>
                                </div>
                            )}
                            {isProcessing && (
                                <div className="px-1 flex items-center h-6 gap-1 sm:gap-1.5 animate-pulse transition-all">
                                    <span className="text-primary text-[10px] sm:text-xs font-semibold whitespace-nowrap">
                                        <span className="sm:hidden">분석 중</span>
                                        <span className="hidden sm:inline">{stageLabel}</span>
                                    </span>
                                </div>
                            )}
                        </div>

                        {/* Center: Title (Icon + Text) - Absolute Center */}
                        <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 flex items-center gap-1.5 sm:gap-2.5 whitespace-nowrap h-6 transition-all">
                            <StickyNote className="w-4 h-4 text-primary shrink-0" />
                            <h3 className="text-[var(--text-primary)] text-sm sm:text-lg font-bold tracking-tight m-0 leading-none">
                                Real-time<span className="hidden sm:inline"> Summary</span>
                            </h3>
                            {items.length > 0 && (
                                <span className="text-gray-400 text-[10px] sm:text-xs ml-0.5 leading-none">({items.length})</span>
                            )}
                        </div>

                        {/* Right: Toggle Button Wrapper (Minimize) */}
                        <div className="flex-1 flex justify-end z-10">
                            <button
                                onClick={onToggleExpand}
                                className="text-gray-400 hover:text-[var(--text-primary)] text-xs flex items-center gap-1 sm:gap-2 px-2.5 sm:px-4 h-6 rounded-xl hover:bg-surface-highlight transition-all"
                            >
                                <Minimize2 className="w-4 h-4 shrink-0" />
                                <span className="flex flex-col items-center leading-tight text-center">
                                    <span className="font-semibold whitespace-nowrap text-[10px] sm:text-xs">간단히 보기</span>
                                    <span className="text-[9px] text-gray-500 hidden md:block">리스트로 돌아가기</span>
                                </span>
                            </button>
                        </div>
                        <div className="w-1" /> {/* Far Right Spacer (4px) */}
                    </div>
                    <div className="h-3" /> {/* Header Bottom Spacer (12px) */}
                </div>
            )}



            {/* 3. Timeline Area Wrapper */}
            <div className={`w-full ${isExpanded ? 'max-w-[1024px]' : ''}`}>
                <div className={`flex flex-col gap-3 relative ${isExpanded ? 'pt-4' : 'mt-4'}`}>
                    {initialLoading && items.length === 0 && (
                        <div className="flex justify-center py-8">
                            <Loader2 className="w-5 h-5 animate-spin text-primary" />
                        </div>
                    )}
                    {!initialLoading && items.length === 0 && (
                        <div className="py-8 text-center text-gray-400 text-sm">
                            {isProcessing ? '요약이 생성되면 여기에 실시간으로 표시됩니다...' : '요약 데이터가 없습니다.'}
                        </div>
                    )}
                    {items.map((item, index) => renderSummaryItem(item, index))}

                    {/* Visual Spacer: List Bottom - Scales with ChatBot on Mobile */}
                    <div className={`flex-none flex flex-col items-center justify-center transition-all duration-500 ${chatbotOpen ? 'h-[50dvh] lg:h-32' : 'h-32'}`}>
                        <footer className="home-footer opacity-30 mt-auto">
                            <p className="text-[var(--text-secondary)] text-[10px] md:text-xs">
                                © 2026 Re:View. AI-Powered Video Analysis Platform
                            </p>
                        </footer>
                    </div>

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
