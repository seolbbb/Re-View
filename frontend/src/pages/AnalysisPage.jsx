import { useState, useEffect, useRef, useCallback } from 'react';
import { useParams } from 'react-router-dom';
import { useTheme } from '../context/ThemeContext';
import { useVideo } from '../context/VideoContext';
import { getVideoStatus, getVideoSummaries } from '../api/videos';
import useVideoStatusStream from '../hooks/useVideoStatusStream';
import Sidebar from '../components/Sidebar';
import VideoPlayer from '../components/VideoPlayer';
import ChatBot from '../components/ChatBot';
import SummaryPanel from '../components/SummaryPanel';
import katex from 'katex';
import { Menu, ChevronRight, Sun, Moon, Download, ChevronDown, Video } from 'lucide-react';

function formatMs(ms) {
    if (ms == null) return '--:--';
    const totalSec = Math.floor(ms / 1000);
    const m = Math.floor(totalSec / 60);
    const s = totalSec % 60;
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

/** Replace $...$ and $$...$$ with KaTeX HTML, and escape the rest for safe HTML. */
function renderTextToHtml(text) {
    if (!text) return '';
    const parts = [];
    // Split on display math first, then inline math
    let remaining = text;
    // Display math $$...$$
    remaining = remaining.replace(/\$\$([\s\S]+?)\$\$/g, (_m, inner) => {
        try { return katex.renderToString(inner.trim(), { displayMode: true, throwOnError: false }); }
        catch { return _m; }
    });
    // Inline math $...$
    remaining = remaining.replace(/\$([^\$\n]+?)\$/g, (_m, inner) => {
        try { return katex.renderToString(inner.trim(), { displayMode: false, throwOnError: false }); }
        catch { return _m; }
    });
    // Bold **...**
    remaining = remaining.replace(/\*\*([\s\S]+?)\*\*/g, '<strong>$1</strong>');
    return remaining;
}

function buildHtmlExport(videoName, items) {
    const lines = [];
    lines.push(`<h1>${videoName}</h1>`);
    items.forEach((item, index) => {
        const segIdx = item.segment_index ?? index + 1;
        const timeRange = `${formatMs(item.start_ms)}\u2013${formatMs(item.end_ms)}`;
        lines.push(`<h2>Segment ${segIdx} (${timeRange})</h2>`);
        const summary = item.summary || {};
        const bullets = summary.bullets || [];
        const definitions = summary.definitions || [];
        const explanations = summary.explanations || [];
        const openQuestions = summary.open_questions || [];

        if (bullets.length > 0) {
            lines.push('<h3>요약</h3><ul>');
            bullets.forEach((b) => {
                const text = typeof b === 'string' ? b : (b.bullet_id ? `(${b.bullet_id}) ` : '') + (b.claim || b.text || JSON.stringify(b));
                lines.push(`<li>${renderTextToHtml(text)}</li>`);
            });
            lines.push('</ul>');
        }
        if (definitions.length > 0) {
            lines.push('<h3>정의</h3><ul>');
            definitions.forEach((d) => {
                lines.push(`<li><strong>${renderTextToHtml(d.term)}</strong>: ${renderTextToHtml(d.definition)}</li>`);
            });
            lines.push('</ul>');
        }
        if (explanations.length > 0) {
            lines.push('<h3>해설</h3><ul>');
            explanations.forEach((e) => {
                lines.push(`<li>${renderTextToHtml(e.point || e.text || JSON.stringify(e))}</li>`);
            });
            lines.push('</ul>');
        }
        if (openQuestions.length > 0) {
            lines.push('<h3>열린 질문</h3><ul>');
            openQuestions.forEach((q) => {
                lines.push(`<li>${renderTextToHtml(q.question || q.text || JSON.stringify(q))}</li>`);
            });
            lines.push('</ul>');
        }
    });
    return lines.join('\n');
}

function buildMarkdownExport(videoName, items) {
    const lines = [`# ${videoName}`, ''];
    items.forEach((item, index) => {
        const segIdx = item.segment_index ?? index + 1;
        const timeRange = `${formatMs(item.start_ms)}–${formatMs(item.end_ms)}`;
        lines.push(`## Segment ${segIdx} (${timeRange})`);

        const summary = item.summary || {};
        const bullets = summary.bullets || [];
        const definitions = summary.definitions || [];
        const explanations = summary.explanations || [];
        const openQuestions = summary.open_questions || [];

        if (bullets.length > 0) {
            lines.push('### 요약');
            bullets.forEach((b) => {
                const text = typeof b === 'string' ? b : (b.bullet_id ? `(${b.bullet_id}) ` : '') + (b.claim || b.text || JSON.stringify(b));
                lines.push(`- ${text}`);
            });
            lines.push('');
        }

        if (definitions.length > 0) {
            lines.push('### 정의');
            definitions.forEach((d) => {
                lines.push(`- **${d.term}**: ${d.definition}`);
            });
            lines.push('');
        }

        if (explanations.length > 0) {
            lines.push('### 해설');
            explanations.forEach((e) => {
                lines.push(`- ${e.point || e.text || JSON.stringify(e)}`);
            });
            lines.push('');
        }

        if (openQuestions.length > 0) {
            lines.push('### 열린 질문');
            openQuestions.forEach((q) => {
                lines.push(`- ${q.question || q.text || JSON.stringify(q)}`);
            });
            lines.push('');
        }
    });
    return lines.join('\n');
}

function AnalysisPage() {
    const { id: videoId } = useParams();
    const [isExpanded, setIsExpanded] = useState(false);
    const { theme, toggleTheme } = useTheme();
    const { setCurrentVideoId } = useVideo();
    const [videoInfo, setVideoInfo] = useState(null);
    const [exportOpen, setExportOpen] = useState(false);
    const exportRef = useRef(null);

    // Close export dropdown on outside click
    useEffect(() => {
        const handleClick = (e) => {
            if (exportRef.current && !exportRef.current.contains(e.target)) {
                setExportOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClick);
        return () => document.removeEventListener('mousedown', handleClick);
    }, []);

    const handleHeaderStatus = useCallback((statusData) => {
        if (!statusData) return;
        setVideoInfo((prev) => {
            if (!prev) return statusData;
            const nextStatus = statusData.video_status;
            const nextErr = statusData.error_message;
            const nextName = statusData.video_name;

            if (
                prev.video_status === nextStatus &&
                prev.error_message === nextErr &&
                prev.video_name === nextName
            ) {
                return prev;
            }

            return {
                ...prev,
                video_status: nextStatus,
                error_message: nextErr,
                video_name: nextName ?? prev.video_name,
            };
        });
    }, []);

    // Keep the header status label in sync (restart/complete transitions).
    useVideoStatusStream(videoId, {
        enabled: !!videoId,
        onStatus: handleHeaderStatus,
    });

    // Shared video element ref & playback state for preserving across expand/collapse
    const videoElRef = useRef(null);
    const playbackStateRef = useRef({ time: 0, playing: false });

    // Scroll-triggered PiP
    const [scrollPip, setScrollPip] = useState(false);
    const videoSectionRef = useRef(null);
    const scrollContainerRef = useRef(null);
    const scrollPipRef = useRef(false);

    // Current playback time (ms) for segment highlighting
    const [currentPlaybackMs, setCurrentPlaybackMs] = useState(0);
    const timeThrottleRef = useRef(0);
    const handleTimeUpdate = useCallback((timeSec) => {
        const now = Date.now();
        if (now - timeThrottleRef.current >= 500) {
            timeThrottleRef.current = now;
            setCurrentPlaybackMs(timeSec * 1000);
        }
    }, []);

    const handleToggleExpand = useCallback((expanded) => {
        // Save current playback state before the VideoPlayer remounts
        const v = videoElRef.current;
        if (v) {
            playbackStateRef.current = {
                time: v.currentTime,
                playing: !v.paused,
            };
        }
        setIsExpanded(expanded);
    }, []);

    // Detect when video section scrolls out of view → show PiP
    useEffect(() => {
        if (isExpanded) {
            setScrollPip(false);
            scrollPipRef.current = false;
            return;
        }
        const section = videoSectionRef.current;
        const container = scrollContainerRef.current;
        if (!section || !container) return;

        const observer = new IntersectionObserver(
            ([entry]) => {
                const shouldPip = !entry.isIntersecting;
                if (shouldPip === scrollPipRef.current) return;
                scrollPipRef.current = shouldPip;
                setScrollPip(shouldPip);
            },
            { root: container, threshold: 0 },
        );

        observer.observe(section);
        return () => observer.disconnect();
    }, [isExpanded]);

    const handleScrollPipClose = useCallback(() => {
        videoSectionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, []);

    const handleSeekTo = useCallback((timeMs) => {
        const v = videoElRef.current;
        if (!v) return;
        v.currentTime = timeMs / 1000;
        setCurrentPlaybackMs(timeMs);
        if (v.paused) {
            v.play().catch(() => { });
        }
    }, []);

    // Keyboard shortcuts: Arrow keys (±5s seek), Spacebar (play/pause)
    useEffect(() => {
        const handleKeyDown = (e) => {
            const tag = e.target.tagName;
            if (tag === 'INPUT' || tag === 'TEXTAREA' || e.target.isContentEditable) return;

            const v = videoElRef.current;
            if (!v) return;

            if (e.key === 'ArrowLeft') {
                e.preventDefault();
                v.currentTime = Math.max(0, v.currentTime - 5);
            } else if (e.key === 'ArrowRight') {
                e.preventDefault();
                v.currentTime = Math.min(v.duration || 0, v.currentTime + 5);
            } else if (e.key === ' ' || e.code === 'Space') {
                e.preventDefault();
                if (v.paused) {
                    v.play().catch(() => {});
                } else {
                    v.pause();
                }
            }
        };

        document.addEventListener('keydown', handleKeyDown);
        return () => document.removeEventListener('keydown', handleKeyDown);
    }, []);

    useEffect(() => {
        if (videoId) {
            setCurrentVideoId(videoId);
            getVideoStatus(videoId)
                .then((data) => setVideoInfo(data))
                .catch(() => { });
        }
    }, [videoId, setCurrentVideoId]);

    const videoName = videoInfo?.video_name || (videoInfo ? videoId : '로딩 중...');
    const statusLabel = videoInfo?.video_status || 'Loading';

    const handleExportMarkdown = async () => {
        setExportOpen(false);
        try {
            const data = await getVideoSummaries(videoId);
            const items = data.items || [];
            const md = buildMarkdownExport(videoName, items);
            const blob = new Blob([md], { type: 'text/markdown;charset=utf-8' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${videoName}.md`;
            a.click();
            URL.revokeObjectURL(url);
        } catch (err) {
            console.error('Markdown export failed:', err);
        }
    };


    return (
        <div className="bg-[var(--bg-primary)] text-[var(--text-primary)] font-display flex h-screen overflow-hidden selection:bg-primary/40 selection:text-white transition-colors duration-300" data-theme={theme}>
            {/* Left Sidebar */}
            <Sidebar />

            {/* Main Content Area */}
            <main className="flex-1 flex flex-col min-w-0 h-full relative bg-[var(--bg-primary)] transition-colors duration-300">
                {/* Header / Breadcrumbs */}
                <header className="h-16 flex items-center justify-between px-6 border-b border-[var(--border-color)] bg-[var(--bg-primary)]/95 backdrop-blur z-30 shrink-0">
                    <div className="flex items-center gap-2">
                        <a href="/" className="text-gray-400 hover:text-[var(--text-primary)] transition-colors" title="홈으로">
                            <Video className="w-5 h-5" />
                        </a>
                        <button className="md:hidden mr-2 text-gray-400">
                            <Menu className="w-5 h-5" />
                        </button>
                        <a href="/" className="text-gray-400 text-sm font-medium hover:text-[var(--text-primary)] transition-colors">Library</a>
                        <ChevronRight className="w-4 h-4 text-gray-400" />
                        <div className="flex items-center gap-2">
                            {videoInfo ? (
                                <span className="text-[var(--text-primary)] text-sm font-medium truncate max-w-[200px]">{videoName}</span>
                            ) : (
                                <span className="text-gray-400 text-sm font-medium">
                                    <span className="inline-flex">
                                        <span className="animate-pulse">.</span>
                                        <span className="animate-pulse" style={{ animationDelay: '0.2s' }}>.</span>
                                        <span className="animate-pulse" style={{ animationDelay: '0.4s' }}>.</span>
                                    </span>
                                </span>
                            )}
                            <span className="bg-primary/20 text-primary text-[10px] font-bold px-1.5 py-0.5 rounded uppercase tracking-wide">{statusLabel}</span>
                        </div>
                    </div>
                    <div className="flex items-center gap-4">
                        <button
                            onClick={toggleTheme}
                            className="flex items-center justify-center text-gray-400 hover:text-[var(--text-primary)] transition-colors mr-2"
                            title={theme === 'dark' ? "Switch to Light Mode" : "Switch to Dark Mode"}
                        >
                            {theme === 'dark' ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
                        </button>
                        <div className="relative" ref={exportRef}>
                            <button
                                onClick={() => setExportOpen((o) => !o)}
                                className="flex items-center gap-2 text-gray-400 hover:text-[var(--text-primary)] transition-colors"
                            >
                                <Download className="w-5 h-5" />
                                <span className="text-sm font-medium hidden sm:block">Export</span>
                                <ChevronDown className="w-4 h-4" />
                            </button>
                            {exportOpen && (
                                <div className="absolute right-0 top-full mt-2 w-48 bg-[var(--bg-secondary,var(--bg-primary))] border border-[var(--border-color)] rounded-lg shadow-xl z-50 overflow-hidden">
                                    <button
                                        onClick={handleExportMarkdown}
                                        className="w-full text-left px-4 py-2.5 text-sm text-[var(--text-primary)] hover:bg-surface-highlight transition-colors"
                                    >
                                        Markdown (.md)
                                    </button>
                                </div>
                            )}
                        </div>
                    </div>
                </header>

                {/* Workspace Grid */}
                <div className="flex flex-1 overflow-hidden relative">
                    {/* Left Column: Video & Summary */}
                    <div className="flex-1 relative min-w-0 bg-[var(--bg-primary)]">
                        {isExpanded ? (
                            <>
                                {/* Detailed Summary View (Full) */}
                                <SummaryPanel isExpanded={true} onToggleExpand={() => handleToggleExpand(false)} videoId={videoId} onSeekTo={handleSeekTo} currentTimeMs={currentPlaybackMs} />

                                {/* PIP Video Player */}
                                <div className="absolute bottom-6 right-6 w-80 lg:w-96 aspect-video z-50 transition-all hover:scale-[1.02]">
                                    <VideoPlayer
                                        isPip={true}
                                        onTogglePip={() => handleToggleExpand(false)}
                                        videoId={videoId}
                                        videoElRef={videoElRef}
                                        playbackRestore={playbackStateRef}
                                        onTimeUpdate={handleTimeUpdate}
                                        className="relative w-full h-full rounded-xl overflow-hidden shadow-[0_8px_30px_rgb(0,0,0,0.5)] border border-[var(--border-color)] ring-1 ring-white/5 group bg-black"
                                    />
                                </div>
                            </>
                        ) : (
                            // Normal View
                            <div ref={scrollContainerRef} className="flex flex-col overflow-y-auto custom-scrollbar p-6 lg:p-8 gap-8 min-w-0 h-full">
                                {/* Video Player Section */}
                                <div ref={videoSectionRef} className="flex flex-col gap-4 shrink-0 w-full max-w-[1600px] mx-auto">
                                    <div className={scrollPip
                                        ? "absolute bottom-6 right-6 w-80 lg:w-96 z-50 aspect-video animate-pipSlideIn transition-transform hover:scale-[1.02]"
                                        : ""}>
                                        <VideoPlayer
                                            isPip={scrollPip}
                                            onTogglePip={scrollPip ? handleScrollPipClose : undefined}
                                            videoId={videoId}
                                            videoElRef={videoElRef}
                                            playbackRestore={playbackStateRef}
                                            onTimeUpdate={handleTimeUpdate}
                                            className={scrollPip
                                                ? "relative w-full h-full rounded-xl overflow-hidden shadow-[0_8px_30px_rgb(0,0,0,0.5)] border border-[var(--border-color)] ring-1 ring-white/5 group bg-black"
                                                : undefined}
                                        />
                                    </div>
                                    {scrollPip && <div style={{ aspectRatio: '16/9' }} className="rounded-xl" />}
                                </div>

                                {/* Summary Section */}
                                <SummaryPanel isExpanded={false} onToggleExpand={() => handleToggleExpand(true)} videoId={videoId} onSeekTo={handleSeekTo} currentTimeMs={currentPlaybackMs} />
                            </div>
                        )}
                    </div>

                    {/* Right Column: AI Chatbot */}
                    <ChatBot videoId={videoId} />
                </div>
            </main>
        </div>
    );
}

export default AnalysisPage;
