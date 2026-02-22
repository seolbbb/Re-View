import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
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
import { Menu, ChevronRight, Sun, Moon, Download, ChevronDown, Video, Bot, Library } from 'lucide-react';

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
    const [sidebarOpen, setSidebarOpen] = useState(false);
    const [chatbotOpen, setChatbotOpen] = useState(typeof window !== 'undefined' ? window.innerWidth >= 1024 : false);
    const [chatPrefill, setChatPrefill] = useState(null);
    const exportRef = useRef(null);

    // Close export dropdown on outside click
    useEffect(() => {
        const handleClick = (e) => {
            if (exportRef.current && !exportRef.current.contains(e.target)) {
                setExportOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClick);

        // Auto-open chatbot on desktop resize
        const handleResize = () => {
            if (window.innerWidth >= 1024) {
                setChatbotOpen(true);
            }
        };
        window.addEventListener('resize', handleResize);

        return () => {
            document.removeEventListener('mousedown', handleClick);
            window.removeEventListener('resize', handleResize);
        };
    }, []);

    // Body scroll lock for mobile overlays
    useEffect(() => {
        if (sidebarOpen || chatbotOpen) {
            document.body.classList.add('no-scroll');
        } else {
            document.body.classList.remove('no-scroll');
        }
        return () => document.body.classList.remove('no-scroll');
    }, [sidebarOpen, chatbotOpen]);

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
    // Scroll-triggered PiP
    const [scrollPip, setScrollPip] = useState(false);
    const videoSectionRef = useRef(null);
    const scrollContainerRef = useRef(null);
    const scrollPipRef = useRef(false);

    const handleScrollPipClose = useCallback(() => {
        videoSectionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, []);

    const handleAskChatBot = useCallback((data) => {
        setChatPrefill(data);
        setChatbotOpen(true);
    }, []);

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
        const section = videoSectionRef.current;
        const container = scrollContainerRef.current;
        if (!section || !container) return;

        const observer = new IntersectionObserver(
            ([entry]) => {
                // If even 10% of the video section is visible, we should show the non-PIP video
                // This prevents the PIP from staying when the original spot is beginning to show
                const shouldPip = !entry.isIntersecting;
                if (shouldPip === scrollPipRef.current) return;
                scrollPipRef.current = shouldPip;
                setScrollPip(shouldPip);
            },
            { root: container, threshold: 0.1 },
        );

        observer.observe(section);
        return () => observer.disconnect();
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
                    v.play().catch(() => { });
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
        <div className="bg-[var(--bg-primary)] text-[var(--text-primary)] font-display flex h-[100dvh] overflow-hidden selection:bg-primary/40 selection:text-white transition-colors duration-300 relative" data-theme={theme}>
            {/* Left Sidebar */}
            <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />

            {/* Main Content Area */}
            <main className="flex-1 flex flex-col min-w-0 h-full relative bg-[var(--bg-primary)] transition-colors duration-300">
                {/* Header / Breadcrumbs */}
                <header className="h-[50px] flex items-center justify-between px-4 lg:px-8 border-b border-[var(--border-color)] bg-[var(--bg-primary)]/95 backdrop-blur z-30 shrink-0 shadow-sm">
                    <div className="flex items-center gap-4 min-w-0 h-full">
                        <button
                            onClick={() => setSidebarOpen(true)}
                            className="lg:hidden w-12 h-[50px] flex items-center justify-center text-gray-400 hover:text-primary transition-colors shrink-0 bg-surface-highlight/5 border-r border-[var(--border-color)]"
                        >
                            <Menu className="w-5 h-5" />
                        </button>

                        <div className="w-1 shrink-0" />
                        <div className="flex items-center gap-3.5 min-w-0 overflow-hidden h-[50px] pl-6 pr-4 bg-surface-highlight/5 border-b border-[var(--border-color)]">
                            <div className="flex items-center gap-3 min-w-0">
                                <span className={`w-2.5 h-2.5 rounded-full shrink-0 ml-4 shadow-sm ${statusLabel === 'DONE' ? 'bg-emerald-500 shadow-emerald-500/50' : 'bg-primary animate-pulse shadow-primary/50'}`} />
                                {videoInfo ? (
                                    <h1 className="text-base font-bold leading-none tracking-tight text-[var(--text-primary)] truncate">{videoName}</h1>
                                ) : (
                                    <div className="flex gap-1.5">
                                        {[0, 1, 2].map(i => (
                                            <span key={i} className="w-1.5 h-1.5 rounded-full bg-gray-600 animate-pulse" style={{ animationDelay: `${i * 0.2}s` }} />
                                        ))}
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>

                    <div className="flex items-center gap-1.5 shrink-0 h-full ml-4">
                        <button
                            onClick={() => setChatbotOpen((prev) => !prev)}
                            className={`lg:hidden w-9 h-9 flex items-center justify-center transition-colors rounded-xl border border-[var(--border-color)] shadow-sm ${chatbotOpen
                                ? 'text-[var(--accent-coral)] bg-[var(--accent-coral)]/20'
                                : 'text-gray-400 hover:text-[var(--accent-coral)] bg-surface-highlight/5'
                                }`}
                            title={chatbotOpen ? "Close AI Chat" : "Open AI Chat"}
                        >
                            <Bot className="w-5 h-5" />
                        </button>

                        <button
                            onClick={toggleTheme}
                            className="w-9 h-9 flex items-center justify-center text-gray-400 hover:text-[var(--text-primary)] transition-colors bg-surface-highlight/5 border border-[var(--border-color)] rounded-xl shadow-sm"
                            title={theme === 'dark' ? "Switch to Light Mode" : "Switch to Dark Mode"}
                        >
                            {theme === 'dark' ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
                        </button>
                        <div className="relative h-9" ref={exportRef}>
                            <button
                                onClick={() => setExportOpen((o) => !o)}
                                className="w-9 h-9 flex items-center justify-center text-gray-400 hover:text-[var(--text-primary)] transition-colors bg-surface-highlight/5 border border-[var(--border-color)] rounded-xl shadow-sm"
                                title="Export Video Summary"
                            >
                                <Download className="w-5 h-5" />
                            </button>
                            {exportOpen && (
                                <div className="absolute right-0 top-full mt-2 w-48 bg-[var(--bg-secondary,var(--bg-primary))] border border-[var(--border-color)] rounded-xl shadow-xl z-50 overflow-hidden animate-fade-in">
                                    <button
                                        onClick={handleExportMarkdown}
                                        className="w-full text-center px-4 py-3 text-sm text-[var(--text-primary)] hover:bg-surface-highlight transition-colors"
                                    >
                                        Markdown (.md)
                                    </button>
                                </div>
                            )}
                        </div>
                    </div>
                </header>

                {/* Workspace Grid */}
                <div className="flex flex-col lg:flex-row flex-1 overflow-hidden relative">
                    {/* Left Column: Video & Summary - Unified Scroll Content */}
                    <div className="relative min-w-0 min-h-0 bg-[var(--bg-primary)] transition-all duration-300 ease-in-out flex-1 h-full">
                        <div ref={scrollContainerRef} className="w-full h-full overflow-y-auto custom-scrollbar flex flex-col items-center">
                            {/* 1. Video Section - Always Constrained at Top */}
                            <div ref={videoSectionRef} className="w-full max-w-[1024px] px-5 sm:px-8 mx-auto shrink-0 pt-6">
                                <div
                                    className={scrollPip
                                        ? `fixed z-50 aspect-video animate-pipSlideIn transition-all duration-300 hover:scale-[1.02] right-6 ${chatbotOpen ? 'lg:right-[calc(1.5rem+var(--chatbot-width))]' : 'lg:right-6'} top-[58px] w-80 md:w-96 lg:top-auto lg:bottom-6 lg:w-96 rounded-xl overflow-hidden border border-[var(--border-color)] shadow-2xl`
                                        : "relative w-full aspect-video border-b border-[var(--border-color)] bg-black shadow-sm"}
                                >
                                    <VideoPlayer
                                        isPip={scrollPip}
                                        onTogglePip={scrollPip ? handleScrollPipClose : undefined}
                                        videoId={videoId}
                                        videoElRef={videoElRef}
                                        playbackRestore={playbackStateRef}
                                        onTimeUpdate={handleTimeUpdate}
                                        className={scrollPip
                                            ? "relative w-full h-full bg-black group"
                                            : "relative w-full h-full bg-black"}
                                    />
                                </div>
                                {scrollPip && <div style={{ aspectRatio: '16/9' }} className="rounded-none bg-black/5" />}
                            </div>

                            {/* 2. Summary Section - Width depends on Expanded state */}
                            <div className="w-full mx-auto transition-all duration-300 max-w-[1024px] px-5 sm:px-8">
                                <SummaryPanel
                                    isExpanded={isExpanded}
                                    onToggleExpand={() => handleToggleExpand(!isExpanded)}
                                    videoId={videoId}
                                    onSeekTo={handleSeekTo}
                                    currentTimeMs={currentPlaybackMs}
                                    chatbotOpen={chatbotOpen}
                                    onAskChatBot={handleAskChatBot}
                                />
                            </div>
                        </div>
                    </div>

                    {/* Right Column: AI Chatbot */}
                    <ChatBot
                        videoId={videoId}
                        isOpen={chatbotOpen}
                        onToggle={() => setChatbotOpen((prev) => !prev)}
                        prefillData={chatPrefill}
                        onPrefillClear={() => setChatPrefill(null)}
                    />
                </div>
            </main>
        </div>
    );
}

export default AnalysisPage;
