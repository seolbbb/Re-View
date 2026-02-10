import { useState, useRef, useEffect, useCallback } from 'react';
import { Sparkles, RotateCcw, Bot, Send, Loader2, Zap, Clock, ChevronDown, GripVertical } from 'lucide-react';
import { streamChatMessage } from '../api/chat';
import { useVideo } from '../context/VideoContext';
import MarkdownRenderer from './MarkdownRenderer';

const MIN_WIDTH = 280;
const MAX_WIDTH = 600;
const DEFAULT_WIDTH = 360;

function ChatBot({ videoId, isOpen, onToggle, prefillData, onPrefillClear }) {
    const { chatSessionId, setChatSessionId } = useVideo();
    const [width, setWidth] = useState(DEFAULT_WIDTH);
    const [isResizing, setIsResizing] = useState(false);
    const resizeRef = useRef(null);
    const [messages, setMessages] = useState([
        {
            id: 'welcome',
            type: 'bot',
            text: '영상에 대해 궁금한 점을 물어보세요!',
            timestamp: 'Now',
        },
    ]);
    const [inputValue, setInputValue] = useState('');
    const [sending, setSending] = useState(false);
    const [reasoningMode, setReasoningMode] = useState('flash'); // 'flash' or 'thinking'
    const [modeMenuOpen, setModeMenuOpen] = useState(false);
    const messagesContainerRef = useRef(null);
    const abortRef = useRef(null);
    const modeMenuRef = useRef(null);
    const textareaRef = useRef(null);
    const messageCounterRef = useRef(0);
    const segmentContextRef = useRef(null);

    const nextMessageId = (prefix) => {
        messageCounterRef.current += 1;
        return `${prefix}-${messageCounterRef.current}`;
    };

    const adjustTextareaHeight = () => {
        const textarea = textareaRef.current;
        if (textarea) {
            textarea.style.height = 'auto';
            const maxHeight = 140;
            const newHeight = Math.min(textarea.scrollHeight, maxHeight);
            textarea.style.height = `${newHeight}px`;
            textarea.style.overflowY = textarea.scrollHeight > maxHeight ? 'auto' : 'hidden';
        }
    };

    useEffect(() => {
        const handleClickOutside = (e) => {
            if (modeMenuRef.current && !modeMenuRef.current.contains(e.target)) {
                setModeMenuOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    const handleMouseDown = useCallback((e) => {
        e.preventDefault();
        setIsResizing(true);
        resizeRef.current = {
            startX: e.clientX,
            startWidth: width,
        };
    }, [width]);

    useEffect(() => {
        const handleMouseMove = (e) => {
            if (!isResizing || !resizeRef.current) return;
            const delta = resizeRef.current.startX - e.clientX;
            const newWidth = Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, resizeRef.current.startWidth + delta));
            setWidth(newWidth);
        };

        const handleMouseUp = () => {
            setIsResizing(false);
            resizeRef.current = null;
        };

        if (isResizing) {
            document.addEventListener('mousemove', handleMouseMove);
            document.addEventListener('mouseup', handleMouseUp);
            document.body.style.cursor = 'col-resize';
            document.body.style.userSelect = 'none';
        }

        return () => {
            document.removeEventListener('mousemove', handleMouseMove);
            document.removeEventListener('mouseup', handleMouseUp);
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
        };
    }, [isResizing]);

    useEffect(() => {
        const container = messagesContainerRef.current;
        if (container) {
            container.scrollTop = container.scrollHeight;
        }
    }, [messages]);

    // Handle prefill from SummaryPanel Bot button
    useEffect(() => {
        if (prefillData) {
            const { segIdx, timeRange, content } = prefillData;
            setInputValue(`[Seg${segIdx}] `);
            segmentContextRef.current = { segIdx, timeRange, content };
            onPrefillClear?.();
            // Focus textarea after state update
            setTimeout(() => {
                const ta = textareaRef.current;
                if (ta) {
                    ta.focus();
                    const len = ta.value.length;
                    ta.setSelectionRange(len, len);
                }
            }, 0);
        }
    }, [prefillData]);

    useEffect(() => {
        return () => {
            abortRef.current?.abort();
        };
    }, []);

    const handleSend = async (presetText = null, modeOverride = null) => {
        const text = (typeof presetText === 'string' ? presetText : inputValue).trim();
        if (!text || sending) return;

        // Build the actual message for the API, including segment context if present
        const segCtx = segmentContextRef.current;
        let apiMessage = text;
        let effectiveMode = modeOverride || reasoningMode;
        if (segCtx) {
            const bullets = segCtx.content?.bullets || [];
            const bulletText = bullets.map((b) => {
                if (typeof b === 'string') return b;
                const prefix = b.bullet_id ? `(${b.bullet_id}) ` : '';
                return prefix + (b.claim || b.text || JSON.stringify(b));
            }).join('\n');
            apiMessage = `[세그먼트 ${segCtx.segIdx} (${segCtx.timeRange}) 요약:\n${bulletText}]\n\n${text}`;
            effectiveMode = 'thinking';
            segmentContextRef.current = null;
        }

        const botId = nextMessageId('bot');
        const userMsg = {
            id: nextMessageId('user'),
            type: 'user',
            text,
            timestamp: 'Now',
        };
        const botMsg = {
            id: botId,
            type: 'bot',
            text: '',
            timestamp: 'Now',
            streaming: true,
            suggestions: [],
            messageId: null,
        };
        setMessages((prev) => [...prev, userMsg, botMsg]);
        setInputValue('');
        setSending(true);
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
        }

        abortRef.current = streamChatMessage({
            videoId: videoId || '',
            message: apiMessage,
            sessionId: chatSessionId,
            reasoningMode: effectiveMode,
            onSessionId: (sessionId) => {
                if (sessionId) setChatSessionId(sessionId);
            },
            onChunk: (chunk, isFinal, payload) => {
                if (!chunk && !isFinal) return;
                setMessages((prev) =>
                    prev.map((msg) => {
                        if (msg.id !== botId) return msg;
                        const nextText = `${msg.text || ''}${chunk || ''}`;
                        return {
                            ...msg,
                            text: nextText,
                            streaming: !isFinal,
                            messageId: payload?.message_id || msg.messageId || null,
                        };
                    })
                );
                if (isFinal) {
                    setSending(false);
                }
            },
            onSuggestions: (payload) => {
                const questions = Array.isArray(payload?.questions)
                    ? payload.questions
                        .map((item) => String(item || '').trim())
                        .filter(Boolean)
                        .slice(0, 2)
                    : [];
                if (!questions.length) return;
                setMessages((prev) =>
                    prev.map((msg) =>
                        msg.id === botId
                            ? {
                                ...msg,
                                suggestions: questions,
                                messageId: payload?.message_id || msg.messageId || null,
                            }
                            : msg
                    )
                );
            },
            onDone: () => {
                setMessages((prev) =>
                    prev.map((msg) =>
                        msg.id === botId ? { ...msg, streaming: false } : msg
                    )
                );
                setSending(false);
                abortRef.current = null;
            },
            onError: (err) => {
                setMessages((prev) => {
                    const next = prev.map((msg) =>
                        msg.id === botId ? { ...msg, streaming: false } : msg
                    );
                    next.push({
                        id: nextMessageId('err'),
                        type: 'bot',
                        text: `오류가 발생했습니다: ${err.message || 'Unknown error'}`,
                        timestamp: 'Now',
                    });
                    return next;
                });
                setSending(false);
                abortRef.current = null;
            },
        });
    };

    const handleSuggestionClick = (question) => {
        if (!question || sending) return;
        handleSend(question);
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const handleClear = () => {
        abortRef.current?.abort();
        abortRef.current = null;
        setSending(false);
        setMessages([
            {
                id: 'welcome',
                type: 'bot',
                text: '영상에 대해 궁금한 점을 물어보세요!',
                timestamp: 'Now',
            },
        ]);
        setChatSessionId(null);
    };

    return (
        <>
            <aside
                className={`fixed lg:relative z-40 transition-all duration-300 ease-in-out pointer-events-none lg:pointer-events-auto
                    /* Mobile: Bottom Sheet 45% height */
                    inset-x-0 bottom-0 h-[45dvh] w-full pt-4 px-2 pb-0
                    /* Desktop: Right Column (Sliding Sidebar Implementation) */
                    lg:inset-auto lg:h-full lg:py-4
                    ${isOpen
                        ? 'translate-y-0 lg:translate-x-0 lg:w-[var(--chatbot-width)] lg:px-4 lg:opacity-100 lg:visible'
                        : 'translate-y-full lg:translate-x-full lg:w-0 lg:px-0 lg:opacity-0 lg:invisible lg:overflow-hidden'
                    }
                    flex flex-col`}
                style={{ '--chatbot-width': `${width}px`, maxWidth: '100%' }}
            >


                {/* Floating Card Content - Flush at bottom, curved at top */}
                <div className="flex flex-col h-full bg-white dark:bg-zinc-800 border-x border-t border-black/10 dark:border-white/20 backdrop-blur-md rounded-t-2xl lg:rounded-none shadow-[0_-15px_50px_rgba(0,0,0,0.2)] relative pointer-events-auto">

                    {/* Header Controls (Mobile/Tablet Only) */}
                    <div className="flex-none flex lg:hidden items-center justify-between h-6 px-4 mt-4 mb-2 z-50">
                        {/* Refresh Button - Top Left */}
                        <button
                            onClick={handleClear}
                            className="w-9 h-9 flex items-center justify-center rounded-full text-zinc-400 hover:text-primary hover:bg-surface-highlight transition-all"
                            title="Refresh Chat"
                        >
                            <RotateCcw size={20} />
                        </button>

                        {/* Close Button - Top Right */}
                        <button
                            onClick={onToggle}
                            className="w-9 h-9 flex items-center justify-center rounded-full text-zinc-400 hover:text-primary hover:bg-surface-highlight transition-all"
                            title="Close Chat"
                        >
                            <ChevronDown size={20} />
                        </button>
                    </div>

                    {/* Messages Container */}
                    <div ref={messagesContainerRef} className="flex-1 overflow-y-auto px-4 py-4 flex flex-col gap-4 custom-scrollbar bg-transparent lg:bg-white dark:bg-zinc-800 rounded-t-2xl lg:rounded-none overflow-hidden">
                        <div className="flex-1" /> {/* Spacer to push content down */}
                        {messages.map((msg) => (
                            <div key={msg.id} className={`flex gap-3 ${msg.type === 'user' ? 'flex-row-reverse' : ''}`}>
                                {msg.type === 'bot' ? (
                                    <div className="w-9 h-9 rounded-full bg-primary/20 flex items-center justify-center shrink-0">
                                        <Bot size={20} className="text-primary" />
                                    </div>
                                ) : (
                                    <div className="w-9 h-9 rounded-full bg-gray-700 flex items-center justify-center shrink-0 text-white text-xs font-bold">
                                        U
                                    </div>
                                )}

                                <div className={`flex flex-col gap-1 ${msg.type === 'user' ? 'items-end' : ''} max-w-[85%]`}>
                                    <div className={`${msg.type === 'bot'
                                        ? 'bg-surface-highlight rounded-tl-none border border-[var(--border-color)] text-[var(--text-primary)]'
                                        : 'bg-surface-highlight rounded-tr-none text-[var(--text-primary)]'} p-3 rounded-2xl text-[13px] leading-snug shadow-sm`}>
                                        {msg.type === 'bot' ? (
                                            <div className="chat-markdown overflow-hidden">
                                                <MarkdownRenderer>{msg.text}</MarkdownRenderer>
                                                {msg.streaming && (
                                                    msg.text ? (
                                                        <span className="inline-block w-2 h-4 bg-primary/70 animate-pulse align-middle ml-1 rounded-sm" />
                                                    ) : (
                                                        <Loader2 size={20} className="animate-spin text-primary inline-block" />
                                                    )
                                                )}
                                            </div>
                                        ) : (
                                            <p className="whitespace-pre-wrap">{msg.text}</p>
                                        )}
                                    </div>
                                    {msg.type === 'bot' && !msg.streaming && Array.isArray(msg.suggestions) && msg.suggestions.length > 0 && (
                                        <div className="flex flex-wrap gap-2 pl-1">
                                            {msg.suggestions.map((question, index) => (
                                                <button
                                                    key={`${msg.id}-suggestion-${index}`}
                                                    type="button"
                                                    onClick={() => handleSuggestionClick(question)}
                                                    disabled={sending}
                                                    className="px-3.5 py-2 rounded-lg border-2 border-[var(--chip-suggestion-border)] bg-[var(--chip-suggestion-bg)] text-[13px] leading-snug text-[var(--chip-suggestion-text)] font-medium shadow-sm hover:bg-[var(--chip-suggestion-hover-bg)] hover:border-[var(--chip-suggestion-hover-border)] transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                                                >
                                                    {question}
                                                </button>
                                            ))}
                                        </div>
                                    )}
                                    <span className={`text-[10px] text-zinc-500 dark:text-zinc-400 font-medium ${msg.type === 'user' ? 'pr-1' : 'pl-1'}`}>{msg.timestamp}</span>
                                </div>
                            </div>
                        ))}
                    </div>

                    {/* Input Area (Internal, Flex-none) */}
                    <div className="flex-none px-4 pt-3 pb-6 bg-transparent lg:bg-[#fafafa] dark:bg-zinc-800 lg:border-t lg:border-[var(--border-color)]">
                        <div className="bg-white dark:bg-zinc-700 border-t border-white/80 dark:border-white/10 rounded-full shadow-[0_10px_30px_-5px_rgba(0,0,0,0.1),0_0_1px_1px_rgba(0,0,0,0.05)] flex items-center px-1 h-9 gap-2 relative z-50 ring-1 ring-black/5 dark:ring-white/5">

                            {/* Mode Selector (Left) */}
                            <div className="relative" ref={modeMenuRef}>
                                <button
                                    onClick={() => setModeMenuOpen(!modeMenuOpen)}
                                    disabled={sending}
                                    className="flex items-center justify-center w-9 h-9 rounded-full hover:bg-surface-highlight transition-colors text-[var(--text-secondary)]"
                                    title={reasoningMode === 'flash' ? '요약 기반 답변' : '원본 분석 답변'}
                                >
                                    {reasoningMode === 'flash' ? (
                                        <Zap size={20} className="text-yellow-500" />
                                    ) : (
                                        <Clock size={20} className="text-blue-500" />
                                    )}
                                </button>
                                {/* Dropdown Menu (opens upward, left-aligned) */}
                                {modeMenuOpen && (
                                    <div className="absolute bottom-full left-0 mb-2 w-64 bg-white dark:bg-zinc-800 border border-[var(--border-color)] rounded-none shadow-[0_10px_40px_rgba(0,0,0,0.2)] overflow-hidden z-50 animate-fade-in origin-bottom-left">
                                        <button
                                            onClick={() => {
                                                setReasoningMode('flash');
                                                setModeMenuOpen(false);
                                            }}
                                            className={`w-full flex items-start gap-3 px-4 py-4 text-left hover:bg-surface-highlight transition-colors ${reasoningMode === 'flash' ? 'bg-primary/5' : ''}`}
                                        >
                                            <div className={`p-2 rounded-xl ${reasoningMode === 'flash' ? 'bg-yellow-500/20' : 'bg-gray-100 dark:bg-zinc-700'}`}>
                                                <Zap size={20} className={`${reasoningMode === 'flash' ? 'text-yellow-500' : 'text-gray-400'}`} />
                                            </div>
                                            <div className="flex-1">
                                                <div className={`text-sm font-bold ${reasoningMode === 'flash' ? 'text-primary' : 'text-[var(--text-primary)]'}`}>Flash Mode</div>
                                                <div className="text-[11px] text-gray-500 mt-0.5 leading-tight">빠른 요약 데이터를 기반으로 즉각 답변합니다.</div>
                                            </div>
                                        </button>
                                        <button
                                            onClick={() => {
                                                setReasoningMode('thinking');
                                                setModeMenuOpen(false);
                                            }}
                                            className={`w-full flex items-start gap-3 px-4 py-4 text-left hover:bg-surface-highlight transition-colors ${reasoningMode === 'thinking' ? 'bg-primary/5' : ''}`}
                                        >
                                            <div className={`p-2 rounded-xl ${reasoningMode === 'thinking' ? 'bg-blue-500/20' : 'bg-gray-100 dark:bg-zinc-700'}`}>
                                                <Clock size={20} className={`${reasoningMode === 'thinking' ? 'text-blue-500' : 'text-gray-400'}`} />
                                            </div>
                                            <div className="flex-1">
                                                <div className={`text-sm font-bold ${reasoningMode === 'thinking' ? 'text-primary' : 'text-[var(--text-primary)]'}`}>Thinking Mode</div>
                                                <div className="text-[11px] text-gray-500 mt-0.5 leading-tight">원본 데이터 전체를 분석하여 깊이 있는 답변을 제공합니다.</div>
                                            </div>
                                        </button>
                                    </div>
                                )}
                            </div>

                            {/* Textarea (Center) */}
                            <textarea
                                ref={textareaRef}
                                className="flex-1 bg-transparent border-0 px-0 py-3 text-sm text-[var(--text-primary)] focus:outline-none focus:ring-0 focus:border-0 focus-visible:ring-0 focus-visible:outline-none placeholder:text-gray-400 resize-none max-h-[140px] appearance-none !ring-0 !outline-none"
                                placeholder="Ask Re:View..."
                                rows={1}
                                value={inputValue}
                                onChange={(e) => {
                                    setInputValue(e.target.value);
                                    adjustTextareaHeight();
                                }}
                                onKeyDown={handleKeyDown}
                                disabled={sending}
                                style={{ minHeight: '24px', maxHeight: '140px', boxShadow: 'none', border: 'none', outline: 'none' }}
                            />

                            {/* Send Button (Right) */}
                            <div className={inputValue.trim() ? "" : "cursor-not-allowed"}>
                                <button
                                    onClick={handleSend}
                                    disabled={sending || !inputValue.trim()}
                                    className={`w-9 h-9 rounded-full flex items-center justify-center transition-all shadow-sm ${inputValue.trim()
                                        ? 'bg-[var(--accent-coral)] text-white hover:bg-[var(--accent-coral-dark)] shadow-[0_2px_8px_rgba(224,126,99,0.3)]'
                                        : 'bg-surface-highlight text-gray-400'
                                        }`}
                                >
                                    <Send size={20} />
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </aside>

            {/* External Open Button (Mobile Only) */}
            {!isOpen && (
                <button
                    onClick={onToggle}
                    className="lg:hidden fixed bottom-6 right-6 z-50 w-9 h-9 flex items-center justify-center rounded-xl shadow-2xl transition-all duration-300 bg-[var(--bg-secondary)] text-[var(--accent-coral)] border border-[var(--border-color)] hover:scale-110 active:scale-95 group"
                    title="Open AI Chat"
                >
                    <Bot size={20} />
                </button>
            )}
        </>
    );
}

export default ChatBot;
