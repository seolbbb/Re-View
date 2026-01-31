import { useState, useRef, useEffect } from 'react';
import { Sparkles, RotateCcw, Bot, Send, Loader2, Zap, Clock, ChevronDown } from 'lucide-react';
import { streamChatMessage } from '../api/chat';
import { useVideo } from '../context/VideoContext';
import MarkdownRenderer from './MarkdownRenderer';

function ChatBot({ videoId }) {
    const { chatSessionId, setChatSessionId } = useVideo();
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

    // Auto-resize textarea
    const adjustTextareaHeight = () => {
        const textarea = textareaRef.current;
        if (textarea) {
            textarea.style.height = 'auto';
            const maxHeight = 140; // ~5 lines
            const newHeight = Math.min(textarea.scrollHeight, maxHeight);
            textarea.style.height = `${newHeight}px`;
            // Only show scrollbar when content exceeds maxHeight
            textarea.style.overflowY = textarea.scrollHeight > maxHeight ? 'auto' : 'hidden';
        }
    };

    // Close menu when clicking outside
    useEffect(() => {
        const handleClickOutside = (e) => {
            if (modeMenuRef.current && !modeMenuRef.current.contains(e.target)) {
                setModeMenuOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    useEffect(() => {
        const container = messagesContainerRef.current;
        if (container) {
            container.scrollTop = container.scrollHeight;
        }
    }, [messages]);

    useEffect(() => {
        return () => {
            abortRef.current?.abort();
        };
    }, []);

    const handleSend = async () => {
        const text = inputValue.trim();
        if (!text || sending) return;

        const botId = `bot-${Date.now()}`;
        const userMsg = {
            id: `user-${Date.now()}`,
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
        };
        setMessages((prev) => [...prev, userMsg, botMsg]);
        setInputValue('');
        setSending(true);
        // Reset textarea height
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
        }

        abortRef.current = streamChatMessage({
            videoId: videoId || '',
            message: text,
            sessionId: chatSessionId,
            reasoningMode: reasoningMode,
            onSessionId: (sessionId) => {
                if (sessionId) setChatSessionId(sessionId);
            },
            onChunk: (chunk, isFinal) => {
                if (!chunk && !isFinal) return;
                setMessages((prev) =>
                    prev.map((msg) => {
                        if (msg.id !== botId) return msg;
                        const nextText = `${msg.text || ''}${chunk || ''}`;
                        return {
                            ...msg,
                            text: nextText,
                            streaming: !isFinal,
                        };
                    })
                );
                if (isFinal) {
                    setSending(false);
                }
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
                        id: `err-${Date.now()}`,
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
        <aside className="w-[360px] hidden lg:flex flex-col border-l border-[var(--border-color)] bg-surface shrink-0 h-full overflow-hidden">
            {/* Chat Header */}
            <div className="p-4 border-b border-[var(--border-color)] bg-surface z-10 shadow-sm flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <Sparkles className="w-5 h-5 text-primary" />
                    <h3 className="text-[var(--text-primary)] font-medium text-sm">Ask Re:View AI</h3>
                </div>
                <button
                    onClick={handleClear}
                    className="text-gray-400 hover:text-[var(--text-primary)]"
                    title="Clear History"
                >
                    <RotateCcw className="w-[18px] h-[18px]" />
                </button>
            </div>

            {/* Messages Container */}
            <div ref={messagesContainerRef} className="flex-1 overflow-y-auto p-4 flex flex-col gap-4 custom-scrollbar">
                {messages.map((msg) => (
                    <div key={msg.id} className={`flex gap-3 ${msg.type === 'user' ? 'flex-row-reverse' : ''}`}>
                        {msg.type === 'bot' ? (
                            <div className="size-8 rounded-full bg-primary/20 flex items-center justify-center shrink-0 border border-primary/20">
                                <Bot className="w-4 h-4 text-primary" />
                            </div>
                        ) : (
                            <div className="size-8 rounded-full bg-gray-700 flex items-center justify-center shrink-0 text-white text-xs font-bold">
                                U
                            </div>
                        )}

                        <div className={`flex flex-col gap-1 ${msg.type === 'user' ? 'items-end' : ''} max-w-[85%]`}>
                            <div className={`${msg.type === 'bot' ? 'bg-surface-highlight rounded-tl-none border border-[var(--border-color)] text-[var(--text-secondary)]' : 'bg-primary rounded-tr-none text-white'} p-3 rounded-2xl text-sm leading-relaxed shadow-sm`}>
                                {msg.type === 'bot' ? (
                                    <div className="chat-markdown overflow-hidden">
                                        <MarkdownRenderer>{msg.text}</MarkdownRenderer>
                                        {msg.streaming && (
                                            msg.text ? (
                                                <span className="inline-block w-2 h-4 bg-primary/70 animate-pulse align-middle ml-1 rounded-sm" />
                                            ) : (
                                                <Loader2 className="w-4 h-4 animate-spin text-primary inline-block" />
                                            )
                                        )}
                                    </div>
                                ) : (
                                    <p className="whitespace-pre-wrap">{msg.text}</p>
                                )}
                            </div>
                            <span className={`text-[10px] text-gray-500 ${msg.type === 'user' ? 'pr-1' : 'pl-1'}`}>{msg.timestamp}</span>
                        </div>
                    </div>
                ))}
            </div>

            {/* Input Area */}
            <div className="p-4 border-t border-[var(--border-color)] bg-surface/95 backdrop-blur">
                <div className="bg-surface-highlight rounded-xl">
                    {/* Textarea */}
                    <textarea
                        ref={textareaRef}
                        className="w-full bg-transparent border-0 px-4 pt-3 pb-0 text-sm text-[var(--text-primary)] focus:outline-none focus:ring-0 placeholder:text-gray-500 resize-none [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-gray-400/50 [&::-webkit-scrollbar-thumb]:rounded-full"
                        placeholder="Ask a question..."
                        rows={1}
                        value={inputValue}
                        onChange={(e) => {
                            setInputValue(e.target.value);
                            adjustTextareaHeight();
                        }}
                        onKeyDown={handleKeyDown}
                        disabled={sending}
                        style={{ minHeight: '24px', maxHeight: '140px', overflowY: 'hidden' }}
                    />
                    {/* Bottom row: Mode selector + Send button */}
                    <div className="flex items-center justify-between px-2 pb-2">
                        {/* Mode Selector */}
                        <div className="relative" ref={modeMenuRef}>
                            <button
                                onClick={() => setModeMenuOpen(!modeMenuOpen)}
                                disabled={sending}
                                className="flex items-center gap-1.5 px-2 py-1.5 rounded-lg text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-surface transition-all disabled:opacity-50 text-xs"
                                title={reasoningMode === 'flash' ? '요약 기반 답변' : '원본 분석 답변'}
                            >
                                {reasoningMode === 'flash' ? (
                                    <Zap className="w-3.5 h-3.5 text-yellow-500" />
                                ) : (
                                    <Clock className="w-3.5 h-3.5 text-blue-500" />
                                )}
                                <span className="font-medium">{reasoningMode === 'flash' ? 'Flash' : 'Thinking'}</span>
                                <ChevronDown className={`w-3 h-3 transition-transform ${modeMenuOpen ? 'rotate-180' : ''}`} />
                            </button>
                            {/* Dropdown Menu (opens upward) */}
                            {modeMenuOpen && (
                                <div className="absolute bottom-full left-0 mb-2 w-52 bg-surface border border-[var(--border-color)] rounded-xl shadow-xl overflow-hidden z-50">
                                    <button
                                        onClick={() => {
                                            setReasoningMode('flash');
                                            setModeMenuOpen(false);
                                        }}
                                        className={`w-full flex items-start gap-3 px-3 py-3 text-left hover:bg-surface-highlight transition-colors ${
                                            reasoningMode === 'flash' ? 'bg-primary/10' : ''
                                        }`}
                                    >
                                        <Zap className={`w-4 h-4 mt-0.5 shrink-0 ${reasoningMode === 'flash' ? 'text-yellow-500' : 'text-gray-400'}`} />
                                        <div>
                                            <div className={`text-sm font-medium ${reasoningMode === 'flash' ? 'text-[var(--text-primary)]' : 'text-[var(--text-secondary)]'}`}>
                                                Flash
                                            </div>
                                            <div className="text-[11px] text-gray-500 mt-0.5">
                                                요약 기반 빠른 답변
                                            </div>
                                        </div>
                                    </button>
                                    <button
                                        onClick={() => {
                                            setReasoningMode('thinking');
                                            setModeMenuOpen(false);
                                        }}
                                        className={`w-full flex items-start gap-3 px-3 py-3 text-left hover:bg-surface-highlight transition-colors ${
                                            reasoningMode === 'thinking' ? 'bg-primary/10' : ''
                                        }`}
                                    >
                                        <Clock className={`w-4 h-4 mt-0.5 shrink-0 ${reasoningMode === 'thinking' ? 'text-blue-500' : 'text-gray-400'}`} />
                                        <div>
                                            <div className={`text-sm font-medium ${reasoningMode === 'thinking' ? 'text-[var(--text-primary)]' : 'text-[var(--text-secondary)]'}`}>
                                                Thinking
                                            </div>
                                            <div className="text-[11px] text-gray-500 mt-0.5">
                                                원본 분석 심층 답변
                                            </div>
                                        </div>
                                    </button>
                                </div>
                            )}
                        </div>
                        {/* Send Button */}
                        <button
                            onClick={handleSend}
                            disabled={sending || !inputValue.trim()}
                            className="p-2 bg-primary hover:bg-[var(--accent-coral-dark)] rounded-lg text-white transition-colors shadow-sm disabled:opacity-50"
                        >
                            <Send className="w-4 h-4" />
                        </button>
                    </div>
                </div>
                <p className="text-[10px] text-center text-gray-600 mt-2">AI can make mistakes. Check important info.</p>
            </div>
        </aside>
    );
}

export default ChatBot;
