import { useState, useRef, useEffect } from 'react';
import { Sparkles, RotateCcw, Bot, Send, Loader2 } from 'lucide-react';
import { streamChatMessage } from '../api/chat';
import { useVideo } from '../context/VideoContext';

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
    const messagesEndRef = useRef(null);
    const abortRef = useRef(null);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
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

        abortRef.current = streamChatMessage({
            videoId: videoId || '',
            message: text,
            sessionId: chatSessionId,
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
        <aside className="w-[360px] hidden lg:flex flex-col border-l border-[var(--border-color)] bg-surface shrink-0 h-full">
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
            <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-4 custom-scrollbar">
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
                                <p className="whitespace-pre-wrap">
                                    {msg.text}
                                    {msg.streaming && (
                                        msg.text ? (
                                            <span className="inline-block w-2 h-4 bg-primary/70 animate-pulse align-middle ml-1 rounded-sm" />
                                        ) : (
                                            <Loader2 className="w-4 h-4 animate-spin text-primary inline-block" />
                                        )
                                    )}
                                </p>
                            </div>
                            <span className={`text-[10px] text-gray-500 ${msg.type === 'user' ? 'pr-1' : 'pl-1'}`}>{msg.timestamp}</span>
                        </div>
                    </div>
                ))}
                <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <div className="p-4 border-t border-[var(--border-color)] bg-surface/95 backdrop-blur">
                <div className="relative flex items-center">
                    <input
                        className="w-full bg-surface-highlight border border-[var(--border-color)] rounded-xl pl-4 pr-12 py-3 text-sm text-[var(--text-primary)] focus:outline-none focus:ring-1 focus:ring-primary focus:border-primary transition-all placeholder:text-gray-500"
                        placeholder="Ask a question..."
                        type="text"
                        value={inputValue}
                        onChange={(e) => setInputValue(e.target.value)}
                        onKeyDown={handleKeyDown}
                        disabled={sending}
                    />
                    <button
                        onClick={handleSend}
                        disabled={sending || !inputValue.trim()}
                        className="absolute right-2 p-1.5 bg-primary hover:bg-[var(--accent-coral-dark)] rounded-lg text-white transition-colors shadow-lg disabled:opacity-50"
                    >
                        <Send className="w-[18px] h-[18px]" />
                    </button>
                </div>
                <p className="text-[10px] text-center text-gray-600 mt-2">AI can make mistakes. Check important info.</p>
            </div>
        </aside>
    );
}

export default ChatBot;
