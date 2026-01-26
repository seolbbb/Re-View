import { useState } from 'react';
import { Sparkles, RotateCcw, Bot, Send } from 'lucide-react';

function ChatBot() {
    const [messages] = useState([
        {
            id: 1,
            type: 'bot',
            text: 'Hi! I\'m analyzing the lecture on Mitosis. Feel free to ask me to clarify any concepts or generate a quiz!',
            timestamp: 'Just now'
        },
        {
            id: 2,
            type: 'user',
            text: 'Can you explain what happens to the spindle fibers in prophase?',
            timestamp: '2 mins ago'
        },
        {
            id: 3,
            type: 'bot',
            text: 'Absolutely. During prophase, the mitotic spindle begins to form. The spindle fibers are made of microtubules that extend from the centrosomes. They push the centrosomes apart as they grow.',
            timestamp: 'Just now',
            actions: ['Explain simpler', 'Make a quiz']
        },
    ]);
    const [inputValue, setInputValue] = useState('');

    return (
        <aside className="w-[360px] hidden lg:flex flex-col border-l border-[var(--border-color)] bg-surface shrink-0 h-full">
            {/* Chat Header */}
            <div className="p-4 border-b border-[var(--border-color)] bg-surface z-10 shadow-sm flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <Sparkles className="w-5 h-5 text-primary" />
                    <h3 className="text-[var(--text-primary)] font-medium text-sm">Ask Re:View AI</h3>
                </div>
                <button className="text-gray-400 hover:text-[var(--text-primary)]" title="Clear History">
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
                            <div className="size-8 rounded-full bg-gray-700 flex items-center justify-center shrink-0" style={{ backgroundImage: 'url("https://lh3.googleusercontent.com/aida-public/AB6AXuB7Wy36nRa92eDOTh8zejr_WMJQpHNQjbGDYsZPIFRNXkpRuhVpaMpqZWoQFCHS8W-KxRKGKJp2-P0ra0IFtCaxuCgCgo4eNPR8L4VTF-yaFOHkTznSci9FC8PrKvL4y7Tjif1ZHBzJ8qZJLsAICI11qX5NTlbDKJ-GvU3aH_OCkHM965naANsWIKgNmOcLjwpwxK3yqfutdAotQDo-MQjgZrhf7aj-imomG4z1Eq1L2ShZkUbogm22mBtU-tSB27wnKWuvqIKnPTd3")', backgroundSize: 'cover' }}></div>
                        )}

                        <div className={`flex flex-col gap-1 ${msg.type === 'user' ? 'items-end' : ''} max-w-[85%]`}>
                            <div className={`${msg.type === 'bot' ? 'bg-surface-highlight rounded-tl-none border border-[var(--border-color)] text-[var(--text-secondary)]' : 'bg-primary rounded-tr-none text-white'} p-3 rounded-2xl text-sm leading-relaxed shadow-sm`}>
                                <p dangerouslySetInnerHTML={{ __html: msg.text.replace('prophase', '<span class="font-bold text-[var(--text-primary)]">prophase</span>').replace('Mitosis', '<span class="text-primary font-medium">Mitosis</span>') }}></p>
                            </div>
                            <span className={`text-[10px] text-gray-500 ${msg.type === 'user' ? 'pr-1' : 'pl-1'}`}>{msg.timestamp}</span>

                            {msg.actions && (
                                <div className="flex gap-2 mt-1">
                                    {msg.actions.map((action, i) => (
                                        <button key={i} className="text-xs bg-[var(--bg-tertiary)] hover:bg-[var(--bg-hover)] px-2 py-1 rounded text-gray-400 border border-[var(--border-color)] transition-colors">
                                            {action}
                                        </button>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                ))}
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
                    />
                    <button className="absolute right-2 p-1.5 bg-primary hover:bg-[var(--accent-coral-dark)] rounded-lg text-white transition-colors shadow-lg">
                        <Send className="w-[18px] h-[18px]" />
                    </button>
                </div>
                <p className="text-[10px] text-center text-gray-600 mt-2">AI can make mistakes. Check important info.</p>
            </div>
        </aside>
    );
}

export default ChatBot;
