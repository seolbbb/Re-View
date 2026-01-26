import { useState } from 'react';
import './ChatBot.css';

function ChatBot() {
    const [messages] = useState([
        { id: 1, type: 'bot', text: '안녕하세요! Re:View AI입니다. 영상에 대해 궁금한 점이 있으시면 질문해주세요.' },
        { id: 2, type: 'user', text: '이 영상의 핵심 내용이 뭐야?' },
        { id: 3, type: 'bot', text: '이 영상은 인공지능 기초 강의로, 머신러닝의 기본 개념과 신경망의 작동 원리에 대해 설명하고 있습니다. 주요 내용은:\n\n1. 지도학습과 비지도학습의 차이\n2. 딥러닝의 발전 역사\n3. 실제 적용 사례' },
    ]);
    const [inputValue, setInputValue] = useState('');

    return (
        <div className="chatbot">
            <div className="chatbot-header">
                <div className="chatbot-title">
                    <span className="bot-icon">🤖</span>
                    <span>AI 챗봇</span>
                </div>
                <div className="status-badge">
                    <span className="status-dot"></span>
                    분석 완료
                </div>
            </div>

            <div className="chatbot-messages">
                {messages.map((msg) => (
                    <div key={msg.id} className={`message ${msg.type}`}>
                        {msg.type === 'bot' && <div className="avatar">AI</div>}
                        <div className="message-content">
                            {msg.text.split('\n').map((line, i) => (
                                <p key={i}>{line}</p>
                            ))}
                        </div>
                    </div>
                ))}
            </div>

            <div className="chatbot-input">
                <input
                    type="text"
                    placeholder="영상에 대해 질문하세요..."
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                />
                <button className="send-btn">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" />
                    </svg>
                </button>
            </div>
        </div>
    );
}

export default ChatBot;
