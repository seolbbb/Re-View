import { useParams, Link } from 'react-router-dom';
import Header from '../components/Header';
import VideoPlayer from '../components/VideoPlayer';
import ChatBot from '../components/ChatBot';
import SummaryPanel from '../components/SummaryPanel';
import './AnalysisPage.css';

function AnalysisPage() {
    const { id } = useParams();

    return (
        <div className="analysis-page">
            <Header />

            <main className="analysis-content">
                <div className="analysis-header">
                    <Link to="/" className="back-btn">← 뒤로가기</Link>
                    <h1>인공지능 기초 강의 1강</h1>
                    <div className="video-meta">
                        <span>📅 2026.01.20</span>
                        <span>⏱️ 14:25</span>
                        <span className="analysis-badge">✨ AI 분석 완료</span>
                    </div>
                </div>

                <div className="analysis-layout">
                    {/* Left: Video Player */}
                    <div className="video-section">
                        <VideoPlayer />
                    </div>

                    {/* Right: ChatBot */}
                    <div className="chat-section">
                        <ChatBot />
                    </div>
                </div>

                {/* Bottom: Summary */}
                <div className="summary-section">
                    <SummaryPanel />
                </div>
            </main>
        </div>
    );
}

export default AnalysisPage;
