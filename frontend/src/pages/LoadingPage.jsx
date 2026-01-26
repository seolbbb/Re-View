import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload, Music, FileText, Bot } from 'lucide-react';
import './LoadingPage.css';

function LoadingPage() {
    const [progress, setProgress] = useState(0);
    const [stage, setStage] = useState(0);
    const navigate = useNavigate();

    const stages = [
        { name: '영상 업로드 중...', Icon: Upload },
        { name: '오디오 추출 중...', Icon: Music },
        { name: 'STT 변환 중...', Icon: FileText },
        { name: 'AI 분석 시작...', Icon: Bot },
    ];

    useEffect(() => {
        const interval = setInterval(() => {
            setProgress((prev) => {
                if (prev >= 100) {
                    clearInterval(interval);
                    setTimeout(() => navigate('/analysis/demo'), 500);
                    return 100;
                }
                return prev + 2;
            });
        }, 100);

        return () => clearInterval(interval);
    }, [navigate]);

    useEffect(() => {
        if (progress < 25) setStage(0);
        else if (progress < 50) setStage(1);
        else if (progress < 75) setStage(2);
        else setStage(3);
    }, [progress]);

    const CurrentIcon = stages[stage].Icon;

    return (
        <div className="loading-page">
            <div className="loading-card">
                <div className="loading-logo">
                    <span className="logo-re">Re:</span>
                    <span className="logo-view">View</span>
                </div>

                <div className="loading-spinner">
                    <div className="spinner"></div>
                </div>

                <div className="loading-stage">
                    <span className="stage-icon"><CurrentIcon className="w-6 h-6" /></span>
                    <span className="stage-name">{stages[stage].name}</span>
                </div>

                <div className="loading-progress">
                    <div className="progress-bar-large">
                        <div
                            className="progress-fill-large"
                            style={{ width: `${progress}%` }}
                        ></div>
                    </div>
                    <span className="progress-text">{progress}%</span>
                </div>

                <div className="loading-steps">
                    {stages.map((s, index) => (
                        <div
                            key={index}
                            className={`step ${index < stage ? 'done' : ''} ${index === stage ? 'active' : ''}`}
                        >
                            <div className="step-dot"></div>
                            <span>{s.name.replace('...', '')}</span>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}

export default LoadingPage;
