import { useEffect, useState, useCallback, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload, Music, FileText, Bot, AlertCircle } from 'lucide-react';
import { useVideo } from '../context/VideoContext';
import { getVideoStatus, getVideoProgress } from '../api/videos';
import usePolling from '../hooks/usePolling';
import './LoadingPage.css';

const STAGES = [
    { name: '영상 업로드 중...', Icon: Upload },
    { name: '오디오 추출 중...', Icon: Music },
    { name: 'STT 변환 중...', Icon: FileText },
    { name: 'AI 분석 중...', Icon: Bot },
];

function statusToStage(videoStatus, jobStatus) {
    const vs = (videoStatus || '').toUpperCase();
    const js = (jobStatus || '').toUpperCase();

    if (vs === 'DONE' || js === 'DONE') return { stage: 3, progress: 100 };
    if (vs === 'FAILED' || js === 'FAILED') return { stage: -1, progress: 0 };
    if (js === 'SUMMARY_RUNNING' || js === 'JUDGE_RUNNING') return { stage: 3, progress: null };
    if (js === 'VLM_RUNNING') return { stage: 3, progress: null };
    if (vs === 'PREPROCESS_DONE' || vs === 'PROCESSING') return { stage: 3, progress: null };
    if (vs === 'PREPROCESSING') return { stage: 1, progress: null };
    return { stage: 0, progress: null };
}

function LoadingPage() {
    const { currentVideoId } = useVideo();
    const navigate = useNavigate();
    const [stage, setStage] = useState(0);
    const [progress, setProgress] = useState(0);
    const [failed, setFailed] = useState(false);
    const [errorMsg, setErrorMsg] = useState('');

    const fetchStatus = useCallback(() => {
        if (!currentVideoId) return Promise.resolve(null);
        return getVideoStatus(currentVideoId);
    }, [currentVideoId]);

    const fetchProgress = useCallback(() => {
        if (!currentVideoId) return Promise.resolve(null);
        return getVideoProgress(currentVideoId);
    }, [currentVideoId]);

    const statusUntil = useCallback((data) => {
        if (!data) return false;
        const vs = (data.video_status || '').toUpperCase();
        return vs === 'DONE' || vs === 'FAILED';
    }, []);

    const { data: statusData } = usePolling(fetchStatus, {
        interval: 3000,
        enabled: !!currentVideoId && !failed,
        until: statusUntil,
    });

    const { data: progressData } = usePolling(fetchProgress, {
        interval: 2000,
        enabled: !!currentVideoId && stage >= 3 && !failed,
        until: useCallback((d) => d?.is_complete === true, []),
    });

    // Map status data to stage/progress
    useEffect(() => {
        if (!statusData) return;
        const jobStatus = statusData.processing_job?.status || '';
        const { stage: s, progress: p } = statusToStage(statusData.video_status, jobStatus);

        if (s === -1) {
            setFailed(true);
            setErrorMsg(statusData.error_message || '파이프라인 실행에 실패했습니다.');
            return;
        }

        setStage(s);
        if (p !== null) setProgress(p);
    }, [statusData]);

    // Map progress data to percentage
    useEffect(() => {
        if (!progressData) return;
        if (progressData.is_complete) {
            setProgress(100);
            return;
        }
        if (progressData.has_processing_job && progressData.progress_percent != null) {
            // Phase2: 40-100% range
            const mapped = 40 + (progressData.progress_percent / 100) * 60;
            setProgress(Math.min(Math.round(mapped), 99));
        }
    }, [progressData]);

    // Calculate progress from stage when no processing job progress
    useEffect(() => {
        if (stage === 0) setProgress(5);
        else if (stage === 1) setProgress(15);
        else if (stage === 2) setProgress(30);
        // stage 3+ is handled by progressData
    }, [stage]);

    // Navigate on completion
    useEffect(() => {
        if (progress >= 100 && currentVideoId) {
            const timer = setTimeout(() => navigate(`/analysis/${currentVideoId}`), 800);
            return () => clearTimeout(timer);
        }
    }, [progress, currentVideoId, navigate]);

    // Fallback: if no videoId, show basic progress
    useEffect(() => {
        if (currentVideoId) return;
        const interval = setInterval(() => {
            setProgress((prev) => {
                if (prev >= 100) {
                    clearInterval(interval);
                    return 100;
                }
                return prev + 2;
            });
        }, 100);
        return () => clearInterval(interval);
    }, [currentVideoId]);

    useEffect(() => {
        if (!currentVideoId) {
            if (progress < 25) setStage(0);
            else if (progress < 50) setStage(1);
            else if (progress < 75) setStage(2);
            else setStage(3);
        }
    }, [progress, currentVideoId]);

    const safeStage = Math.max(0, Math.min(stage, STAGES.length - 1));
    const CurrentIcon = STAGES[safeStage].Icon;

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

                {failed ? (
                    <div className="loading-stage" style={{ color: '#ef4444' }}>
                        <span className="stage-icon"><AlertCircle className="w-6 h-6" /></span>
                        <span className="stage-name">처리 실패</span>
                        <p style={{ fontSize: '0.875rem', marginTop: '0.5rem', opacity: 0.8 }}>{errorMsg}</p>
                        <button
                            onClick={() => navigate('/')}
                            style={{
                                marginTop: '1rem',
                                padding: '0.5rem 1.5rem',
                                borderRadius: '0.5rem',
                                background: 'var(--accent-coral)',
                                color: '#fff',
                                border: 'none',
                                cursor: 'pointer',
                            }}
                        >
                            홈으로 돌아가기
                        </button>
                    </div>
                ) : (
                    <>
                        <div className="loading-stage">
                            <span className="stage-icon"><CurrentIcon className="w-6 h-6" /></span>
                            <span className="stage-name">{STAGES[safeStage].name}</span>
                        </div>

                        <div className="loading-progress">
                            <div className="progress-bar-large">
                                <div
                                    className="progress-fill-large"
                                    style={{ width: `${Math.min(progress, 100)}%` }}
                                ></div>
                            </div>
                            <span className="progress-text">{Math.min(progress, 100)}%</span>
                        </div>

                        <div className="loading-steps">
                            {STAGES.map((s, index) => (
                                <div
                                    key={index}
                                    className={`step ${index < safeStage ? 'done' : ''} ${index === safeStage ? 'active' : ''}`}
                                >
                                    <div className="step-dot"></div>
                                    <span>{s.name.replace('...', '')}</span>
                                </div>
                            ))}
                        </div>
                    </>
                )}
            </div>
        </div>
    );
}

export default LoadingPage;
