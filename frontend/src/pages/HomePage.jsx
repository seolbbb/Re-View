import { useState, useEffect } from 'react';
import Header from '../components/Header';
import VideoCard from '../components/VideoCard';
import UploadArea from '../components/UploadArea';
import { Library, Loader2 } from 'lucide-react';
import { listVideos, deleteVideo } from '../api/videos';
import './HomePage.css';

// DB 상태 → UI 상태 매핑
function mapStatus(dbStatus) {
    if (!dbStatus) return 'pending';
    const s = dbStatus.toUpperCase();
    if (s === 'DONE') return 'done';
    if (s === 'FAILED') return 'done';
    if (['PREPROCESSING', 'PREPROCESS_DONE', 'PROCESSING', 'VLM_RUNNING', 'SUMMARY_RUNNING', 'JUDGE_RUNNING'].includes(s))
        return 'progress';
    return 'pending';
}

function formatDuration(sec) {
    if (!sec) return '--:--';
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return `${m}:${String(s).padStart(2, '0')}`;
}

function formatDate(isoStr) {
    if (!isoStr) return '';
    const d = new Date(isoStr);
    return `${d.getFullYear()}.${String(d.getMonth() + 1).padStart(2, '0')}.${String(d.getDate()).padStart(2, '0')}`;
}

function HomePage() {
    const [videos, setVideos] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [filter, setFilter] = useState('all');

    useEffect(() => {
        setLoading(true);
        listVideos()
            .then((data) => {
                setVideos(data.videos || []);
                setError(null);
            })
            .catch((err) => setError(err.message || 'Failed to load videos'))
            .finally(() => setLoading(false));
    }, []);

    const handleDelete = async (videoId) => {
        const ok = window.confirm('이 영상을 삭제할까요? (되돌릴 수 없습니다)');
        if (!ok) return;

        try {
            await deleteVideo(videoId);
            setVideos((prev) => (prev || []).filter((v) => v.id !== videoId));
            window.dispatchEvent(new Event('videos:changed'));
        } catch (err) {
            setError(err?.message || 'Failed to delete video');
        }
    };

    const filtered = videos.filter((v) => {
        if (filter === 'all') return true;
        const uiStatus = mapStatus(v.status);
        if (filter === 'done') return uiStatus === 'done';
        if (filter === 'progress') return uiStatus === 'progress' || uiStatus === 'pending';
        return true;
    });

    return (
        <div className="home-page">
            <Header />

            <main className="home-content">
                {/* Hero Upload Section */}
                <section className="hero-section">
                    <div className="hero-content">
                        <h1 className="hero-title">
                            <span className="gradient-text">AI로 영상을</span>
                            <br />
                            스마트하게 분석하세요
                        </h1>
                        <p className="hero-subtitle">
                            영상을 업로드하면 AI가 자동으로 요약하고, 질문에 답변해드립니다
                        </p>
                        <div className="hero-upload">
                            <UploadArea />
                        </div>
                        <div className="hero-stats">
                            <div className="stat-item">
                                <span className="stat-number">{videos.length}</span>
                                <span className="stat-label">분석된 영상</span>
                            </div>
                            <div className="stat-divider"></div>
                            <div className="stat-item">
                                <span className="stat-number">98%</span>
                                <span className="stat-label">정확도</span>
                            </div>
                            <div className="stat-divider"></div>
                            <div className="stat-item">
                                <span className="stat-number">2분</span>
                                <span className="stat-label">평균 분석 시간</span>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Library Section */}
                <section className="library-section">
                    <div className="section-header">
                        <div className="section-title-group">
                            <h2><Library className="inline-block w-6 h-6 mr-2 align-text-bottom" /> 내 라이브러리</h2>
                            <span className="video-count">{filtered.length}개 영상</span>
                        </div>
                        <div className="section-filters">
                            <button className={`filter-btn ${filter === 'all' ? 'active' : ''}`} onClick={() => setFilter('all')}>전체</button>
                            <button className={`filter-btn ${filter === 'done' ? 'active' : ''}`} onClick={() => setFilter('done')}>분석 완료</button>
                            <button className={`filter-btn ${filter === 'progress' ? 'active' : ''}`} onClick={() => setFilter('progress')}>진행 중</button>
                        </div>
                    </div>

                    {loading && (
                        <div className="flex justify-center py-12">
                            <Loader2 className="w-8 h-8 animate-spin text-primary" />
                        </div>
                    )}

                    {error && (
                        <div className="text-center py-12 text-red-400">
                            <p>{error}</p>
                        </div>
                    )}

                    {!loading && !error && filtered.length === 0 && (
                        <div className="text-center py-12 text-gray-400">
                            <p>아직 업로드된 영상이 없습니다.</p>
                        </div>
                    )}

                    <div className="video-grid">
                        {filtered.map((video, index) => (
                            <div key={video.id} style={{ '--index': index }}>
                                <VideoCard
                                    id={video.id}
                                    title={video.name || video.original_filename}
                                    thumbnailVideoId={video.id}
                                    duration={formatDuration(video.duration_sec)}
                                    date={formatDate(video.created_at)}
                                    status={mapStatus(video.status)}
                                    onDelete={handleDelete}
                                />
                            </div>
                        ))}
                    </div>
                </section>
            </main>

            {/* Footer */}
            <footer className="home-footer">
                <p>&copy; 2026 Re:View. AI-Powered Video Analysis Platform</p>
            </footer>
        </div>
    );
}

export default HomePage;
