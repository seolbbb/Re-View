import Header from '../components/Header';
import VideoCard from '../components/VideoCard';
import UploadArea from '../components/UploadArea';
import './HomePage.css';

// 더미 데이터
const dummyVideos = [
    { id: '1', title: '인공지능 기초 강의 1강', thumbnail: 'https://picsum.photos/seed/ai1/400/225', duration: '14:25', date: '2026.01.20', status: 'done' },
    { id: '2', title: '딥러닝 실습 - CNN 구현', thumbnail: 'https://picsum.photos/seed/dl2/400/225', duration: '32:18', date: '2026.01.18', status: 'done' },
    { id: '3', title: '파이썬 데이터 분석', thumbnail: 'https://picsum.photos/seed/py3/400/225', duration: '28:42', date: '2026.01.15', status: 'progress' },
    { id: '4', title: 'NLP 자연어 처리 입문', thumbnail: 'https://picsum.photos/seed/nlp4/400/225', duration: '45:10', date: '2026.01.12', status: 'done' },
    { id: '5', title: '컴퓨터 비전 프로젝트', thumbnail: 'https://picsum.photos/seed/cv5/400/225', duration: '21:33', date: '2026.01.10', status: 'pending' },
    { id: '6', title: 'MLOps 배포 가이드', thumbnail: 'https://picsum.photos/seed/mlops6/400/225', duration: '38:55', date: '2026.01.08', status: 'done' },
];

function HomePage() {
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
                                <span className="stat-number">1,234</span>
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
                            <h2>📚 내 라이브러리</h2>
                            <span className="video-count">{dummyVideos.length}개 영상</span>
                        </div>
                        <div className="section-filters">
                            <button className="filter-btn active">전체</button>
                            <button className="filter-btn">분석 완료</button>
                            <button className="filter-btn">진행 중</button>
                        </div>
                    </div>
                    <div className="video-grid">
                        {dummyVideos.map((video, index) => (
                            <div key={video.id} style={{ '--index': index }}>
                                <VideoCard
                                    id={video.id}
                                    title={video.title}
                                    thumbnail={video.thumbnail}
                                    duration={video.duration}
                                    date={video.date}
                                    status={video.status}
                                />
                            </div>
                        ))}
                    </div>
                </section>
            </main>

            {/* Footer */}
            <footer className="home-footer">
                <p>© 2026 Re:View. AI-Powered Video Analysis Platform</p>
            </footer>
        </div>
    );
}

export default HomePage;
