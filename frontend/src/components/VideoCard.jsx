import { useState } from 'react';
import { Link } from 'react-router-dom';
import { Play, Calendar } from 'lucide-react';
import { getThumbnailUrl } from '../api/videos';
import './VideoCard.css';

function VideoCard({ id, title, thumbnail, thumbnailVideoId, duration, date, status = 'done', onDelete }) {
    const [imgError, setImgError] = useState(false);

    const statusConfig = {
        done: { label: '완료', color: '#22c55e' },
        progress: { label: '분석 중', color: '#f59e0b' },
        pending: { label: '대기', color: '#6b7280' },
    };

    const currentStatus = statusConfig[status] || statusConfig.done;

    // Use thumbnail prop if provided (legacy), otherwise use API
    const imgSrc = thumbnail || (thumbnailVideoId ? getThumbnailUrl(thumbnailVideoId) : null);

    const handleDeleteClick = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (onDelete) {
            onDelete(id, title);
        }
    };

    return (
        <Link to={`/analysis/${id}`} className="video-card">
            <div className="video-thumbnail">
                {imgSrc && !imgError ? (
                    <img src={imgSrc} alt={title} onError={() => setImgError(true)} />
                ) : (
                    <div className="video-thumbnail-placeholder" style={{
                        background: 'linear-gradient(135deg, var(--bg-surface) 0%, var(--bg-secondary) 100%)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        width: '100%',
                        height: '100%',
                        color: 'var(--text-secondary)',
                        fontSize: '0.75rem',
                    }}>
                        <Play className="w-8 h-8" style={{ opacity: 0.3 }} />
                    </div>
                )}
                <div className="video-overlay">
                    <div className="play-icon">
                        <Play className="w-6 h-6" fill="currentColor" />
                    </div>
                </div>
                <span className="video-duration">{duration}</span>
                <span
                    className="video-status"
                    style={{ '--status-color': currentStatus.color }}
                >
                    {status === 'progress' && <span className="status-pulse"></span>}
                    {currentStatus.label}
                </span>
            </div>
            <div className="video-info">
                <h3 className="video-title">{title}</h3>
                <div className="video-meta">
                    <span className="video-date"><Calendar className="w-3 h-3" />{date}</span>
                    {onDelete && (
                        <button
                            className="video-delete-btn"
                            onClick={handleDeleteClick}
                        >
                            삭제
                        </button>
                    )}
                </div>
            </div>
        </Link>
    );
}

export default VideoCard;
