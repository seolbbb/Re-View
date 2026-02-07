import { useState } from 'react';
import { Link } from 'react-router-dom';
import { Play, Calendar, Trash2 } from 'lucide-react';
import { getThumbnailUrl } from '../api/videos';
import { useAuth } from '../context/AuthContext';
import './VideoCard.css';

function VideoCard({ id, title, thumbnail, thumbnailVideoId, duration, date, status = 'done', onDelete }) {
    const [failedSrc, setFailedSrc] = useState(null);
    const { mediaTicket } = useAuth();

    const statusConfig = {
        done: { label: '완료', color: '#22c55e' },
        progress: { label: '분석 중', color: '#f59e0b' },
        pending: { label: '대기', color: '#6b7280' },
    };

    const currentStatus = statusConfig[status] || statusConfig.done;

    // Use thumbnail prop if provided (legacy), otherwise use API
    const imgSrc = thumbnail || (thumbnailVideoId ? getThumbnailUrl(thumbnailVideoId, mediaTicket) : null);
    const showImage = imgSrc && failedSrc !== imgSrc;

    return (
        <Link to={`/analysis/${id}`} className="video-card">
            {onDelete && (
                <button
                    type="button"
                    className="video-delete"
                    title={status === 'done' ? '삭제' : '처리 중인 영상은 삭제가 제한될 수 있습니다'}
                    aria-label="영상 삭제"
                    onClick={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        onDelete(id);
                    }}
                >
                    <Trash2 className="w-4 h-4" />
                </button>
            )}
            <div className="video-thumbnail">
                {showImage ? (
                    <img src={imgSrc} alt={title} onError={() => setFailedSrc(imgSrc)} />
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
                <span className="video-date"><Calendar className="w-3 h-3" />{date}</span>
            </div>
        </Link>
    );
}

export default VideoCard;
