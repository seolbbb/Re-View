import { Link } from 'react-router-dom';
import { Play, Calendar } from 'lucide-react';
import './VideoCard.css';

function VideoCard({ id, title, thumbnail, duration, date, status = 'done' }) {
    const statusConfig = {
        done: { label: '완료', color: '#22c55e' },
        progress: { label: '분석 중', color: '#f59e0b' },
        pending: { label: '대기', color: '#6b7280' },
    };

    const currentStatus = statusConfig[status] || statusConfig.done;

    return (
        <Link to={`/analysis/${id}`} className="video-card">
            <div className="video-thumbnail">
                <img src={thumbnail} alt={title} />
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
