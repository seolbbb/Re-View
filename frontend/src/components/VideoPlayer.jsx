import './VideoPlayer.css';

function VideoPlayer() {
    return (
        <div className="video-player">
            <div className="video-container">
                <div className="video-placeholder">
                    <div className="video-icon">▶</div>
                    <p>영상이 여기에 표시됩니다</p>
                </div>
            </div>
            <div className="video-controls">
                <button className="control-btn">⏮</button>
                <button className="control-btn play-btn">▶</button>
                <button className="control-btn">⏭</button>
                <div className="progress-bar">
                    <div className="progress-fill" style={{ width: '35%' }}></div>
                </div>
                <span className="time-display">02:34 / 07:15</span>
                <button className="control-btn">🔊</button>
                <button className="control-btn">⛶</button>
            </div>
        </div>
    );
}

export default VideoPlayer;
