import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './UploadArea.css';

function UploadArea() {
    const [isDragging, setIsDragging] = useState(false);
    const navigate = useNavigate();

    const handleDragOver = (e) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = () => {
        setIsDragging(false);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragging(false);
        // 데모: 로딩 페이지로 이동
        navigate('/loading');
    };

    const handleClick = () => {
        // 데모: 로딩 페이지로 이동
        navigate('/loading');
    };

    return (
        <div
            className={`upload-area ${isDragging ? 'dragging' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={handleClick}
        >
            <div className="upload-icon">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242" />
                    <path d="M12 12v9" />
                    <path d="m16 16-4-4-4 4" />
                </svg>
            </div>
            <p className="upload-text">여기에 영상을 드래그하거나</p>
            <p className="upload-text">클릭하여 업로드</p>
            <p className="upload-hint">MP4, MOV, AVI (최대 2GB)</p>
        </div>
    );
}

export default UploadArea;
