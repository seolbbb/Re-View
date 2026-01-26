import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { CloudUpload } from 'lucide-react';
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
                <CloudUpload className="w-12 h-12" strokeWidth={1.5} />
            </div>
            <p className="upload-text">여기에 영상을 드래그하거나</p>
            <p className="upload-text">클릭하여 업로드</p>
            <p className="upload-hint">MP4, MOV, AVI (최대 2GB)</p>
        </div>
    );
}

export default UploadArea;
