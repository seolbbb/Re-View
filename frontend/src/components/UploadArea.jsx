import { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { CloudUpload, Loader2 } from 'lucide-react';
import { uploadVideo } from '../api/videos';
import { useVideo } from '../context/VideoContext';
import './UploadArea.css';

function UploadArea() {
    const [isDragging, setIsDragging] = useState(false);
    const [uploading, setUploading] = useState(false);
    const [error, setError] = useState(null);
    const fileInputRef = useRef(null);
    const navigate = useNavigate();
    const { setCurrentVideoId, setCurrentVideoName } = useVideo();

    const handleFile = async (file) => {
        if (!file) return;
        setUploading(true);
        setError(null);
        try {
            const result = await uploadVideo(file);
            setCurrentVideoId(result.video_id);
            setCurrentVideoName(result.video_name);
            navigate('/loading');
        } catch (err) {
            setError(err.message || '업로드에 실패했습니다.');
        } finally {
            setUploading(false);
        }
    };

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
        const file = e.dataTransfer.files?.[0];
        handleFile(file);
    };

    const handleClick = () => {
        if (!uploading) fileInputRef.current?.click();
    };

    const handleFileChange = (e) => {
        const file = e.target.files?.[0];
        handleFile(file);
    };

    return (
        <div
            className={`upload-area ${isDragging ? 'dragging' : ''} ${uploading ? 'uploading' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={handleClick}
        >
            <input
                ref={fileInputRef}
                type="file"
                accept="video/*"
                style={{ display: 'none' }}
                onChange={handleFileChange}
            />
            <div className="upload-icon">
                {uploading ? (
                    <Loader2 className="w-12 h-12 animate-spin" />
                ) : (
                    <CloudUpload className="w-12 h-12" strokeWidth={1.5} />
                )}
            </div>
            {uploading ? (
                <p className="upload-text">업로드 중...</p>
            ) : (
                <>
                    <p className="upload-text">여기에 영상을 드래그하거나</p>
                    <p className="upload-text">클릭하여 업로드</p>
                </>
            )}
            <p className="upload-hint">MP4, MOV, AVI (최대 2GB)</p>
            {error && <p className="upload-error" style={{ color: '#ef4444', fontSize: '0.875rem', marginTop: '0.5rem' }}>{error}</p>}
        </div>
    );
}

export default UploadArea;
