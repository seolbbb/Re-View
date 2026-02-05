import { useEffect } from 'react';
import { X, AlertTriangle, Loader2 } from 'lucide-react';
import './ConfirmDialog.css';

function ConfirmDialog({
    isOpen,
    onClose,
    onConfirm,
    title = '확인',
    message = '이 작업을 진행하시겠습니까?',
    confirmText = '확인',
    cancelText = '취소',
    isLoading = false,
    variant = 'danger', // 'danger' | 'warning' | 'info'
}) {
    // ESC 키로 닫기
    useEffect(() => {
        const handleEsc = (e) => {
            if (e.key === 'Escape' && isOpen && !isLoading) {
                onClose();
            }
        };
        window.addEventListener('keydown', handleEsc);
        return () => window.removeEventListener('keydown', handleEsc);
    }, [isOpen, isLoading, onClose]);

    // 스크롤 방지
    useEffect(() => {
        if (isOpen) {
            document.body.style.overflow = 'hidden';
        } else {
            document.body.style.overflow = '';
        }
        return () => {
            document.body.style.overflow = '';
        };
    }, [isOpen]);

    if (!isOpen) return null;

    return (
        <div className="confirm-dialog-overlay" onClick={!isLoading ? onClose : undefined}>
            <div className="confirm-dialog" onClick={(e) => e.stopPropagation()}>
                <button
                    className="confirm-dialog-close"
                    onClick={onClose}
                    disabled={isLoading}
                    aria-label="닫기"
                >
                    <X className="w-5 h-5" />
                </button>

                <div className={`confirm-dialog-icon ${variant}`}>
                    <AlertTriangle className="w-6 h-6" />
                </div>

                <h2 className="confirm-dialog-title">{title}</h2>
                <p className="confirm-dialog-message">{message}</p>

                <div className="confirm-dialog-actions">
                    <button
                        className="confirm-dialog-btn cancel"
                        onClick={onClose}
                        disabled={isLoading}
                    >
                        {cancelText}
                    </button>
                    <button
                        className={`confirm-dialog-btn confirm ${variant}`}
                        onClick={onConfirm}
                        disabled={isLoading}
                    >
                        {isLoading ? (
                            <>
                                <Loader2 className="w-4 h-4 animate-spin" />
                                처리 중...
                            </>
                        ) : (
                            confirmText
                        )}
                    </button>
                </div>
            </div>
        </div>
    );
}

export default ConfirmDialog;
