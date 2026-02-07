import { useEffect, useRef } from 'react';
import { AlertTriangle, X } from 'lucide-react';
import './ConfirmModal.css';

function ConfirmModal({ isOpen, title, message, confirmText = '삭제', cancelText = '취소', onConfirm, onCancel, variant = 'danger' }) {
    const modalRef = useRef(null);
    const confirmButtonRef = useRef(null);

    useEffect(() => {
        if (isOpen) {
            confirmButtonRef.current?.focus();
            document.body.style.overflow = 'hidden';
        } else {
            document.body.style.overflow = '';
        }
        return () => {
            document.body.style.overflow = '';
        };
    }, [isOpen]);

    useEffect(() => {
        const handleEscape = (e) => {
            if (e.key === 'Escape' && isOpen) {
                onCancel();
            }
        };
        document.addEventListener('keydown', handleEscape);
        return () => document.removeEventListener('keydown', handleEscape);
    }, [isOpen, onCancel]);

    const handleBackdropClick = (e) => {
        if (e.target === modalRef.current) {
            onCancel();
        }
    };

    if (!isOpen) return null;

    return (
        <div className="confirm-modal-backdrop" ref={modalRef} onClick={handleBackdropClick}>
            <div className={`confirm-modal confirm-modal--${variant}`}>
                <button className="confirm-modal-close" onClick={onCancel} aria-label="닫기">
                    <X className="w-5 h-5" />
                </button>

                <div className="confirm-modal-icon">
                    <AlertTriangle className="w-8 h-8" />
                </div>

                <h2 className="confirm-modal-title">{title}</h2>
                <p className="confirm-modal-message">{message}</p>

                <div className="confirm-modal-actions">
                    <button className="confirm-modal-btn confirm-modal-btn--cancel" onClick={onCancel}>
                        {cancelText}
                    </button>
                    <button
                        ref={confirmButtonRef}
                        className={`confirm-modal-btn confirm-modal-btn--confirm confirm-modal-btn--${variant}`}
                        onClick={onConfirm}
                    >
                        {confirmText}
                    </button>
                </div>
            </div>
        </div>
    );
}

export default ConfirmModal;
