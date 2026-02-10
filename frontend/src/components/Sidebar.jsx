import React, { useState, useEffect, useRef } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
    Home,
    Library,
    Video,
    Trash2,
    ChevronRight,
    LogOut,
    User,
    Settings,
    PlusCircle,
    Loader2,
    ChevronsUpDown
} from 'lucide-react';
import { supabase } from '../lib/supabase';
import ConfirmModal from './ConfirmModal';

function Sidebar({ isOpen, onClose }) {
    const [recentVideos, setRecentVideos] = useState([]);
    const [loading, setLoading] = useState(true);
    const [profileOpen, setProfileOpen] = useState(false);
    const [userEmail, setUserEmail] = useState('');
    const [displayName, setDisplayName] = useState('');
    const [deleteTarget, setDeleteTarget] = useState(null);
    const profileRef = useRef(null);
    const navigate = useNavigate();

    useEffect(() => {
        fetchRecentVideos();
        fetchUserData();

        const handleClickOutside = (event) => {
            if (profileRef.current && !profileRef.current.contains(event.target)) {
                setProfileOpen(false);
            }
        };

        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    const fetchUserData = async () => {
        const { data: { user } } = await supabase.auth.getUser();
        if (user) {
            setUserEmail(user.email);
            setDisplayName(user.user_metadata?.full_name || user.email.split('@')[0]);
        }
    };

    const fetchRecentVideos = async () => {
        try {
            const { data: { user } } = await supabase.auth.getUser();
            if (!user) return;

            const { data, error } = await supabase
                .from('videos')
                .select('*')
                .eq('user_id', user.id)
                .order('created_at', { ascending: false })
                .limit(5);

            if (error) throw error;
            setRecentVideos(data || []);
        } catch (error) {
            console.error('Error fetching recent videos:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleSignOut = async () => {
        await supabase.auth.signOut();
        navigate('/login');
    };

    const handleDeleteRequest = (e, videoId, status) => {
        e.preventDefault();
        e.stopPropagation();
        setDeleteTarget({ id: videoId, status });
    };

    const handleDeleteConfirm = async () => {
        if (!deleteTarget) return;

        try {
            const { error } = await supabase
                .from('videos')
                .delete()
                .eq('id', deleteTarget.id);

            if (error) throw error;

            setRecentVideos(prev => prev.filter(v => v.id !== deleteTarget.id));

            const currentPath = window.location.pathname;
            if (currentPath.includes(deleteTarget.id)) {
                navigate('/');
            }
        } catch (error) {
            console.error('Error deleting video:', error);
            alert('영상 삭제 중 오류가 발생했습니다.');
        } finally {
            setDeleteTarget(null);
        }
    };

    const handleDeleteCancel = () => {
        setDeleteTarget(null);
    };

    const formatTimeAgo = (dateString) => {
        const now = new Date();
        const past = new Date(dateString);
        const diffInMs = now - past;
        const diffInMins = Math.floor(diffInMs / (1000 * 60));
        const diffInHours = Math.floor(diffInMs / (1000 * 60 * 60));
        const diffInDays = Math.floor(diffInMs / (1000 * 60 * 60 * 24));

        if (diffInMins < 60) return `${diffInMins}m ago`;
        if (diffInHours < 24) return `${diffInHours}h ago`;
        return `${diffInDays}d ago`;
    };

    const getStatusIcon = (status) => {
        if (status === 'processing') return <Loader2 size={20} className="animate-spin text-primary" />;
        return <Library size={20} className="text-gray-400 group-hover:text-primary transition-colors" />;
    };

    const userInitial = displayName ? displayName.charAt(0).toUpperCase() : 'U';

    return (
        <>
            {/* Mobile Overlay */}
            {isOpen && (
                <div
                    className="fixed inset-0 bg-black/50 z-[90] lg:hidden backdrop-blur-sm transition-opacity"
                    onClick={onClose}
                />
            )}

            <aside className={`fixed lg:relative z-40 transition-all duration-300 ease-in-out pointer-events-none lg:pointer-events-auto
                /* Mobile: Full Screen Overlay */
                inset-0 w-full
                /* Desktop: Left Column with Padding for Gradient Effect */
                lg:inset-auto lg:h-full lg:w-[250px] lg:pt-4 lg:pl-4 lg:pb-0
                ${isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
                flex flex-col`}>

                {/* Floating Card Content - Flush with background, shadow for gradient separation */}
                <div className="flex flex-col h-full bg-[var(--bg-primary)] lg:bg-surface border-r lg:border-r-0 lg:border-y lg:border-l border-[var(--border-color)] lg:rounded-l-2xl shadow-[0_-15px_50px_rgba(0,0,0,0.2)] relative pointer-events-auto overflow-y-auto custom-scrollbar">
                    {/* Main Content Scrollable Container */}
                    <div className="flex flex-col flex-1 w-full p-6 gap-4">
                        {/* Header: Modular Card Style */}
                        <div className="flex items-center justify-between px-4 w-full h-[50px] bg-surface-highlight/5 rounded-xl shadow-sm">
                            <div className="w-12 flex justify-center shrink-0">
                                <div className="bg-primary/20 flex items-center justify-center rounded-lg w-10 h-10">
                                    <Video size={20} className="text-primary" />
                                </div>
                            </div>

                            <div className="flex flex-col items-center justify-center flex-1 min-w-0">
                                <h1 className="text-base font-bold leading-none tracking-tight text-[var(--text-primary)]">Re:View</h1>
                            </div>

                            <div className="w-12 flex justify-center shrink-0">
                                <button
                                    onClick={onClose}
                                    className="lg:hidden p-1 text-gray-400 hover:text-primary transition-all duration-300 bg-surface-highlight rounded-full"
                                >
                                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-x"><path d="M18 6 6 18"></path><path d="m6 6 12 12"></path></svg>
                                </button>
                            </div>
                        </div>

                        {/* Navigation */}
                        <div className="flex flex-col gap-2">
                            <Link
                                to="/"
                                onClick={() => { if (window.innerWidth < 1024) onClose(); }}
                                className="flex items-center px-4 py-3.5 rounded-2xl text-[var(--text-primary)] hover:bg-surface-highlight transition-all duration-300 group"
                            >
                                <div className="w-12 flex justify-center shrink-0">
                                    <Home size={20} className="text-gray-400 group-hover:text-[var(--text-primary)]" />
                                </div>
                                <span className="flex-1 text-center text-base font-medium text-[var(--text-primary)]">Home</span>
                                <div className="w-12 shrink-0" />
                            </Link>
                        </div>

                        {/* Recent Lectures */}
                        <div className="flex flex-col gap-4">
                            <div className="px-3 text-center">
                                <p className="text-gray-400 text-xs font-semibold uppercase tracking-wider">Recent Lectures</p>
                            </div>
                            <div className="flex flex-col gap-2">
                                {loading && (
                                    <div className="flex justify-center py-6">
                                        <Loader2 className="w-4 h-4 animate-spin text-gray-400" />
                                    </div>
                                )}
                                {!loading && recentVideos.length === 0 && (
                                    <p className="px-3 text-gray-500 text-sm lg:text-xs text-center">No videos yet</p>
                                )}
                                {recentVideos.map((v) => (
                                    <Link
                                        key={v.id}
                                        to={`/analysis/${v.id}`}
                                        onClick={() => { if (window.innerWidth < 1024) onClose(); }}
                                        className="flex items-center px-4 py-3 rounded-2xl hover:bg-surface-highlight transition-all duration-300 group"
                                    >
                                        <div className="w-12 flex justify-center shrink-0">
                                            {getStatusIcon(v.status)}
                                        </div>
                                        <div className="flex flex-col items-center text-center flex-1 min-w-0">
                                            <p className="text-[var(--text-primary)] text-sm font-medium truncate w-full">{v.name || v.original_filename}</p>
                                            <p className="text-gray-400 text-xs truncate w-full">{formatTimeAgo(v.created_at)}</p>
                                        </div>
                                        <div className="w-12 flex justify-center shrink-0">
                                            <button
                                                onClick={(e) => handleDeleteRequest(e, v.id, v.status)}
                                                className="opacity-100 text-gray-400 hover:text-red-400 transition-all duration-300 p-2 rounded-lg hover:bg-red-500/10"
                                            >
                                                <Trash2 size={20} />
                                            </button>
                                        </div>
                                    </Link>
                                ))}
                            </div>
                        </div>
                    </div>

                    {/* Profile Section */}
                    <div className="w-full p-6 bg-[var(--bg-primary)] lg:bg-transparent sticky bottom-0">
                        <div className="relative" ref={profileRef}>
                            {profileOpen && (
                                <div className="absolute bottom-full left-0 right-0 mb-3 bg-[var(--bg-secondary,var(--bg-primary))] border border-[var(--border-color)] rounded-2xl shadow-xl z-50 p-4 flex items-center justify-between animate-fade-in group/dropup">
                                    {/* Left Spacer to match Avatar width */}
                                    <div className="w-12 shrink-0" />

                                    <div className="flex flex-col items-center text-center flex-1 min-w-0">
                                        <p className="text-[var(--text-primary)] text-sm font-bold truncate w-full">{displayName}</p>
                                        <p className="text-gray-400 text-[11px] truncate w-full">{userEmail}</p>
                                    </div>

                                    <div className="w-12 flex justify-center shrink-0">
                                        <button
                                            onClick={handleSignOut}
                                            className="text-gray-400 hover:text-red-400 p-2 rounded-lg hover:bg-red-500/10 transition-all duration-300 shrink-0"
                                            title="로그아웃"
                                        >
                                            <LogOut className="w-5 h-5" />
                                        </button>
                                    </div>
                                </div>
                            )}
                            <button
                                onClick={() => setProfileOpen((o) => !o)}
                                className="flex items-center w-full border border-[var(--border-color)] bg-surface-highlight/5 hover:bg-surface-highlight/10 p-4 rounded-2xl transition-all duration-300 shadow-sm group"
                            >
                                <div className="w-12 flex justify-center shrink-0">
                                    <div className="bg-center bg-no-repeat bg-cover rounded-full w-6 h-6 bg-gray-600 flex items-center justify-center text-white text-[10px] font-bold ring-2 ring-primary/20 group-hover:ring-primary/40 transition-all">
                                        {userInitial}
                                    </div>
                                </div>
                                <div className="flex flex-col items-center text-center flex-1 min-w-0">
                                    <p className="text-[var(--text-primary)] text-sm font-bold truncate w-full">{displayName}</p>
                                    <p className="text-gray-400 text-xs font-medium w-full">Free Plan</p>
                                </div>
                                <div className="w-12 flex justify-center shrink-0">
                                    <ChevronsUpDown size={20} className="text-gray-400 group-hover:text-primary transition-colors" />
                                </div>
                            </button>
                        </div>
                    </div>

                    <ConfirmModal
                        isOpen={deleteTarget !== null}
                        title="영상 삭제"
                        message={deleteTarget?.status?.toUpperCase() === 'PROCESSING'
                            ? '이 영상은 현재 처리 중입니다.\n정말 삭제하시겠습니까?'
                            : '이 영상을 삭제하시겠습니까?\n삭제된 영상은 복구할 수 없습니다.'}
                        confirmText="삭제"
                        cancelText="취소"
                        onConfirm={handleDeleteConfirm}
                        onCancel={handleDeleteCancel}
                        variant="danger"
                    />
                </div>
            </aside>
        </>
    );
}

export default Sidebar;
