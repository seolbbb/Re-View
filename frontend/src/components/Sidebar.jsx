
import React, { useState, useEffect, useRef } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Video, Home, Library, History, CheckCircle, ChevronsUpDown, Loader2, LogOut } from 'lucide-react';
import { listVideos } from '../api/videos';
import { useAuth } from '../context/AuthContext';

function Sidebar() {
    const [recentVideos, setRecentVideos] = useState([]);
    const [loading, setLoading] = useState(true);
    const [profileOpen, setProfileOpen] = useState(false);
    const profileRef = useRef(null);
    const { user, signOut } = useAuth();
    const navigate = useNavigate();

    useEffect(() => {
        let mounted = true;

        const refresh = () => {
            setLoading(true);
            listVideos()
                .then((data) => {
                    if (!mounted) return;
                    setRecentVideos((data.videos || []).slice(0, 5));
                })
                .catch(() => {})
                .finally(() => {
                    if (!mounted) return;
                    setLoading(false);
                });
        };

        refresh();
        window.addEventListener('videos:changed', refresh);
        return () => {
            mounted = false;
            window.removeEventListener('videos:changed', refresh);
        };
    }, []);

    // Close profile dropup on outside click
    useEffect(() => {
        const handleClick = (e) => {
            if (profileRef.current && !profileRef.current.contains(e.target)) {
                setProfileOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClick);
        return () => document.removeEventListener('mousedown', handleClick);
    }, []);

    const handleSignOut = async () => {
        setProfileOpen(false);
        try {
            await signOut();
            navigate('/login');
        } catch (err) {
            console.error('Sign out failed:', err);
        }
    };

    const getStatusIcon = (status) => {
        const s = (status || '').toUpperCase();
        if (s === 'DONE') return <CheckCircle className="w-5 h-5 text-gray-400 group-hover:text-[var(--text-primary)]" />;
        return <History className="w-5 h-5 text-gray-400 group-hover:text-[var(--text-primary)]" />;
    };

    const formatTimeAgo = (isoStr) => {
        if (!isoStr) return '';
        const diff = Date.now() - new Date(isoStr).getTime();
        const mins = Math.floor(diff / 60000);
        if (mins < 60) return `${mins}m ago`;
        const hours = Math.floor(mins / 60);
        if (hours < 24) return `${hours}h ago`;
        const days = Math.floor(hours / 24);
        return `${days}d ago`;
    };

    const userEmail = user?.email || '';
    const userInitial = userEmail ? userEmail[0].toUpperCase() : 'U';
    const displayName = user?.user_metadata?.full_name || userEmail.split('@')[0] || 'User';

    return (
        <aside className="w-64 flex flex-col bg-surface border-r border-[var(--border-color)] shrink-0 z-20 hidden md:flex h-full">
            <div className="flex flex-col flex-1 p-4 gap-6">
                {/* App Logo */}
                <div className="flex items-center gap-3 px-2">
                    <div className="bg-primary/20 flex items-center justify-center rounded-lg size-10">
                        <Video className="w-6 h-6 text-primary" />
                    </div>
                    <div className="flex flex-col">
                        <h1 className="text-[var(--text-primary)] text-base font-bold leading-tight tracking-tight">Re:View</h1>
                        <p className="text-gray-400 text-xs font-normal">Workspace</p>
                    </div>
                </div>
                {/* Navigation */}
                <div className="flex flex-col gap-1">
                    <Link to="/" className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-[var(--text-primary)] hover:bg-surface-highlight transition-colors group">
                        <Home className="w-5 h-5 text-gray-400 group-hover:text-[var(--text-primary)]" />
                        <span className="text-sm font-medium">Home</span>
                    </Link>
                    <a href="#" className="flex items-center gap-3 px-3 py-2.5 rounded-lg bg-surface-highlight text-[var(--text-primary)] transition-colors">
                        <Library className="w-5 h-5 text-primary" />
                        <span className="text-sm font-medium">Library</span>
                    </a>
                </div>
                {/* Recent Lectures */}
                <div className="flex flex-col gap-2 mt-2">
                    <div className="px-3">
                        <p className="text-gray-400 text-xs font-semibold uppercase tracking-wider">Recent Lectures</p>
                    </div>
                    <div className="flex flex-col gap-1">
                        {loading && (
                            <div className="flex justify-center py-2">
                                <Loader2 className="w-4 h-4 animate-spin text-gray-400" />
                            </div>
                        )}
                        {!loading && recentVideos.length === 0 && (
                            <p className="px-3 text-gray-500 text-xs">No videos yet</p>
                        )}
                        {recentVideos.map((v) => (
                            <Link
                                key={v.id}
                                to={`/analysis/${v.id}`}
                                className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-surface-highlight group"
                            >
                                {getStatusIcon(v.status)}
                                <div className="flex flex-col truncate">
                                    <p className="text-[var(--text-primary)] text-sm font-medium truncate">{v.name || v.original_filename}</p>
                                    <p className="text-gray-400 text-xs truncate">{formatTimeAgo(v.created_at)}</p>
                                </div>
                            </Link>
                        ))}
                    </div>
                </div>
            </div>
            {/* User Profile */}
            <div className="p-4 border-t border-[var(--border-color)] relative" ref={profileRef}>
                {/* Dropup Menu */}
                {profileOpen && (
                    <div className="absolute bottom-full left-4 right-4 mb-2 bg-[var(--bg-secondary,var(--bg-primary))] border border-[var(--border-color)] rounded-lg shadow-xl z-50 overflow-hidden">
                        <div className="px-4 py-3 border-b border-[var(--border-color)]">
                            <p className="text-[var(--text-primary)] text-sm font-medium truncate">{userEmail}</p>
                            <p className="text-gray-400 text-xs">Free Plan</p>
                        </div>
                        <button
                            onClick={handleSignOut}
                            className="w-full flex items-center gap-2 px-4 py-2.5 text-sm text-red-400 hover:bg-surface-highlight transition-colors text-left"
                        >
                            <LogOut className="w-4 h-4" />
                            로그아웃
                        </button>
                    </div>
                )}
                <button
                    onClick={() => setProfileOpen((o) => !o)}
                    className="flex items-center gap-3 w-full hover:bg-surface-highlight p-2 rounded-lg transition-colors text-left"
                >
                    <div className="bg-center bg-no-repeat bg-cover rounded-full size-8 bg-gray-600 flex items-center justify-center text-white text-xs font-bold">
                        {userInitial}
                    </div>
                    <div className="flex flex-col">
                        <p className="text-[var(--text-primary)] text-sm font-medium truncate">{displayName}</p>
                        <p className="text-gray-400 text-xs">Free Plan</p>
                    </div>
                    <ChevronsUpDown className="w-[18px] h-[18px] text-gray-400 ml-auto" />
                </button>
            </div>
        </aside>
    );
}

export default Sidebar;
