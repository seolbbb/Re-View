import { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { useTheme } from '../context/ThemeContext';
import { useVideo } from '../context/VideoContext';
import { getVideoStatus } from '../api/videos';
import Sidebar from '../components/Sidebar';
import VideoPlayer from '../components/VideoPlayer';
import ChatBot from '../components/ChatBot';
import SummaryPanel from '../components/SummaryPanel';
import { Menu, ChevronRight, Sun, Moon, Share2, Download } from 'lucide-react';

function AnalysisPage() {
    const { id: videoId } = useParams();
    const [isExpanded, setIsExpanded] = useState(false);
    const { theme, toggleTheme } = useTheme();
    const { setCurrentVideoId } = useVideo();
    const [videoInfo, setVideoInfo] = useState(null);

    useEffect(() => {
        if (videoId) {
            setCurrentVideoId(videoId);
            getVideoStatus(videoId)
                .then((data) => setVideoInfo(data))
                .catch(() => {});
        }
    }, [videoId, setCurrentVideoId]);

    const videoName = videoInfo?.video_status
        ? (videoInfo.video_name || videoId)
        : videoId;

    const statusLabel = videoInfo?.video_status || 'Loading';

    return (
        <div className="bg-[var(--bg-primary)] text-[var(--text-primary)] font-display flex h-screen overflow-hidden selection:bg-primary/40 selection:text-white transition-colors duration-300" data-theme={theme}>
            {/* Left Sidebar */}
            <Sidebar />

            {/* Main Content Area */}
            <main className="flex-1 flex flex-col min-w-0 h-full relative bg-[var(--bg-primary)] transition-colors duration-300">
                {/* Header / Breadcrumbs */}
                <header className="h-16 flex items-center justify-between px-6 border-b border-[var(--border-color)] bg-[var(--bg-primary)]/95 backdrop-blur z-10 shrink-0">
                    <div className="flex items-center gap-2">
                        <button className="md:hidden mr-2 text-gray-400">
                            <Menu className="w-5 h-5" />
                        </button>
                        <a href="/" className="text-gray-400 text-sm font-medium hover:text-[var(--text-primary)] transition-colors">Library</a>
                        <ChevronRight className="w-4 h-4 text-gray-400" />
                        <div className="flex items-center gap-2">
                            <span className="text-[var(--text-primary)] text-sm font-medium truncate max-w-[200px]">{videoName}</span>
                            <span className="bg-primary/20 text-primary text-[10px] font-bold px-1.5 py-0.5 rounded uppercase tracking-wide">{statusLabel}</span>
                        </div>
                    </div>
                    <div className="flex items-center gap-4">
                        <button
                            onClick={toggleTheme}
                            className="flex items-center justify-center text-gray-400 hover:text-[var(--text-primary)] transition-colors mr-2"
                            title={theme === 'dark' ? "Switch to Light Mode" : "Switch to Dark Mode"}
                        >
                            {theme === 'dark' ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
                        </button>
                        <button className="flex items-center gap-2 text-gray-400 hover:text-[var(--text-primary)] transition-colors">
                            <Share2 className="w-5 h-5" />
                            <span className="text-sm font-medium hidden sm:block">Share</span>
                        </button>
                        <button className="flex items-center gap-2 text-gray-400 hover:text-[var(--text-primary)] transition-colors">
                            <Download className="w-5 h-5" />
                            <span className="text-sm font-medium hidden sm:block">Export</span>
                        </button>
                    </div>
                </header>

                {/* Workspace Grid */}
                <div className="flex flex-1 overflow-hidden relative">
                    {/* Left Column: Video & Summary */}
                    <div className="flex-1 relative min-w-0 bg-[var(--bg-primary)]">
                        {isExpanded ? (
                            <>
                                {/* Detailed Summary View (Full) */}
                                <SummaryPanel isExpanded={true} onToggleExpand={() => setIsExpanded(false)} videoId={videoId} />

                                {/* PIP Video Player */}
                                <div className="absolute bottom-6 right-6 w-80 lg:w-96 aspect-video z-50 transition-all hover:scale-[1.02]">
                                    <VideoPlayer
                                        isPip={true}
                                        onTogglePip={() => setIsExpanded(false)}
                                        videoId={videoId}
                                        className="relative w-full h-full rounded-xl overflow-hidden shadow-[0_8px_30px_rgb(0,0,0,0.5)] border border-[var(--border-color)] ring-1 ring-white/5 group bg-black"
                                    />
                                </div>
                            </>
                        ) : (
                            // Normal View
                            <div className="flex flex-col overflow-y-auto custom-scrollbar p-6 lg:p-8 gap-8 min-w-0 h-full">
                                {/* Video Player Section */}
                                <div className="flex flex-col gap-4">
                                    <VideoPlayer isPip={false} videoId={videoId} />
                                </div>

                                {/* Summary Section */}
                                <SummaryPanel isExpanded={false} onToggleExpand={() => setIsExpanded(true)} videoId={videoId} />
                            </div>
                        )}
                    </div>

                    {/* Right Column: AI Chatbot */}
                    <ChatBot videoId={videoId} />
                </div>
            </main>
        </div>
    );
}

export default AnalysisPage;
