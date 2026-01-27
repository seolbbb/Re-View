import React, { useRef, useState, useEffect } from 'react';
import { Play, Pause, Maximize2, Volume2, VolumeX, Captions, Settings, Maximize } from 'lucide-react';
import { getVideoStreamUrl } from '../api/videos';

function formatTime(sec) {
    if (!sec || isNaN(sec)) return '0:00';
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${String(s).padStart(2, '0')}`;
}

function VideoPlayer({ isPip, onTogglePip, videoId, className = "", videoElRef, playbackRestore, onTimeUpdate }) {
    const internalRef = useRef(null);
    const videoRef = videoElRef || internalRef;
    const [playing, setPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);
    const [muted, setMuted] = useState(false);
    const progressPercent = duration > 0 ? (currentTime / duration) * 100 : 0;

    const defaultClasses = "relative w-full rounded-xl overflow-hidden shadow-2xl bg-black aspect-video group";
    const finalClasses = className || defaultClasses;

    const streamUrl = videoId ? getVideoStreamUrl(videoId) : null;

    const togglePlay = (e) => {
        e.stopPropagation();
        const v = videoRef.current;
        if (!v) return;
        if (v.paused) {
            v.play();
            setPlaying(true);
        } else {
            v.pause();
            setPlaying(false);
        }
    };

    const handleSeek = (e) => {
        const v = videoRef.current;
        if (!v || !duration) return;
        const rect = e.currentTarget.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const pct = x / rect.width;
        v.currentTime = pct * duration;
    };

    useEffect(() => {
        const v = videoRef.current;
        if (!v) return;

        const restorePlayback = () => {
            if (playbackRestore?.current) {
                const { time, playing: wasPlaying } = playbackRestore.current;
                if (time > 0) {
                    v.currentTime = time;
                    setCurrentTime(time);
                }
                if (wasPlaying) {
                    v.play().catch(() => {});
                    setPlaying(true);
                }
                playbackRestore.current = { time: 0, playing: false };
            }
        };

        const onTime = () => {
            setCurrentTime(v.currentTime);
            onTimeUpdate?.(v.currentTime);
        };
        const onDur = () => {
            setDuration(v.duration);
            restorePlayback();
        };
        const onEnd = () => setPlaying(false);
        v.addEventListener('timeupdate', onTime);
        v.addEventListener('loadedmetadata', onDur);
        v.addEventListener('ended', onEnd);

        // If metadata already loaded (cached video), restore immediately
        if (v.readyState >= 1) {
            setDuration(v.duration);
            restorePlayback();
        }

        return () => {
            v.removeEventListener('timeupdate', onTime);
            v.removeEventListener('loadedmetadata', onDur);
            v.removeEventListener('ended', onEnd);
        };
    }, [streamUrl]);

    return (
        <div className={finalClasses}>
            {streamUrl ? (
                <video
                    ref={videoRef}
                    src={streamUrl}
                    className="absolute inset-0 w-full h-full object-contain"
                    muted={muted}
                    playsInline
                />
            ) : (
                <div className="absolute inset-0 bg-gray-900 flex items-center justify-center">
                    <p className="text-gray-500 text-sm">No video source</p>
                </div>
            )}

            {/* Play Overlay (only when paused) */}
            {!playing && (
                <div
                    className={`absolute inset-0 flex items-center justify-center ${isPip ? 'bg-black/10 group-hover:bg-black/20' : 'bg-black/20 group-hover:bg-black/10'} transition-all cursor-pointer`}
                    onClick={togglePlay}
                >
                    <button className={`flex items-center justify-center rounded-full ${isPip ? 'size-12 opacity-0 group-hover:opacity-100 scale-90 group-hover:scale-100' : 'size-16 hover:scale-105 hover:bg-primary'} bg-white/20 backdrop-blur-sm text-white hover:bg-primary transition-all shadow-lg border border-white/10`}>
                        <Play className={isPip ? 'w-6 h-6' : 'w-8 h-8'} fill="currentColor" />
                    </button>
                </div>
            )}

            {/* Click to pause when playing */}
            {playing && (
                <div
                    className="absolute inset-0 cursor-pointer"
                    onClick={togglePlay}
                />
            )}

            {/* PIP Mode Top Overlay */}
            {isPip && (
                <div className="absolute top-0 inset-x-0 p-2 bg-gradient-to-b from-black/80 to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex justify-between items-start">
                    <span className="bg-red-500/80 text-white text-[10px] font-bold px-1.5 rounded uppercase tracking-wide">Live</span>
                    <button
                        onClick={(e) => { e.stopPropagation(); onTogglePip?.(); }}
                        className="text-white hover:text-primary transition-colors bg-black/40 rounded p-1 backdrop-blur-md"
                    >
                        <Maximize2 className="w-4 h-4" />
                    </button>
                </div>
            )}

            {/* Bottom Controls */}
            <div className={`absolute inset-x-0 bottom-0 ${isPip ? 'p-3' : 'p-4'} bg-gradient-to-t from-black/90 to-transparent`}>
                <div className="flex flex-col gap-1">
                    {/* Progress Bar */}
                    <div
                        className="relative group/slider cursor-pointer h-1.5 bg-white/30 rounded-full flex items-center"
                        onClick={handleSeek}
                    >
                        <div className="absolute h-full bg-primary rounded-full" style={{ width: `${progressPercent}%` }}></div>
                        {!isPip && <div className="absolute size-3 bg-white rounded-full shadow opacity-0 group-hover/slider:opacity-100 transition-opacity" style={{ left: `${progressPercent}%` }}></div>}
                    </div>

                    {/* Button Row */}
                    <div className={`flex items-center justify-between ${isPip ? 'mt-0.5' : 'mt-1'}`}>
                        {isPip ? (
                            <>
                                <span className="text-[10px] font-medium text-white/90">{formatTime(currentTime)} / {formatTime(duration)}</span>
                                <button onClick={(e) => { e.stopPropagation(); setMuted(!muted); }} className="text-white hover:text-primary transition-colors">
                                    {muted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
                                </button>
                            </>
                        ) : (
                            <>
                                <div className="flex items-center gap-4">
                                    <button onClick={togglePlay} className="text-white hover:text-primary transition-colors">
                                        {playing ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
                                    </button>
                                    <button onClick={(e) => { e.stopPropagation(); setMuted(!muted); }} className="text-white hover:text-primary transition-colors">
                                        {muted ? <VolumeX className="w-5 h-5" /> : <Volume2 className="w-5 h-5" />}
                                    </button>
                                    <span className="text-xs font-medium text-white/90">{formatTime(currentTime)} / {formatTime(duration)}</span>
                                </div>
                                <div className="flex items-center gap-4">
                                    <button className="text-white hover:text-primary transition-colors"><Captions className="w-5 h-5" /></button>
                                    <button className="text-white hover:text-primary transition-colors"><Settings className="w-5 h-5" /></button>
                                    <button className="text-white hover:text-primary transition-colors"><Maximize className="w-5 h-5" /></button>
                                </div>
                            </>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}

export default VideoPlayer;
