import React, { useRef, useState, useEffect, useCallback } from 'react';
import { Play, Pause, Maximize2, Volume2, VolumeX, Volume1, Settings, Maximize, Minimize, Check } from 'lucide-react';
import { getVideoStreamUrl } from '../api/videos';
import { useAuth } from '../context/AuthContext';

function formatTime(sec) {
    if (!sec || isNaN(sec)) return '0:00';
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${String(s).padStart(2, '0')}`;
}

function VideoPlayer({ isPip, onTogglePip, videoId, className = "", videoElRef, playbackRestore, onTimeUpdate }) {
    const internalRef = useRef(null);
    const videoRef = videoElRef || internalRef;
    const { mediaTicket } = useAuth();
    const [playing, setPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);
    const [muted, setMuted] = useState(false);
    const [volume, setVolume] = useState(1);
    const [showVolumeSlider, setShowVolumeSlider] = useState(false);
    const [isFullscreen, setIsFullscreen] = useState(false);
    const [playbackRate, setPlaybackRate] = useState(1);
    const [showSettings, setShowSettings] = useState(false);
    const volumeTimerRef = useRef(null);
    const containerRef = useRef(null);
    const settingsRef = useRef(null);
    const progressPercent = duration > 0 ? (currentTime / duration) * 100 : 0;

    const PLAYBACK_RATES = [0.5, 0.75, 1, 1.25, 1.5, 2];

    // Thumbnail preview state
    const [hoverInfo, setHoverInfo] = useState(null); // { x, time }
    const [thumbUrl, setThumbUrl] = useState(null);
    const thumbVideoRef = useRef(null);
    const thumbCanvasRef = useRef(null);
    const progressBarRef = useRef(null);
    const pendingSeekRef = useRef(null);
    const seekingRef = useRef(false);

    const defaultClasses = "relative w-full rounded-xl overflow-hidden shadow-2xl bg-black aspect-video group";
    const finalClasses = className || defaultClasses;

    const desiredStreamUrl = videoId && mediaTicket ? getVideoStreamUrl(videoId, mediaTicket) : null;
    const [streamUrl, setStreamUrl] = useState(null);

    useEffect(() => {
        if (!desiredStreamUrl) {
            setStreamUrl(null);
            return;
        }

        if (!streamUrl) {
            setStreamUrl(desiredStreamUrl);
            return;
        }

        if (streamUrl === desiredStreamUrl) return;

        // When media tickets rotate, <video src> must be updated or future range requests
        // can fail with 401. Preserve playback position when swapping URLs.
        const v = videoRef.current;
        const restore = {
            time: v?.currentTime ?? 0,
            wasPlaying: v ? !v.paused && !v.ended : false,
            playbackRate: v?.playbackRate ?? 1,
        };

        if (v) {
            v.addEventListener(
                'loadedmetadata',
                () => {
                    try {
                        if (restore.playbackRate) v.playbackRate = restore.playbackRate;
                    } catch {
                        // ignore
                    }

                    try {
                        if (typeof restore.time === 'number' && restore.time > 0) {
                            v.currentTime = restore.time;
                        }
                    } catch {
                        // ignore
                    }

                    if (restore.wasPlaying) {
                        v.play().catch(() => { });
                    }
                },
                { once: true }
            );
        }

        setStreamUrl(desiredStreamUrl);
    }, [desiredStreamUrl, streamUrl, videoRef]);

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

    // --- Volume ---
    const applyVolume = useCallback((val) => {
        const v = videoRef.current;
        if (!v) return;
        const clamped = Math.max(0, Math.min(1, val));
        v.volume = clamped;
        setVolume(clamped);
        if (clamped === 0) {
            setMuted(true);
            v.muted = true;
        } else if (muted) {
            setMuted(false);
            v.muted = false;
        }
    }, [muted]);

    const toggleMute = useCallback((e) => {
        e.stopPropagation();
        const v = videoRef.current;
        if (!v) return;
        if (muted) {
            v.muted = false;
            setMuted(false);
            if (volume === 0) {
                v.volume = 0.5;
                setVolume(0.5);
            }
        } else {
            v.muted = true;
            setMuted(true);
        }
    }, [muted, volume]);

    const handleVolumeChange = useCallback((e) => {
        e.stopPropagation();
        const rect = e.currentTarget.getBoundingClientRect();
        const y = rect.bottom - e.clientY;
        const pct = Math.max(0, Math.min(1, y / rect.height));
        applyVolume(pct);
    }, [applyVolume]);

    const handleVolumeAreaEnter = useCallback(() => {
        if (volumeTimerRef.current) clearTimeout(volumeTimerRef.current);
        setShowVolumeSlider(true);
    }, []);

    const handleVolumeAreaLeave = useCallback(() => {
        volumeTimerRef.current = setTimeout(() => setShowVolumeSlider(false), 300);
    }, []);

    const VolumeIcon = muted || volume === 0 ? VolumeX : volume < 0.5 ? Volume1 : Volume2;

    // --- Fullscreen ---
    const toggleFullscreen = useCallback((e) => {
        e.stopPropagation();
        if (document.fullscreenElement) {
            document.exitFullscreen();
        } else {
            containerRef.current?.requestFullscreen();
        }
    }, []);

    useEffect(() => {
        const handleFullscreenChange = () => {
            setIsFullscreen(!!document.fullscreenElement);
        };
        document.addEventListener('fullscreenchange', handleFullscreenChange);
        return () => {
            document.removeEventListener('fullscreenchange', handleFullscreenChange);
        };
    }, []);

    // --- Playback Rate ---
    const changePlaybackRate = useCallback((rate) => {
        const v = videoRef.current;
        if (!v) return;
        v.playbackRate = rate;
        setPlaybackRate(rate);
        setShowSettings(false);
    }, []);

    // Close settings menu on outside click
    useEffect(() => {
        if (!showSettings) return;
        const handleClickOutside = (e) => {
            if (settingsRef.current && !settingsRef.current.contains(e.target)) {
                setShowSettings(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => {
            document.removeEventListener('mousedown', handleClickOutside);
        };
    }, [showSettings]);

    // --- Thumbnail preview ---
    useEffect(() => {
        if (isPip || !streamUrl) return;

        const thumbVideo = document.createElement('video');
        thumbVideo.src = streamUrl;
        thumbVideo.crossOrigin = 'anonymous';
        thumbVideo.preload = 'metadata';
        thumbVideo.muted = true;
        thumbVideo.playsInline = true;
        thumbVideoRef.current = thumbVideo;

        const canvas = document.createElement('canvas');
        canvas.width = 160;
        canvas.height = 90;
        thumbCanvasRef.current = canvas;

        const onSeeked = () => {
            try {
                const ctx = canvas.getContext('2d');
                ctx.drawImage(thumbVideo, 0, 0, canvas.width, canvas.height);
                const url = canvas.toDataURL('image/jpeg', 0.6);
                setThumbUrl(url);
            } catch {
                // CORS or other errors
            }
            seekingRef.current = false;
            // If another seek was requested while we were busy, process it now
            if (pendingSeekRef.current !== null) {
                const next = pendingSeekRef.current;
                pendingSeekRef.current = null;
                seekingRef.current = true;
                thumbVideo.currentTime = next;
            }
        };

        thumbVideo.addEventListener('seeked', onSeeked);

        return () => {
            thumbVideo.removeEventListener('seeked', onSeeked);
            thumbVideo.src = '';
            thumbVideoRef.current = null;
            thumbCanvasRef.current = null;
            seekingRef.current = false;
            pendingSeekRef.current = null;
        };
    }, [streamUrl, isPip]);

    const requestThumbnail = useCallback((time) => {
        const thumbVideo = thumbVideoRef.current;
        if (!thumbVideo || isPip) return;

        if (seekingRef.current) {
            // Already seeking — queue the latest request
            pendingSeekRef.current = time;
        } else {
            seekingRef.current = true;
            pendingSeekRef.current = null;
            thumbVideo.currentTime = time;
        }
    }, [isPip]);

    const handleProgressHover = useCallback((e) => {
        if (!duration || isPip) return;
        const rect = progressBarRef.current?.getBoundingClientRect();
        if (!rect) return;
        const x = e.clientX - rect.left;
        const pct = Math.max(0, Math.min(1, x / rect.width));
        const time = pct * duration;

        setHoverInfo({ x, time });
        requestThumbnail(time);
    }, [duration, isPip, requestThumbnail]);

    const handleProgressLeave = useCallback(() => {
        setHoverInfo(null);
        setThumbUrl(null);
        pendingSeekRef.current = null;
    }, []);

    // --- Main video events ---
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

    // Tooltip position: use fixed positioning to escape overflow-hidden
    const getTooltipFixedStyle = () => {
        if (!hoverInfo || !progressBarRef.current) return { position: 'fixed', visibility: 'hidden' };
        const barRect = progressBarRef.current.getBoundingClientRect();
        const tooltipWidth = thumbUrl ? 164 : 56;
        const halfW = tooltipWidth / 2;
        // Clamp so tooltip doesn't overflow beyond the progress bar edges
        const barX = barRect.left + Math.max(halfW, Math.min(hoverInfo.x, barRect.width - halfW));
        const barY = barRect.top;
        return {
            position: 'fixed',
            left: `${barX}px`,
            top: `${barY}px`,
            transform: 'translate(-50%, -100%)',
            paddingBottom: '12px',
        };
    };

    return (
        <div ref={containerRef} className={finalClasses}>
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
                        ref={progressBarRef}
                        className="relative group/slider cursor-pointer h-1.5 bg-white/30 rounded-full flex items-center"
                        onClick={handleSeek}
                        onMouseMove={handleProgressHover}
                        onMouseLeave={handleProgressLeave}
                    >
                        <div className="absolute h-full bg-primary rounded-full" style={{ width: `${progressPercent}%` }}></div>
                        {!isPip && <div className="absolute size-3 bg-white rounded-full shadow opacity-0 group-hover/slider:opacity-100 transition-opacity" style={{ left: `${progressPercent}%` }}></div>}
                    </div>

                    {/* Thumbnail / Time Preview Tooltip (fixed position to escape overflow-hidden) */}
                    {!isPip && hoverInfo && (
                        <div
                            className="pointer-events-none z-[9999] flex flex-col items-center"
                            style={getTooltipFixedStyle()}
                        >
                            {thumbUrl && (
                                <div className="rounded-md overflow-hidden border border-white/20 shadow-lg mb-1" style={{ width: 160, height: 90 }}>
                                    <img
                                        src={thumbUrl}
                                        alt=""
                                        className="block w-full h-full"
                                    />
                                </div>
                            )}
                            <span className="bg-black/85 text-white text-[11px] font-mono px-2 py-0.5 rounded shadow whitespace-nowrap">
                                {formatTime(hoverInfo.time)}
                            </span>
                        </div>
                    )}

                    {/* Button Row */}
                    <div className={`flex items-center justify-between ${isPip ? 'mt-0.5' : 'mt-1'}`}>
                        {isPip ? (
                            <>
                                <span className="text-[10px] font-medium text-white/90">{formatTime(currentTime)} / {formatTime(duration)}</span>
                                <button onClick={toggleMute} className="text-white hover:text-primary transition-colors">
                                    <VolumeIcon className="w-4 h-4" />
                                </button>
                            </>
                        ) : (
                            <>
                                <div className="flex items-center gap-4">
                                    <button onClick={togglePlay} className="text-white hover:text-primary transition-colors">
                                        {playing ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
                                    </button>

                                    {/* Volume button + slider */}
                                    <div
                                        className="relative flex items-center"
                                        onMouseEnter={handleVolumeAreaEnter}
                                        onMouseLeave={handleVolumeAreaLeave}
                                    >
                                        <button onClick={toggleMute} className="text-white hover:text-primary transition-colors">
                                            <VolumeIcon className="w-5 h-5" />
                                        </button>

                                        {/* Vertical volume slider */}
                                        <div className={`absolute bottom-full left-1/2 -translate-x-1/2 mb-2 transition-all origin-bottom ${showVolumeSlider ? 'opacity-100 scale-100' : 'opacity-0 scale-95 pointer-events-none'}`}>
                                            <div className="bg-black/90 backdrop-blur-md rounded-lg px-2.5 py-3 shadow-xl border border-white/10 flex flex-col items-center gap-1.5">
                                                <span className="text-[10px] text-white/70 font-mono tabular-nums">{Math.round((muted ? 0 : volume) * 100)}</span>
                                                <div
                                                    className="relative w-1.5 h-24 bg-white/20 rounded-full cursor-pointer"
                                                    onClick={handleVolumeChange}
                                                    onMouseDown={(e) => {
                                                        e.preventDefault();
                                                        e.stopPropagation();
                                                        handleVolumeChange(e);
                                                        const onMove = (ev) => {
                                                            const rect = e.currentTarget.getBoundingClientRect();
                                                            const y = rect.bottom - ev.clientY;
                                                            const pct = Math.max(0, Math.min(1, y / rect.height));
                                                            applyVolume(pct);
                                                        };
                                                        const onUp = () => {
                                                            window.removeEventListener('mousemove', onMove);
                                                            window.removeEventListener('mouseup', onUp);
                                                        };
                                                        window.addEventListener('mousemove', onMove);
                                                        window.addEventListener('mouseup', onUp);
                                                    }}
                                                >
                                                    <div
                                                        className="absolute bottom-0 left-0 w-full bg-primary rounded-full transition-[height] duration-75"
                                                        style={{ height: `${(muted ? 0 : volume) * 100}%` }}
                                                    ></div>
                                                    <div
                                                        className="absolute left-1/2 -translate-x-1/2 w-3 h-3 bg-white rounded-full shadow-md border border-white/30 transition-[bottom] duration-75"
                                                        style={{ bottom: `calc(${(muted ? 0 : volume) * 100}% - 6px)` }}
                                                    ></div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    <span className="text-xs font-medium text-white/90">{formatTime(currentTime)} / {formatTime(duration)}</span>
                                </div>
                                <div className="flex items-center gap-4">
                                    {/* Settings button with playback speed menu */}
                                    <div className="relative flex items-center" ref={settingsRef}>
                                        <button
                                            onClick={(e) => { e.stopPropagation(); setShowSettings(!showSettings); }}
                                            className="text-white hover:text-primary transition-colors"
                                        >
                                            <Settings className="w-5 h-5" />
                                        </button>

                                        {/* Playback speed menu */}
                                        {showSettings && (
                                            <div className="absolute bottom-full right-0 mb-2 bg-black/90 backdrop-blur-md rounded-lg shadow-xl border border-white/10 overflow-hidden min-w-[140px]">
                                                <div className="px-3 py-2 border-b border-white/10">
                                                    <span className="text-xs text-white/70 font-medium">재생 속도</span>
                                                </div>
                                                <div className="py-1">
                                                    {PLAYBACK_RATES.map((rate) => (
                                                        <button
                                                            key={rate}
                                                            onClick={(e) => { e.stopPropagation(); changePlaybackRate(rate); }}
                                                            className={`w-full flex items-center justify-between px-3 py-1.5 text-sm transition-colors ${playbackRate === rate ? 'text-primary bg-white/10' : 'text-white hover:bg-white/10'}`}
                                                        >
                                                            <span>{rate === 1 ? '1x (기본)' : `${rate}x`}</span>
                                                            {playbackRate === rate && <Check className="w-4 h-4" />}
                                                        </button>
                                                    ))}
                                                </div>
                                            </div>
                                        )}
                                    </div>

                                    {/* Fullscreen button */}
                                    <button
                                        onClick={toggleFullscreen}
                                        className="text-white hover:text-primary transition-colors"
                                    >
                                        {isFullscreen ? <Minimize className="w-5 h-5" /> : <Maximize className="w-5 h-5" />}
                                    </button>
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
