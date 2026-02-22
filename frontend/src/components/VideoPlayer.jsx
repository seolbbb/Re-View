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

    const defaultClasses = "relative w-full overflow-hidden shadow-2xl bg-black aspect-video group";
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
        const x = e.clientX - rect.left;
        const pct = Math.max(0, Math.min(1, x / rect.width));
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
            if (showSettings && settingsRef.current && !settingsRef.current.contains(e.target)) {
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
                    v.play().catch(() => { });
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
        const onPlay = () => setPlaying(true);
        const onPause = () => setPlaying(false);
        v.addEventListener('timeupdate', onTime);
        v.addEventListener('loadedmetadata', onDur);
        v.addEventListener('ended', onEnd);
        v.addEventListener('play', onPlay);
        v.addEventListener('pause', onPause);

        if (v.readyState >= 1) {
            setDuration(v.duration);
            restorePlayback();
        }

        return () => {
            v.removeEventListener('timeupdate', onTime);
            v.removeEventListener('loadedmetadata', onDur);
            v.removeEventListener('ended', onEnd);
            v.removeEventListener('play', onPlay);
            v.removeEventListener('pause', onPause);
        };
    }, [streamUrl]);

    // Tooltip position: use fixed positioning to escape overflow-hidden
    const getTooltipFixedStyle = () => {
        const bar = progressBarRef.current;
        if (!hoverInfo || !bar) return { position: 'fixed', visibility: 'hidden', pointerEvents: 'none' };

        try {
            const barRect = bar.getBoundingClientRect();
            if (!barRect) return { position: 'fixed', visibility: 'hidden', pointerEvents: 'none' };

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
                pointerEvents: 'none',
                zIndex: 9999,
            };
        } catch (err) {
            return { position: 'fixed', visibility: 'hidden', pointerEvents: 'none' };
        }
    };

    return (
        <div ref={containerRef} className={finalClasses}>
            {streamUrl ? (
                <video
                    ref={videoRef}
                    src={streamUrl}
                    className="absolute inset-0 w-full h-full object-cover"
                    muted={muted}
                    playsInline
                />
            ) : (
                <div className="absolute inset-0 bg-gray-900 flex items-center justify-center">
                    <p className="text-gray-500 text-sm">No video source</p>
                </div>
            )}

            {/* Click to pause when playing */}
            {playing && (
                <div
                    className="absolute inset-0 cursor-pointer"
                    onClick={togglePlay}
                />
            )}

            {/* Top Controls Overlay (Live Label only in PIP) */}
            {isPip && (
                <div className="absolute top-0 inset-x-0 p-2 bg-gradient-to-b from-black/80 to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex justify-between items-start z-10">
                    <span className="bg-red-500/80 text-white text-[10px] font-bold px-1.5 rounded uppercase tracking-wide">Live</span>
                </div>
            )}

            {/* Bottom Controls */}
            <div className={`video-controls absolute inset-x-0 bottom-0 ${isPip ? 'p-3' : 'pb-6 pt-12'} bg-gradient-to-t from-black/90 via-black/40 to-transparent z-10`}>
                <div className="flex flex-col gap-0">
                    {/* Progress Bar */}
                    <div
                        ref={progressBarRef}
                        className="relative group/slider cursor-pointer h-1.5 bg-white/30 flex items-center mb-1"
                        onClick={handleSeek}
                        onMouseMove={handleProgressHover}
                        onMouseLeave={handleProgressLeave}
                    >
                        <div className="absolute h-full bg-primary" style={{ width: `${progressPercent}%` }}></div>
                        {hoverInfo && !isPip && (
                            <div className="absolute top-0 h-full bg-white/30" style={{ left: 0, width: `${hoverInfo.x}px` }}></div>
                        )}
                    </div>

                    {/* Button Row */}
                    <div className="flex items-center justify-between py-0.5">
                        {/* Left Controls */}
                        <div className="flex items-center">
                            <div className={isPip ? "w-0.5" : "w-1"} />
                            <div className="flex items-center gap-1.5 sm:gap-2">
                                <button onClick={togglePlay} className="text-white hover:text-primary transition-colors">
                                    {playing ? <Pause className={isPip ? "w-3.5 h-3.5" : "w-4 h-4"} /> : <Play className={isPip ? "w-3.5 h-3.5" : "w-4 h-4"} />}
                                </button>

                                {/* Volume button + slider (Mini version for PIP) */}
                                <div
                                    className={`flex items-center ${isPip ? 'h-8 rounded-lg' : 'h-10 rounded-xl'} border border-white/10 bg-white/5 transition-all duration-300 overflow-hidden`}
                                    style={{ width: showVolumeSlider ? (isPip ? '120px' : '160px') : (isPip ? '36px' : '44px') }}
                                    onMouseEnter={handleVolumeAreaEnter}
                                    onMouseLeave={handleVolumeAreaLeave}
                                >
                                    <button
                                        className={`flex items-center justify-center ${isPip ? 'w-8' : 'w-11'} shrink-0 text-white hover:text-primary transition-colors`}
                                        onClick={toggleMute}
                                    >
                                        <VolumeIcon className={isPip ? "w-3.5 h-3.5" : "w-4 h-4"} />
                                    </button>
                                    <div className={`flex-1 pr-4 flex items-center h-full transition-opacity duration-300 ${showVolumeSlider ? 'opacity-100' : 'opacity-0'}`}>
                                        <div
                                            className="relative w-full h-1 bg-white/20 rounded-full cursor-pointer"
                                            onClick={handleVolumeChange}
                                        >
                                            <div
                                                className="absolute left-0 top-0 h-full bg-primary rounded-full"
                                                style={{ width: `${(muted ? 0 : volume) * 100}%` }}
                                            ></div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Right Controls */}
                        <div className="flex items-center">
                            <div className="flex items-center gap-1.5 sm:gap-2">
                                <div className={`flex items-center ${isPip ? 'h-8 px-2.5 rounded-lg' : 'h-10 px-4 rounded-xl'} border border-white/10 bg-white/5 shadow-sm`}>
                                    <span className={`${isPip ? 'text-[11px]' : 'text-sm'} font-bold leading-none tracking-tight text-white/90 tabular-nums`}>
                                        {formatTime(currentTime)} <span className="text-white/30 mx-1">/</span> {formatTime(duration)}
                                    </span>
                                </div>

                                {/* Settings (Simplified icon for PIP) */}
                                <div className="relative flex items-center" ref={settingsRef}>
                                    <button
                                        onClick={(e) => { e.stopPropagation(); setShowSettings(!showSettings); }}
                                        className="text-white hover:text-primary transition-colors"
                                    >
                                        <Settings className={isPip ? "w-3.5 h-3.5" : "w-4 h-4"} />
                                    </button>

                                    {showSettings && (
                                        <div className="absolute bottom-full right-0 mb-3 bg-[var(--bg-primary)] backdrop-blur-md rounded-xl shadow-2xl border border-[var(--border-color)] overflow-hidden min-w-[140px] animate-fade-in z-20">
                                            <div className="px-4 py-2.5 border-b border-[var(--border-color)] bg-surface-highlight/5">
                                                <span className="text-[10px] text-[var(--text-secondary)] font-bold uppercase tracking-wider block text-center">재생 속도</span>
                                            </div>
                                            <div className="py-1">
                                                {PLAYBACK_RATES.map((rate) => (
                                                    <button
                                                        key={rate}
                                                        onClick={(e) => { e.stopPropagation(); changePlaybackRate(rate); }}
                                                        className={`w-full flex items-center justify-center px-4 py-2 relative text-sm transition-all ${playbackRate === rate ? 'text-primary bg-primary/10 font-bold' : 'text-[var(--text-primary)] hover:bg-surface-highlight'}`}
                                                    >
                                                        <span>{rate === 1 ? '1.0x' : `${rate}x`}</span>
                                                        {playbackRate === rate && <Check className="absolute right-3 w-3.5 h-3.5" />}
                                                    </button>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                </div>

                                {!isPip && (
                                    <button
                                        onClick={toggleFullscreen}
                                        className="text-white hover:text-primary transition-colors"
                                    >
                                        {isFullscreen ? <Minimize className="w-4 h-4" /> : <Maximize className="w-4 h-4" />}
                                    </button>
                                )}
                            </div>
                            <div className={isPip ? "w-0.5" : "w-1"} />
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default VideoPlayer;
