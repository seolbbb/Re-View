import React from 'react';
import { Play, Maximize2, Volume2, Captions, Settings, Maximize } from 'lucide-react';

function VideoPlayer({ isPip, onTogglePip, className = "" }) {
    // Default classes for normal view if not provided
    const defaultClasses = "relative w-full rounded-xl overflow-hidden shadow-2xl bg-black aspect-video group";
    const finalClasses = className || defaultClasses;

    return (
        <div className={finalClasses}>
            {/* Background / Video Content */}
            <div
                className={`absolute inset-0 bg-cover bg-center ${isPip ? 'opacity-80' : 'opacity-60'}`}
                data-alt="Abstract biology background showing cells dividing"
                style={{ backgroundImage: 'url("https://lh3.googleusercontent.com/aida-public/AB6AXuCxBTMOz0TAUgSrhG2w0rMKpV4QyGJYiKxmt4VHMog33tJf_bzzN5GYX7mgs98n8u_cHo5Ue5CB2OfccCAwLYz9Dcomrtg4Dlon9CuKMr4QTrrmUp86YPJBwDWuKo7q8Ka8l1B5v14wAHTJV7yw9lVfqxdmscpfDw36CbyeAVBczD5seTl4mqdtY3nS7m_qMjVhn1Z19Px2dQFJdRayPZAOrSuH04q76jWEhRvHNoOb7xWxLYm5UhPM1u6vBRqnKvPJ2ySUjSkNm7Qi")' }}
            ></div>

            {/* Play Overlay */}
            <div className={`absolute inset-0 flex items-center justify-center ${isPip ? 'bg-black/10 group-hover:bg-black/20' : 'bg-black/20 group-hover:bg-black/10'} transition-all`}>
                <button className={`flex items-center justify-center rounded-full ${isPip ? 'size-12 opacity-0 group-hover:opacity-100 scale-90 group-hover:scale-100' : 'size-16 hover:scale-105 hover:bg-primary'} bg-white/20 backdrop-blur-sm text-white hover:bg-primary transition-all shadow-lg border border-white/10`}>
                    <Play className={isPip ? 'w-6 h-6' : 'w-8 h-8'} fill="currentColor" />
                </button>
            </div>

            {/* PIP Mode Top Overlay */}
            {isPip && (
                <div className="absolute top-0 inset-x-0 p-2 bg-gradient-to-b from-black/80 to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex justify-between items-start">
                    <span className="bg-red-500/80 text-white text-[10px] font-bold px-1.5 rounded uppercase tracking-wide">Live</span>
                    <button
                        onClick={onTogglePip}
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
                    <div className="relative group/slider cursor-pointer h-1.5 bg-white/30 rounded-full flex items-center">
                        <div className="absolute h-full bg-primary rounded-full" style={{ width: '28%' }}></div>
                        {!isPip && <div className="absolute left-[28%] size-3 bg-white rounded-full shadow opacity-0 group-hover/slider:opacity-100 transition-opacity"></div>}

                        {/* PIP specific markers */}
                        {isPip && (
                            <>
                                <div className="absolute top-1/2 -translate-y-1/2 left-[5%] w-1 h-1 bg-gray-400 rounded-full z-20 hover:scale-[2] hover:bg-white transition-all cursor-pointer" title="Introduction"></div>
                                <div className="absolute top-1/2 -translate-y-1/2 left-[15%] w-1 h-1 bg-gray-400 rounded-full z-20 hover:scale-[2] hover:bg-white transition-all cursor-pointer" title="Interphase"></div>
                                <div className="absolute top-1/2 -translate-y-1/2 left-[28%] w-2 h-2 bg-white rounded-full z-30 shadow-lg ring-2 ring-primary"></div>
                                <div className="absolute top-1/2 -translate-y-1/2 left-[45%] w-1 h-1 bg-gray-400/50 rounded-full z-10 hover:scale-[2] hover:bg-white transition-all cursor-pointer" title="Prophase"></div>
                            </>
                        )}
                    </div>

                    {/* Button Row */}
                    <div className={`flex items-center justify-between ${isPip ? 'mt-0.5' : 'mt-1'}`}>
                        {isPip ? (
                            <>
                                <span className="text-[10px] font-medium text-white/90">14:20 / 45:00</span>
                                <button className="text-white hover:text-primary transition-colors"><Volume2 className="w-4 h-4" /></button>
                            </>
                        ) : (
                            <>
                                <div className="flex items-center gap-4">
                                    <button className="text-white hover:text-primary transition-colors"><Play className="w-5 h-5" /></button>
                                    <button className="text-white hover:text-primary transition-colors"><Volume2 className="w-5 h-5" /></button>
                                    <span className="text-xs font-medium text-white/90">14:20 / 45:00</span>
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
