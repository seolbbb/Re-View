import React from 'react';
import { StickyNote, RefreshCw, Pencil, Minimize2, Maximize2, PlayCircle, Image } from 'lucide-react';

function SummaryPanel({ isExpanded, onToggleExpand }) {
    // Processing Status Component (Inline for convenience)
    const ProcessingStatus = () => (
        <div className="bg-surface rounded-lg p-4 border border-[var(--border-color)] flex flex-col gap-3 shadow-sm mb-4">
            <div className="flex gap-6 justify-between items-center">
                <div className="flex items-center gap-2">
                    <RefreshCw className="w-5 h-5 text-primary animate-spin" />
                    <p className="text-[var(--text-primary)] text-sm font-medium">Analyzing video content & generating notes...</p>
                </div>
                <p className="text-gray-400 text-xs font-mono">75%</p>
            </div>
            <div className="h-1.5 w-full bg-surface-highlight rounded-full overflow-hidden">
                <div className="h-full bg-primary rounded-full relative overflow-hidden" style={{ width: '75%' }}>
                    <div className="absolute inset-0 bg-white/20 w-full animate-[shimmer_2s_infinite] translate-x-[-100%]"></div>
                </div>
            </div>
        </div>
    );

    return (
        <div className={`flex flex-col ${isExpanded ? 'items-center p-6 lg:p-10 overflow-y-auto custom-scrollbar absolute inset-0' : 'gap-4 pb-12 max-w-4xl'}`}>
            <div className={`w-full ${isExpanded ? 'max-w-4xl flex flex-col gap-6 pb-24' : ''}`}>

                {/* Header Section */}
                {isExpanded ? (
                    <div className="flex items-center justify-between border-b border-[var(--border-color)] pb-4 sticky top-0 bg-[var(--bg-primary)]/95 backdrop-blur z-40 py-2">
                        <div className="flex items-center gap-3">
                            <StickyNote className="w-7 h-7 text-primary" />
                            <div className="flex flex-col">
                                <h3 className="text-[var(--text-primary)] text-2xl font-bold tracking-tight">Real-time Summary</h3>
                                <p className="text-gray-400 text-xs">Generating detailed notes from video transcript...</p>
                            </div>
                        </div>
                        <div className="flex items-center gap-3">
                            <button className="text-gray-400 hover:text-[var(--text-primary)] text-xs flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-[var(--border-color)] hover:bg-surface-highlight transition-colors bg-surface">
                                <Pencil className="w-4 h-4" />
                                Edit Notes
                            </button>
                            <button
                                onClick={onToggleExpand}
                                className="text-white bg-primary hover:bg-[var(--accent-coral-dark)] text-xs flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-transparent transition-colors shadow-lg shadow-primary/20"
                            >
                                <Minimize2 className="w-4 h-4" />
                                Collapse View
                            </button>
                        </div>
                    </div>
                ) : (
                    // In Normal view, logic for status vs header
                    <>
                        <ProcessingStatus />
                        <div className="flex items-center justify-between border-b border-[var(--border-color)] pb-2">
                            <div className="flex items-center gap-2">
                                <StickyNote className="w-6 h-6 text-primary" />
                                <h3 className="text-[var(--text-primary)] text-xl font-bold tracking-tight">Real-time Summary</h3>
                            </div>
                            <button
                                onClick={onToggleExpand}
                                className="text-gray-400 hover:text-[var(--text-primary)] text-xs flex items-center gap-1 transition-colors"
                            >
                                <Maximize2 className="w-4 h-4" />
                                Expand
                            </button>
                        </div>
                    </>
                )}

                {/* Status for Expanded (Rendered after header) */}
                {isExpanded && <ProcessingStatus />}

                {/* Timeline Content */}
                <div className={`flex flex-col gap-1 relative pl-2 ${isExpanded ? 'pt-4' : ''}`}>
                    <div className="absolute left-[8.5rem] top-4 bottom-4 w-px bg-border-color hidden md:block"></div>

                    {/* Item 1 */}
                    <div className={`group flex flex-col md:flex-row gap-2 md:gap-8 ${isExpanded ? 'p-4 rounded-xl hover:bg-surface/30 opacity-80 hover:opacity-100' : 'p-3 rounded-lg hover:bg-[var(--bg-hover)] opacity-70'} transition-colors cursor-pointer`}>
                        <div className="md:w-24 shrink-0 flex md:justify-end">
                            <span className="font-mono text-sm text-gray-400 bg-surface/50 px-2 py-0.5 rounded border border-[var(--border-color)] group-hover:border-primary/50 transition-colors h-fit">00:00</span>
                        </div>
                        <div className="flex-1">
                            <h4 className={`text-[var(--text-primary)] font-medium ${isExpanded ? 'mb-2 text-lg' : 'mb-1'}`}>Introduction to Cell Division</h4>
                            <ul className={`list-disc list-outside ml-4 text-[var(--text-secondary)] ${isExpanded ? 'text-base space-y-2' : 'text-sm space-y-1'} leading-relaxed`}>
                                <li>
                                    {isExpanded && <span className="text-xs font-mono text-primary font-medium mr-1">[00:45]</span>}
                                    Professor Smith introduces the concept of {isExpanded ? <span className="text-[var(--text-secondary)] border-b border-gray-600 border-dashed cursor-help">cellular replication</span> : 'cellular replication'}...
                                </li>
                                <li>Overview of the lecture structure{isExpanded ? ': Mitosis vs Meiosis' : '.'}</li>
                            </ul>
                        </div>
                    </div>

                    {/* Item 2 (Active) */}
                    {isExpanded ? (
                        // Expanded Active Item (Complex)
                        <div className="group flex flex-col md:flex-row gap-2 md:gap-8 p-6 rounded-xl bg-surface border-l-4 border-primary shadow-lg transition-all cursor-pointer relative overflow-hidden my-2">
                            <div className="absolute top-0 right-0 p-2 opacity-10">
                                <PlayCircle className="w-24 h-24" />
                            </div>
                            <div className="md:w-24 shrink-0 flex md:justify-end z-10">
                                <span className="font-mono text-sm text-primary bg-primary/10 px-2 py-0.5 rounded font-bold h-fit border border-primary/20">14:20</span>
                            </div>
                            <div className="flex-1 z-10">
                                <h4 className="text-[var(--text-primary)] font-bold mb-2 text-xl flex items-center gap-2">
                                    Phases of Mitosis: Prophase
                                    <span className="bg-primary text-white text-[10px] px-2 py-0.5 rounded-full uppercase tracking-wider font-bold">Current</span>
                                </h4>
                                <p className="text-[var(--text-secondary)] mb-3 text-base">The first and longest phase of mitosis.</p>
                                <ul className="list-disc list-outside ml-4 text-[var(--text-secondary)] text-base leading-relaxed space-y-2">
                                    <li><strong className="text-[var(--text-primary)]">Chromatin Condensation:</strong> Loose chromatin coils into visible chromosomes.</li>
                                    <li><strong className="text-[var(--text-primary)]">Nuclear Envelope:</strong> The membrane surrounding the nucleus begins to break down.</li>
                                    <li><strong className="text-[var(--text-primary)]">Centrosome Movement:</strong> Centrosomes begin migrating to opposite poles.</li>
                                </ul>
                                <div className="flex gap-2 mt-4">
                                    <button className="text-xs bg-[var(--bg-primary)] hover:bg-surface-highlight border border-[var(--border-color)] text-[var(--text-primary)] px-3 py-1.5 rounded-md transition-colors flex items-center gap-1">
                                        <Image className="w-4 h-4" /> View Diagram
                                    </button>
                                </div>
                            </div>
                        </div>
                    ) : (
                        // Normal Active Item
                        <div className="group flex flex-col md:flex-row gap-2 md:gap-8 p-3 rounded-lg bg-surface border-l-2 border-primary shadow-sm transition-all cursor-pointer">
                            <div className="md:w-24 shrink-0 flex md:justify-end">
                                <span className="font-mono text-sm text-primary bg-primary/10 px-2 py-0.5 rounded font-bold">14:20</span>
                            </div>
                            <div className="flex-1">
                                <h4 className="text-[var(--text-primary)] font-semibold mb-1">Phases of Mitosis: Prophase</h4>
                                <ul className="list-disc list-outside ml-4 text-[var(--text-secondary)] text-sm leading-relaxed space-y-1">
                                    <li><strong className="text-[var(--text-primary)]">Prophase</strong> is the first stage of cell division.</li>
                                    <li>Chromatin condenses into chromosomes...</li>
                                    <li>Centrosomes move to opposite poles of the cell.</li>
                                </ul>
                            </div>
                        </div>
                    )}

                    {/* Item 3 (Future) */}
                    <div className={`group flex flex-col md:flex-row gap-2 md:gap-8 ${isExpanded ? 'p-4 rounded-xl hover:bg-surface/30 opacity-50' : 'p-3 rounded-lg hover:bg-[var(--bg-hover)] opacity-50'} transition-colors cursor-pointer`}>
                        <div className="md:w-24 shrink-0 flex md:justify-end">
                            <span className="font-mono text-sm text-gray-400 bg-surface/50 px-2 py-0.5 rounded border border-[var(--border-color)] h-fit">22:45</span>
                        </div>
                        <div className="flex-1">
                            <h4 className={`text-[var(--text-primary)] font-medium ${isExpanded ? 'mb-2 text-lg' : 'mb-1'}`}>Metaphase & Anaphase</h4>
                            <div className={isExpanded ? 'space-y-3' : ''}>
                                <div className="h-4 w-3/4 bg-[var(--border-color)] rounded animate-pulse mt-1"></div>
                                <div className="h-4 w-1/2 bg-[var(--border-color)] rounded animate-pulse mt-2"></div>
                                {isExpanded && <div className="h-4 w-5/6 bg-[var(--border-color)] rounded animate-pulse"></div>}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default SummaryPanel;
