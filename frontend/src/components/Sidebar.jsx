
import React from 'react';
import { Link } from 'react-router-dom';
import { Video, Home, Library, Settings, History, CheckCircle, ChevronsUpDown } from 'lucide-react';

function Sidebar() {
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
                    <a href="#" className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-[var(--text-primary)] hover:bg-surface-highlight transition-colors group">
                        <Settings className="w-5 h-5 text-gray-400 group-hover:text-[var(--text-primary)]" />
                        <span className="text-sm font-medium">Settings</span>
                    </a>
                </div>
                {/* Recent Lectures */}
                <div className="flex flex-col gap-2 mt-2">
                    <div className="px-3">
                        <p className="text-gray-400 text-xs font-semibold uppercase tracking-wider">Recent Lectures</p>
                    </div>
                    <div className="flex flex-col gap-1">
                        <a href="#" className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-surface-highlight group">
                            <History className="w-5 h-5 text-gray-400 group-hover:text-[var(--text-primary)]" />
                            <div className="flex flex-col truncate">
                                <p className="text-[var(--text-primary)] text-sm font-medium truncate">Mitosis Lecture</p>
                                <p className="text-gray-400 text-xs truncate">Bio 101 ??Just now</p>
                            </div>
                        </a>
                        <a href="#" className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-surface-highlight group">
                            <CheckCircle className="w-5 h-5 text-gray-400 group-hover:text-[var(--text-primary)]" />
                            <div className="flex flex-col truncate">
                                <p className="text-[var(--text-secondary)] text-sm font-medium truncate">World War II Intro</p>
                                <p className="text-gray-400 text-xs truncate">History 202 ??2h ago</p>
                            </div>
                        </a>
                        <a href="#" className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-surface-highlight group">
                            <CheckCircle className="w-5 h-5 text-gray-400 group-hover:text-[var(--text-primary)]" />
                            <div className="flex flex-col truncate">
                                <p className="text-[var(--text-secondary)] text-sm font-medium truncate">Intro to Python</p>
                                <p className="text-gray-400 text-xs truncate">CS 101 ??Yesterday</p>
                            </div>
                        </a>
                    </div>
                </div>
            </div>
            {/* User Profile */}
            <div className="p-4 border-t border-[var(--border-color)]">
                <button className="flex items-center gap-3 w-full hover:bg-surface-highlight p-2 rounded-lg transition-colors text-left">
                    <div className="bg-center bg-no-repeat bg-cover rounded-full size-8 bg-gray-600" data-alt="User profile avatar" style={{ backgroundImage: 'url("https://lh3.googleusercontent.com/aida-public/AB6AXuCzH9Fjp8umL9xTPIZLwLy8QlGvR3RGrOCmj9clOjR__tq5u2wdBqvnpBfxhPprfSU0XD0F0D6R0TpogJs74kJynVDK0LH75JPSgSss1gWQFHNo2boLySpWzvBH8GCEcILsXzR9GfS4Z4sfj6vP7HqtoS-kECbxe5dwIVSygghnYKk6mPWFY6u9tf0mFzVbv7z4CPdxkVAo-MlHzOl53ed7caHtMWwv0p8TGwgpRz_bM7oU1uSv7rYouzxl8aUFoch3dcAwuIo5ICoJ")' }}></div>
                    <div className="flex flex-col">
                        <p className="text-[var(--text-primary)] text-sm font-medium">Alex Student</p>
                        <p className="text-gray-400 text-xs">Free Plan</p>
                    </div>
                    <ChevronsUpDown className="w-[18px] h-[18px] text-gray-400 ml-auto" />
                </button>
            </div>
        </aside>
    );
}

export default Sidebar;
