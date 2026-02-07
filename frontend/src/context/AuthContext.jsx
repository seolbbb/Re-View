import { createContext, useContext, useState, useEffect, useRef } from 'react';
import { supabase } from '../lib/supabase';
import { getMediaTicket } from '../api/videos';

const AuthContext = createContext();

export function AuthProvider({ children }) {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);
    const [mediaTicket, setMediaTicket] = useState(null);
    const [mediaTicketExpAt, setMediaTicketExpAt] = useState(0);
    const mediaTicketTimerRef = useRef(null);

    const refreshMediaTicket = async () => {
        if (mediaTicketTimerRef.current) {
            clearTimeout(mediaTicketTimerRef.current);
            mediaTicketTimerRef.current = null;
        }

        try {
            const data = await getMediaTicket();
            const ticket = data?.ticket;
            const expiresIn = Number(data?.expires_in || 0);

            if (!ticket || !expiresIn) {
                throw new Error('Invalid media ticket response');
            }

            const expAt = Date.now() + expiresIn * 1000;
            setMediaTicket(ticket);
            setMediaTicketExpAt(expAt);

            // Refresh slightly before expiry to keep <video>/<img> URLs working.
            const refreshInMs = Math.max(10_000, expiresIn * 1000 - 30_000);
            mediaTicketTimerRef.current = setTimeout(() => {
                refreshMediaTicket();
            }, refreshInMs);

            return ticket;
        } catch {
            setMediaTicket(null);
            setMediaTicketExpAt(0);
            // Best-effort retry later (network hiccup, cold start, etc.)
            mediaTicketTimerRef.current = setTimeout(() => {
                refreshMediaTicket();
            }, 30_000);
            return null;
        }
    };

    useEffect(() => {
        if (!supabase) {
            setLoading(false);
            return;
        }

        // 현재 세션 확인
        supabase.auth.getSession().then(({ data: { session } }) => {
            setUser(session?.user ?? null);
            setLoading(false);
            if (session?.user) {
                refreshMediaTicket();
            }
        });

        // 인증 상태 변경 리스너
        const { data: { subscription } } = supabase.auth.onAuthStateChange(
            (_event, session) => {
                setUser(session?.user ?? null);
                if (session?.user) {
                    refreshMediaTicket();
                } else {
                    setMediaTicket(null);
                    setMediaTicketExpAt(0);
                    if (mediaTicketTimerRef.current) {
                        clearTimeout(mediaTicketTimerRef.current);
                        mediaTicketTimerRef.current = null;
                    }
                }
            }
        );

        return () => {
            subscription.unsubscribe();
            if (mediaTicketTimerRef.current) {
                clearTimeout(mediaTicketTimerRef.current);
                mediaTicketTimerRef.current = null;
            }
        };
    }, []);

    const signIn = async (email, password) => {
        if (!supabase) throw new Error('Supabase not configured');

        const { data, error } = await supabase.auth.signInWithPassword({
            email,
            password,
        });

        if (error) throw error;
        return data;
    };

    const signUp = async (email, password) => {
        if (!supabase) throw new Error('Supabase not configured');

        const { data, error } = await supabase.auth.signUp({
            email,
            password,
        });

        if (error) throw error;
        return data;
    };

    const signOut = async () => {
        if (!supabase) throw new Error('Supabase not configured');

        const { error } = await supabase.auth.signOut();
        if (error) throw error;
    };

    const signInWithGoogle = async () => {
        if (!supabase) throw new Error('Supabase not configured');

        const { data, error } = await supabase.auth.signInWithOAuth({
            provider: 'google',
            options: {
                redirectTo: window.location.origin,
            },
        });

        if (error) throw error;
        return data;
    };

    return (
        <AuthContext.Provider value={{
            user,
            loading,
            mediaTicket,
            mediaTicketExpAt,
            refreshMediaTicket,
            signIn,
            signUp,
            signOut,
            signInWithGoogle,
        }}>
            {children}
        </AuthContext.Provider>
    );
}

export function useAuth() {
    const context = useContext(AuthContext);
    if (!context) {
        throw new Error('useAuth must be used within an AuthProvider');
    }
    return context;
}
