/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    darkMode: "class",
    theme: {
        extend: {
            colors: {
                "primary": "var(--accent-coral)",
                "primary-light": "var(--accent-coral-light)",
                "background-main": "var(--bg-primary)",
                "background-dark": "var(--bg-primary)",
                "background-light": "var(--bg-secondary)",
                "surface": "var(--bg-secondary)",
                "surface-highlight": "var(--bg-hover)",
                "border-color": "var(--border-color)",
                "border-custom": "var(--border-color)",
                "text-main": "var(--text-primary)",
                "text-secondary": "var(--text-secondary)",
                "text-muted": "var(--text-muted)"
            },
            fontFamily: {
                "display": ["Inter", "sans-serif"]
            },
            borderRadius: { "DEFAULT": "0.25rem", "lg": "0.5rem", "xl": "0.75rem", "full": "9999px" },
            animation: {
                'shimmer': 'shimmer 2s infinite',
                'fade-in': 'fadeIn 0.3s ease-in-out',
                'pipSlideIn': 'pipSlideIn 0.3s ease-out forwards',
            },
            keyframes: {
                shimmer: {
                    '0%': { transform: 'translateX(-100%)' },
                    '100%': { transform: 'translateX(100%)' }
                },
                fadeIn: {
                    '0%': { opacity: '0', transform: 'translateY(10px)' },
                    '100%': { opacity: '1', transform: 'translateY(0)' }
                },
                pipSlideIn: {
                    '0%': { transform: 'translateY(100%)', opacity: '0' },
                    '100%': { transform: 'translateY(0)', opacity: '1' }
                }
            }
        },
    },
    plugins: [
        require('@tailwindcss/forms'),
        require('@tailwindcss/container-queries'),
    ],
}
