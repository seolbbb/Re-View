import React, { useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeRaw from 'rehype-raw';
import katex from 'katex';
import 'katex/dist/katex.min.css';

const IS_DEV = Boolean(import.meta.env && import.meta.env.DEV);

/**
 * Pre-process raw text so that markdown emphasis (**bold**, *italic*)
 * renders reliably regardless of surrounding punctuation or scripts.
 *
 * CommonMark's right-flanking delimiter rules cause emphasis to fail
 * when the closing ** is preceded by punctuation (e.g. `)`) and
 * followed by non-Latin characters (e.g. Korean).
 *
 * We bypass the emphasis parser entirely by converting **…** / *…*
 * to <strong>/<em> HTML tags, then use rehype-raw to parse them.
 * Math ($…$, $$…$$) is pre-rendered to HTML via katex.renderToString()
 * and protected from further processing. Code blocks are also protected.
 */
function preprocessMarkdown(text) {
    if (!text) return text;

    // Repair control characters that can be introduced when LaTeX backslashes are not
    // JSON-escaped in model outputs (e.g. "\beta" in JSON becomes backspace + "eta").
    // This makes KaTeX parsing reliable even if the upstream JSON escaping was imperfect.
    const BS = '\b';
    const FF = '\f';
    const TAB = '\t';
    const hadControlEscapes = text.includes(BS) || text.includes(FF) || text.includes(TAB);
    // Avoid control-character regexes (eslint no-control-regex).
    text = text
        .split(BS).join('\\b')
        .split(FF).join('\\f')
        .split(TAB).join('\\t');
    if (IS_DEV && hadControlEscapes) console.warn('[MarkdownRenderer] Repaired control characters in input.');

    const saved = [];
    let idx = 0;
    const protect = (html) => {
        const ph = `\x00P${idx++}\x00`;
        saved.push({ ph, val: html });
        return ph;
    };

    // Protect content that must not be touched
    text = text.replace(/```[\s\S]*?```/g, protect);    // fenced code
    text = text.replace(/`[^`]+`/g, protect);            // inline code

    // Pre-render display math ($$...$$) with KaTeX, then protect
    text = text.replace(/\$\$([\s\S]+?)\$\$/g, (_m, inner) => {
        try {
            return protect(katex.renderToString(inner.trim(), { displayMode: true, throwOnError: false, output: 'html' }));
        } catch (err) {
            if (IS_DEV) console.warn('[MarkdownRenderer] KaTeX display math failed:', inner, err);
            return protect(_m);
        }
    });

    // Pre-render inline math ($...$) with KaTeX, then protect
    text = text.replace(/\$([^$\n]+?)\$/g, (_m, inner) => {
        try {
            return protect(katex.renderToString(inner.trim(), { displayMode: false, throwOnError: false, output: 'html' }));
        } catch (err) {
            if (IS_DEV) console.warn('[MarkdownRenderer] KaTeX inline math failed:', inner, err);
            return protect(_m);
        }
    });

    // **…** → <strong>…</strong>  (collapse internal newlines)
    text = text.replace(/\*\*([\s\S]+?)\*\*/g, (_m, inner) => {
        const collapsed = inner.replace(/\n+/g, ' ').trim();
        return `<strong>${collapsed}</strong>`;
    });

    // *…* → <em>…</em>  (single-line only, avoid matching **)
    text = text.replace(/(?<!\*)\*(?!\*)([^*\n]+?)\*(?!\*)/g, (_m, inner) => {
        return `<em>${inner.trim()}</em>`;
    });

    // Restore protected blocks
    for (const { ph, val } of saved) {
        text = text.split(ph).join(val);
    }

    return text;
}

function MarkdownRenderer({ children, className = '', inline = false }) {
    const raw = children == null ? '' : String(children);
    const processed = useMemo(() => preprocessMarkdown(raw), [raw]);
    if (!raw) return null;

    return (
        <div
            className={`markdown-body ${className}`}
            style={inline ? { display: 'inline' } : undefined}
        >
            <ReactMarkdown
                rehypePlugins={[rehypeRaw]}
            >
                {processed}
            </ReactMarkdown>
        </div>
    );
}

export default MarkdownRenderer;
