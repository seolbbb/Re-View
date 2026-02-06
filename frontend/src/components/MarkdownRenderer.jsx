import React, { useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from 'rehype-raw';
import 'katex/dist/katex.min.css';

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
 * Math ($…$, $$…$$) and code blocks are protected from conversion.
 */
function preprocessMarkdown(text) {
    if (!text) return text;

    const saved = [];
    let idx = 0;
    const protect = (match) => {
        const ph = `\x00P${idx++}\x00`;
        saved.push({ ph, val: match });
        return ph;
    };

    // Protect content that must not be touched
    text = text.replace(/```[\s\S]*?```/g, protect);    // fenced code
    text = text.replace(/`[^`]+`/g, protect);            // inline code
    text = text.replace(/\$\$([\s\S]+?)\$\$/g, protect); // display math
    text = text.replace(/\$([^\$\n]+?)\$/g, protect);    // inline math

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

function MarkdownRenderer({ children, className = '' }) {
    if (!children) return null;

    const processed = useMemo(() => preprocessMarkdown(children), [children]);

    return (
        <div className={`markdown-body ${className}`}>
            <ReactMarkdown
                remarkPlugins={[remarkMath]}
                rehypePlugins={[rehypeKatex, rehypeRaw]}
            >
                {processed}
            </ReactMarkdown>
        </div>
    );
}

export default MarkdownRenderer;
