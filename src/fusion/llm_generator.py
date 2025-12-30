"""
Note generation via Google Gemini 1.5 Flash.

Inputs:
    - contexts: List[FusedContext] containing slide, script, and OCR JSON.
Outputs:
    - str: Markdown-formatted lecture notes.
"""

import os
from typing import List

import google.generativeai as genai
from dotenv import load_dotenv

from src.common.schemas import FusedContext


PROMPT_PREAMBLE = """You are a lecture note taker.
Input includes Image, Script, and OCR JSON Data (BBox + Text).
Use the OCR JSON Data to locate specific charts or formulas in the image.
If you see math symbols in OCR text, convert them to proper LaTeX format.
If you see a chart structure in OCR BBoxes, describe the data trends.
Output formatted Markdown."""


class NoteGenerator:
    """Generate lecture notes from fused slide/audio/OCR context."""

    def __init__(self, model_name: str = "gemini-1.5-flash", api_key: str | None = None) -> None:
        load_dotenv()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set in the environment.")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)

    def _build_prompt(self, contexts: List[FusedContext]) -> str:
        context_blocks = []
        for ctx in contexts:
            block = {
                "slide_image": ctx.slide.image_path,
                "timestamp": ctx.slide.timestamp,
                "script": ctx.script,
                "ocr_json": [item.model_dump() for item in ctx.ocr_data],
            }
            context_blocks.append(block)

        prompt = f"{PROMPT_PREAMBLE}\n\nContexts:\n{context_blocks}"
        return prompt

    def generate_note(self, contexts: List[FusedContext]) -> str:
        """
        Produce lecture notes using Gemini.

        Returns:
            str: Markdown text.
        """
        if not contexts:
            return ""

        prompt = self._build_prompt(contexts)
        response = self.model.generate_content(prompt)
        return response.text or ""
