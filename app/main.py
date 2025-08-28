from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from playwright.async_api import async_playwright  # ✅ correct import
import os
import json
from typing import Tuple, List

# only if you want to use LLM now (needs pip install openai)
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()  # load values from .env into os.environ



import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(title="The Reader")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ok for local dev
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeOut(BaseModel):
    url: str
    title: str
    answer: str
    summary: List[str]

@app.get("/health")
async def health():
    return {"ok": True}

async def fetch_dom(url: str) -> tuple[str, str]:
    """Open the page (headless) and return (title, html)."""
    async with async_playwright() as p:                 # ✅ async context
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, timeout=30_000, wait_until="domcontentloaded")
        title = await page.title()
        html = await page.content()
        await browser.close()
    return title, html

import re
import trafilatura


def extract_article(html: str) -> str:
    if not html:
        return ""
    text = trafilatura.extract(
        html,
        include_links=False,
        include_comments=False,
        include_images=False,
        favor_recall=False,
    )
    if text and len(text) > 200:
        return text.strip()
    # fallback if empty or too short
    return fallback_extract_body_text(html)


from lxml import html as lxml_html

def fallback_extract_body_text(html_str: str) -> str:
    """
    Very dumb fallback: strip scripts/styles and return visible text from <body>.
    """
    try:
        doc = lxml_html.fromstring(html_str)
        # remove script/style/noscript
        for bad in doc.xpath('//script|//style|//noscript'):
            bad.drop_tree()
        text = doc.text_content() or ""
        # normalize whitespace
        text = re.sub(r"\s+\n", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()
    except Exception:
        return ""


def summarize(text: str, max_bullets: int = 3) -> list[str]:
    """
    Super-dumb summary: take the first few sentences (<= ~500 chars total).
    No LLM yet. Good enough to test the pipe end-to-end.
    """
    if not text:
        return []
    # naive sentence split
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    bullets, total = [], 0
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        bullets.append(s[:240])  # cap a long sentence
        total += len(s)
        if len(bullets) >= max_bullets or total > 500:
            break
    return bullets

def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        s, e = text.find("{"), text.rfind("}")
        return json.loads(text[s:e+1]) if s != -1 and e != -1 else {"answer":"", "summary":[]}

def infer_answer_and_summary(title: str, article_text: str) -> Tuple[str, List[str]]:
    if not article_text:
        return "No article text extracted.", []

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # fallback: current naive approach
        first = article_text.split(".")[0].strip()
        return (first or "(MVP) answer TBD"), summarize(article_text, 3)

    client = OpenAI(api_key=api_key)
    system = (
        "You are a ruthless time-saver. Use ONLY the provided article text. "
        "If the title implies a missing word/name/number and it isn’t stated, say exactly: "
        "\"The article never states it.\" No speculation."
    )
    user = f"""TITLE: {title}

ARTICLE:
{article_text[:12000]}

TASKS:
1) Infer the implied reader question behind the title.
2) Answer it in ONE precise sentence using ONLY the article text.
3) Provide 3–5 bullet summary of hard facts.
Return pure JSON: {{"answer":"...","summary":["...","..."]}}"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )
    data = _extract_json(resp.choices[0].message.content)
    answer = (data.get("answer") or "").strip() or "(MVP) answer TBD"
    bullets = data.get("summary") or summarize(article_text, 3)
    return answer, bullets[:5]


@app.get("/analyze", response_model=AnalyzeOut)
async def analyze(url: Optional[str] = None):
    if not url:
        raise HTTPException(status_code=400, detail="Pass ?url=")
    try:
        title, html = await fetch_dom(url)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Fetch failed: {e!s}")

    text = extract_article(html)
    print(">>> Extracted text preview:", text[:300] if text else "[EMPTY]")

    answer, bullets = infer_answer_and_summary(title, text if text else "")
    if not bullets:
        bullets = ["No main content found."]

    return AnalyzeOut(
        url=url,
        title=title,
        answer=answer,
        summary=bullets,
    )


def infer_answer(title: str, article_text: str) -> str:
    """
    Super-simple MVP:
    - If article_text contains the title words, take the first sentence.
    - Otherwise return a placeholder.
    """
    if not article_text:
        return "No article text extracted."
    sentences = article_text.split(".")
    first_sentence = sentences[0].strip() if sentences else ""
    return first_sentence or "(MVP) answer TBD"
