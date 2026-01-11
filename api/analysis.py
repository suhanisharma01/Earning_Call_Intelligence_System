from __future__ import annotations
import os
import re
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple
from datetime import datetime, timezone

import re
from typing import List, Dict, Any

BOILERPLATE_PATTERNS = [
    r"thank you for standing by\.?",
    r"this is the conference operator\.?",
    r"welcome to .*?results conference call\.?",
    r"\[operator instructions\].*?",
    r"forward-looking statements?.*?(?=\.|\n)",
    r"non-gaap.*?(?=\.|\n)",
    r"please see the cautionary statements.*?(?=\.|\n)",
    r"available on our website.*?(?=\.|\n)",
    r"i(?:'|’)ll now turn the call over to.*?(?=\.|\n)",
    r"i would now like to turn the conference over to.*?(?=\.|\n)",
]

SPEAKER_LINE = re.compile(r"^[A-Z][A-Za-z.\' ]{2,80}\s[—-]\s.{2,80}$", re.MULTILINE)

def truncate_at_word(s: str, max_len: int = 260) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    if len(s) <= max_len:
        return s
    cut = s[:max_len]
    # cut back to last space so we don't end with "opport..."
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:") + "…"


def clean_text(s: str) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)

    s = s.replace("\uFFFD", " ")  # �
    s = re.sub(r"[\x00-\x1F\x7F]", " ", s)  # control chars
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_transcript(text: str) -> str:
    if not text:
        return ""

    t = text.replace("\r", "\n")

    # Remove headings
    t = re.sub(r"(?i)\bprepared remarks:\b", "", t)
    t = re.sub(r"(?i)\bquestion-and-answer session:\b", "", t)
    t = re.sub(r"(?i)\bquestions and answers:\b", "", t)

    # Remove speaker title lines
    t = SPEAKER_LINE.sub("", t)

    # Remove bracketed directions
    t = re.sub(r"\[[^\]]{1,200}\]", "", t)

    # Kill common boilerplate sentences/phrases (operator + disclaimers)
    for pat in BOILERPLATE_PATTERNS:
        t = re.sub(pat, "", t, flags=re.IGNORECASE)

    # De-duplicate repeated paragraphs (Motley Fool sometimes repeats intros)
    # Normalize paragraphs -> remove exact duplicates
    paras = [p.strip() for p in re.split(r"\n{2,}", t) if p.strip()]
    seen = set()
    deduped = []
    for p in paras:
        key = re.sub(r"\s+", " ", p).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)

    t = "\n\n".join(deduped)

    # Collapse whitespace
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)

    return t.strip()


def split_sentences(text: str) -> List[str]:
    if not text:
        return []

    # normalize whitespace
    t = re.sub(r"\s+", " ", text).strip()

    # naive sentence split
    parts = re.split(r"(?<=[.!?])\s+", t)
    sentences = []

    for s in parts:
        s = s.strip()
        if len(s) < 60:
            continue

        # filter remaining boilerplate-y sentences
        low = s.lower()
        if any(x in low for x in [
            "conference operator",
            "forward-looking",
            "non-gaap",
            "cautionary statement",
            "available on our website",
            "turn the call over",
        ]):
            continue

        sentences.append(s)

    return sentences



def pick_evidence_sentences(text: str, keywords: List[str], k: int = 2) -> List[str]:
    sentences = split_sentences(text)
    scored: List[Tuple[int, str]] = []
    kw = [w.lower() for w in keywords]

    for s in sentences:
        s_l = s.lower()
        score = sum(1 for w in kw if w in s_l)
        if score > 0:
            scored.append((score, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:k]]


# guidance extraction

GUIDANCE_PATTERNS = [
    (r"\bguidance\b", ["guidance", "outlook", "forecast", "expect"]),
    (r"\boutlook\b", ["outlook", "expect", "anticipate", "project"]),
    (r"\b(revenue|sales)\b", ["revenue", "sales"]),
    (r"\b(eps|earnings per share)\b", ["eps", "earnings per share"]),
    (r"\bmargin(s)?\b", ["margin", "gross margin", "operating margin"]),
]

def extract_guidance(text: str, max_items: int = 6) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    text_clean = text.replace("\n", " ")
    for pat, kws in GUIDANCE_PATTERNS:
        if len(items) >= max_items: 
            break
        if re.search(pat, text_clean, flags=re.IGNORECASE):
            evidence = pick_evidence_sentences(text_clean, kws, k=2) 
            items.append({
                "type": kws[0],
                "evidence": evidence
            })
    return items

# Output schema

@dataclass
class AnalysisResult:
    summary: str
    summary_bullets: List[str]
    themes: List[Dict[str, Any]]
    risk_flags: List[Dict[str, Any]]
    guidance: List[Dict[str, Any]]
    


# Provider interface

ProviderName = Literal["heuristic", "openai"]

def get_provider_name() -> ProviderName:
    # default works without any extra setup
    #where is .env file???
    return os.getenv("ANALYSIS_PROVIDER", "heuristic").lower()  # heuristic|openai 



# Heuristic provider
THEME_KEYWORDS = {
    "Growth / Demand": [
        r"\bgrowth\b", r"\bbookings?\b", r"\brevenue\b", r"\btrips?\b", r"\bdemand\b", r"\bconsumer\b"
    ],
    "Margins / Profitability": [
        r"\bmargin(s)?\b", r"\bprofit\b", r"\bEBITDA\b", r"\boperating leverage\b", r"\bcost\b", r"\befficien"
    ],
    "Guidance / Outlook": [
        r"\bguidance\b", r"\boutlook\b", r"\bexpect\b", r"\bwe believe\b", r"\btarget\b", r"\b2023\b", r"\b2024\b"
    ],
    "Competition / Market": [
        r"\bcompetition\b", r"\bcompetitive\b", r"\bmarket share\b", r"\bpricing\b"
    ],
    "Operations / Supply": [
        r"\bsupply\b", r"\bdriver\b", r"\bdispatch\b", r"\boperations\b", r"\bresilien"
    ],
    "Regulatory / Legal": [
        r"\bregulat", r"\blegal\b", r"\bcompliance\b", r"\blaw\b"
    ],
    "Macro / FX": [
        r"\bmacro\b", r"\binflation\b", r"\brecession\b", r"\bFX\b", r"\bforeign exchange\b", r"\bheadwind\b"
    ],
}

RISK_PATTERNS = [
    ("Macro uncertainty", r"\bmacro\b|\brecession\b|\binflation\b|\buncertaint"),
    ("Demand weakness", r"\bweaker\b|\bsoft\b|\bslowdown\b|\bheadwind"),
    ("Regulatory / Legal", r"\bregulat|\blegal|\bcompliance|\blaw"),
    ("Execution risk", r"\bexecution\b|\bramp\b|\binvest\b|\bspend\b|\bpromot"),
    ("Competition intensity", r"\bcompetition\b|\bcompetitive\b|\bpricing pressure\b"),
]

def pick_evidence(sentences: List[str], regex_list: List[str], k: int = 2) -> List[str]:
    hits = []
    for s in sentences:
        for rgx in regex_list:
            if re.search(rgx, s, flags=re.IGNORECASE):
                hits.append(s)
                break
        if len(hits) >= k:
            break
    return hits


def build_themes(sentences: List[str]) -> List[Dict[str, Any]]:
    themes = []
    for theme, regs in THEME_KEYWORDS.items():
        ev = pick_evidence(sentences, regs, k=2)
        if ev:
            themes.append({
                "theme": theme,
                "evidence": [truncate_at_word(x, 260) for x in ev]
            })
    return themes


def build_risk_flags(sentences: List[str]) -> List[Dict[str, Any]]:
    flags = []
    for name, rgx in RISK_PATTERNS:
        ev = pick_evidence(sentences, [rgx], k=1)
        if ev:
            flags.append({
                "risk": name,
                "severity": "medium",
                "evidence": [truncate_at_word(ev[0], 260)]
            })
    return flags


def build_guidance(sentences: List[str]) -> List[Dict[str, Any]]:
    guidance = []
    # Simple signal categories
    guidance_map = {
        "guidance": [r"\bguidance\b", r"\boutlook\b", r"\bexpect\b", r"\btarget\b"],
        "revenue": [r"\brevenue\b", r"\bbookings?\b", r"\btop line\b"],
        "margin": [r"\bmargin\b", r"\bEBITDA\b", r"\bprofit\b", r"\boperating leverage\b"],
    }
    for typ, regs in guidance_map.items():
        ev = pick_evidence(sentences, regs, k=1)
        if ev:
            guidance.append({"type": typ, "evidence": ev})
    return guidance


def build_summary_bullets(themes: List[Dict[str, Any]], risk_flags: List[Dict[str, Any]], max_bullets: int = 7) -> List[str]:
    """
    Turn themes + risks into clean bullets. No speakers, no "Operator", no dialogue tone.
    """
    bullets: List[str] = []

    # Theme-based bullets
    for t in themes:
        # take the first evidence sentence and compress it a bit
        if t.get("evidence"):
            bullets.append(f"{t['theme']}: {t['evidence'][0]}")

    for rf in risk_flags[:2]:
        if rf.get("evidence"):
            bullets.append(f"Key risk — {rf['risk']}: {rf['evidence'][0]}")

    # trim
    return bullets[:max_bullets]


def build_executive_summary(ticker: str, quarter: str, call_date: str, bullets: List[str]) -> str:
    """
    Make a clean 3–5 sentence executive summary from the best bullets.
    """
    top = bullets[:4]
    # Strip "Theme:" prefixes to make prose
    top_clean = [re.sub(r"^[^:]{3,50}:\s*", "", b).strip() for b in top]

    body = " ".join(top_clean)
    # Keep it from becoming gigantic
    body = body[:900].rstrip()

    return (
        f"{ticker} {quarter} ({call_date}): Earnings-call intelligence summary.\n\n"
        f"{body}"
    ).strip()



# OpenAI-compatible provider 
def openai_analyze(ticker: str, quarter: str, date: str, transcript: str) -> AnalysisResult:
    """
    Requires:
      pip install openai
      set OPENAI_API_KEY
    Optional:
      OPENAI_MODEL (default: gpt-4o-mini)
      OPENAI_BASE_URL (if using a compatible provider)
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or None
    )
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    system = (
        "You are a financial earnings-call analyst. "
        "Return ONLY valid JSON matching the schema. "
        "Use short evidence quotes copied verbatim from the transcript."
    )

    schema_hint = {
        "summary": "string",
        "summary_bullets": ["string"],
        "themes": [{"theme": "string", "evidence": ["string"]}],
        "risk_flags": [{"risk": "string", "severity": "low|medium|high", "evidence": ["string"]}],
        "guidance": [{"type": "string", "evidence": ["string"]}],
        
    }

    user = f"""
Ticker: {ticker}
Quarter: {quarter}
Date: {date}

TASK:
Create an "earnings call intelligence" report. Focus on: growth, margins, guidance, risks, demand, strategy.
- summary_bullets: 5-10 bullets
- themes: 3-6
- risk_flags: 0-5 (severity low/medium/high)
- guidance: capture any outlook/guidance statements with evidence
- Always include evidence quotes.

JSON SCHEMA (example types):
{json.dumps(schema_hint, indent=2)}

TRANSCRIPT:
{transcript[:120000]}  # truncate to avoid token blowups
""".strip()

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )

    text = resp.choices[0].message.content.strip()

    # Robust JSON extraction: find first {...} block
    json_str = extract_first_json_object(text)
    data = json.loads(json_str)

    # Minimal validation + defaults
    summary = str(data.get("summary", "")).strip()
    summary_bullets = data.get("summary_bullets", [])
    themes = data.get("themes", [])
    risk_flags = data.get("risk_flags", [])
    guidance = data.get("guidance", [])
    

    return AnalysisResult(
        summary=summary,
        summary_bullets=summary_bullets if isinstance(summary_bullets, list) else [],
        themes=themes if isinstance(themes, list) else [],
        risk_flags=risk_flags if isinstance(risk_flags, list) else [],
        guidance=guidance if isinstance(guidance, list) else [],
       
    )


def extract_first_json_object(s: str) -> str:
    # find the first {...} region that parses as JSON
    start = s.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output.")
    # simple brace matching
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = s[start:i+1]
                # validate
                json.loads(candidate)
                return candidate
    raise ValueError("Could not extract a valid JSON object from model output.")

# Main entrypoint used by API

def analyze_transcript(ticker: str, quarter: str, call_date: datetime, transcript: str) -> Dict[str, Any]:
    cleaned = clean_transcript(transcript)
    sentences = split_sentences(cleaned)

    themes = build_themes(sentences)
    risk_flags = build_risk_flags(sentences)
    guidance = build_guidance(sentences)

    summary_bullets = build_summary_bullets(themes, risk_flags, max_bullets=7)
    summary = build_executive_summary(ticker, quarter, call_date, summary_bullets)

    return {
        "summary": summary,
        "summary_bullets": summary_bullets,
        "themes": themes,
        "risk_flags": risk_flags,
        "guidance": guidance,
        
    }


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
