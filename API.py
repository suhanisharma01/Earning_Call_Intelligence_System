import sqlite3
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from api.analysis import analyze_transcript, now_utc_iso
import json


# Adjust if your DB is elsewhere
DB_PATH = Path("earnings_calls.db")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Performance-friendly pragmas for read-heavy API usage
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    return conn


def init_db() -> None:
    """Create required tables/indexes for Phase 2 if they don't already exist."""
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
CREATE TABLE IF NOT EXISTS call_insights (
    call_id TEXT PRIMARY KEY,
    summary TEXT,
    summary_bullets_json TEXT,
    themes_json TEXT,
    risk_flags_json TEXT,
    guidance_json TEXT,
    created_at TEXT NOT NULL
);
""")
    

    def add_column_if_missing(conn, table: str, col: str, col_type: str = "TEXT"):
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table});").fetchall()]
        if col not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type};")

    add_column_if_missing(conn, "call_insights", "summary_bullets_json", "TEXT")
    add_column_if_missing(conn, "call_insights", "guidance_json", "TEXT")

    # Ensure expected base table exists
    base_exists = cur.execute("""
        SELECT 1 FROM sqlite_master WHERE type='table' AND name='earnings_calls';
    """).fetchone()
    if not base_exists:
        conn.close()
        raise RuntimeError(
            "Base table 'earnings_calls' not found in earnings_calls.db. "
            "Run your ingestion script first."
        )

    # Helpful indexes for common filters
    cur.execute("CREATE INDEX IF NOT EXISTS idx_calls_ticker_date ON earnings_calls(ticker, date);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_calls_exchange ON earnings_calls(exchange);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_calls_quarter ON earnings_calls(quarter);")

    conn.commit()
    conn.close()


def fts_available() -> bool:
    conn = get_conn()
    cur = conn.cursor()
    exists = cur.execute("""
        SELECT 1 FROM sqlite_master WHERE type='table' AND name='earnings_calls_fts';
    """).fetchone()
    conn.close()
    return bool(exists)



def extract_first_json_object(s: str) -> str:
    start = s.find("{")
    if start == -1:
        raise ValueError("No JSON found in model output.")
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = s[start:i+1]
                json.loads(candidate)
                return candidate
    raise ValueError("Could not extract valid JSON from model output.")


def openai_analyze(ticker: str, quarter: str, date: str, transcript: str) -> dict[str, any]:
    # Optional: only used if ANALYSIS_PROVIDER=openai and OPENAI_API_KEY is set
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    schema_hint = {
        "summary": "string",
        "summary_bullets": ["string"],
        "themes": [{"theme": "string", "evidence": ["string"]}],
        "risk_flags": [{"risk": "string", "severity": "low|medium|high", "evidence": ["string"]}],
        "guidance": [{"type": "string", "evidence": ["string"]}],
    
    }

    system = (
        "You are a financial earnings-call analyst. "
        "Return ONLY valid JSON matching the schema. "
        "Use short evidence quotes copied verbatim from the transcript."
    )

    user = f"""
Ticker: {ticker}
Quarter: {quarter}
Date: {date}

Create an earnings-call intelligence report.
Rules:
- summary_bullets: 5-10 bullets
- themes: 3-6, each with 1-3 evidence quotes
- risk_flags: 0-5 with severity low/medium/high and evidence quotes
- guidance: capture outlook/guidance-related statements with evidence
- Output ONLY JSON

Schema hint:
{json.dumps(schema_hint, indent=2)}

Transcript:
{transcript[:120000]}
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
    data = json.loads(extract_first_json_object(text))

    # Normalize
    return {
        "summary": str(data.get("summary", "")).strip(),
        "summary_bullets": data.get("summary_bullets", []) if isinstance(data.get("summary_bullets"), list) else [],
        "themes": data.get("themes", []) if isinstance(data.get("themes"), list) else [],
        "risk_flags": data.get("risk_flags", []) if isinstance(data.get("risk_flags"), list) else [],
        "guidance": data.get("guidance", []) if isinstance(data.get("guidance"), list) else [],
        
    }





#main
from fastapi import FastAPI, HTTPException, Query
from datetime import datetime, timezone
import json


app = FastAPI(title="Earnings Call Intelligence API")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    yield
    # Shutdown (nothing needed here for now)

app = FastAPI(
    title="Earnings Call Intelligence API",
    lifespan=lifespan
)



@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/calls")
def list_calls(
    ticker: str | None = None,
    exchange: str | None = None,
    quarter: str | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """
    List calls with optional filters and pagination.
    Returns metadata only (not transcript) for speed.
    """
    where = []
    params: list[str | int] = []

    if ticker:
        where.append("ticker = ?")
        params.append(ticker.upper().strip())
    if exchange:
        where.append("exchange = ?")
        params.append(exchange.upper().strip())
    if quarter:
        where.append("quarter = ?")
        params.append(quarter.upper().strip())

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    sql = f"""
        SELECT call_id, ticker, exchange, quarter, date
        FROM earnings_calls
        {where_sql}
        ORDER BY date DESC
        LIMIT ? OFFSET ?;
    """
    params.extend([limit, offset])

    conn = get_conn()
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/calls/{call_id}")
def get_call(call_id: str):
    """
    Fetch a single call including transcript.
    """
    conn = get_conn()
    row = conn.execute("""
        SELECT call_id, ticker, exchange, quarter, date, transcript
        FROM earnings_calls
        WHERE call_id = ?
        LIMIT 1;
    """, (call_id,)).fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="call_id not found")

    return dict(row)


@app.get("/search")
def search_calls(
    q: str = Query(..., min_length=2),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """
    Full-text search across transcripts using FTS5.
    Requires you to have built the FTS table:
      earnings_calls_fts(call_id, ticker, quarter, transcript)
    """
    if not fts_available():
        raise HTTPException(
            status_code=400,
            detail="FTS table 'earnings_calls_fts' not found. Build it during ingestion first."
        )

    conn = get_conn()
    rows = conn.execute("""
        SELECT ec.call_id, ec.ticker, ec.exchange, ec.quarter, ec.date
        FROM earnings_calls_fts f
        JOIN earnings_calls ec ON ec.call_id = f.call_id
        WHERE earnings_calls_fts MATCH ?
        ORDER BY ec.date DESC
        LIMIT ? OFFSET ?;
    """, (q, limit, offset)).fetchall()
    conn.close()
    return [dict(r) for r in rows]



@app.get("/calls/{call_id}/insights")
def get_insights(call_id: str):
    conn = get_conn()
    row = conn.execute("""
        SELECT call_id, summary, summary_bullets_json, themes_json, risk_flags_json, guidance_json, created_at
        FROM call_insights
        WHERE call_id = ?
        LIMIT 1;
    """, (call_id,)).fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="No insights yet. Call POST /calls/{call_id}/analyze first.")

    d = dict(row)
    d["summary_bullets"] = json.loads(d["summary_bullets_json"] or "[]")
    d["themes"] = json.loads(d["themes_json"] or "[]")
    d["risk_flags"] = json.loads(d["risk_flags_json"] or "[]")
    d["guidance"] = json.loads(d["guidance_json"] or "[]")
    

    return d



@app.post("/calls/{call_id}/analyze")
def analyze_call(call_id: str, force: bool = False):
    conn = get_conn()

    call = conn.execute("""
        SELECT call_id, ticker, exchange, quarter, date, transcript
        FROM earnings_calls
        WHERE call_id = ?
        LIMIT 1;
    """, (call_id,)).fetchone()

    if not call:
        conn.close()
        raise HTTPException(status_code=404, detail="call_id not found")

    if not force:
        existing = conn.execute("""
            SELECT 1 FROM call_insights WHERE call_id = ? LIMIT 1;
        """, (call_id,)).fetchone()
        if existing:
            conn.close()
            return {"call_id": call_id, "status": "already_cached"}

    result = analyze_transcript(
        ticker=call["ticker"],
        quarter=call["quarter"],
        call_date=call["date"],
        transcript=call["transcript"] or ""
    )

    conn.execute("""
        INSERT OR REPLACE INTO call_insights(
            call_id, summary, summary_bullets_json, themes_json, risk_flags_json, guidance_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?);
    """, (
        call_id,
        result.get("summary", ""),
        json.dumps(result.get("summary_bullets", [])),
        json.dumps(result.get("themes", [])),
        json.dumps(result.get("risk_flags", [])),
        json.dumps(result.get("guidance", [])),
        now_utc_iso()
    ))

    conn.commit()
    conn.close()
    return {"call_id": call_id, "status": "cached", "provider": os.getenv("ANALYSIS_PROVIDER", "heuristic")}

