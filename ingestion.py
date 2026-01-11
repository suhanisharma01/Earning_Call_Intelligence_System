import pandas as pd
import sqlite3
import hashlib

DB_PATH = "earnings_calls.db"

df = pd.read_pickle("motley-fool-data.pkl")

print("Original columns:", df.columns)

df = df.rename(columns={
    "call_date": "date",
    "q": "quarter",
    "full_transcript": "transcript"
})

# keep only required columns
df = df[["date", "exchange", "quarter", "ticker", "transcript"]].copy()

# Fix exchange like "NASDAQ: BILI" -> "NASDAQ"
df["exchange"] = df["exchange"].astype(str).str.split(":").str[0].str.strip().str.upper()

# Clean ticker
df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

# Parse date
df["date"] = (
    df["date"]
    .str.replace(r"\s+ET$", "", regex=True)
)

df["date"] = (df["date"]
              .str.replace("a.m.", "AM", regex=False)
              .str.replace("p.m.", "PM", regex=False))

df["date"] = pd.to_datetime(df["date"], format="%b %d, %Y, %I:%M %p", errors="coerce")


# Drop rows with bad dates/transcripts
df["transcript"] = df["transcript"].astype(str)
df = df.dropna(subset=["date"])
df = df[df["transcript"].str.len() > 500]

# Create stable call_id
df["call_id"] = df.apply(
    lambda r: hashlib.sha1(f"{r['ticker']}|{r['date']}|{r['quarter']}".encode()).hexdigest(),
    axis=1
)

# Store date as ISO string for SQLite sorting
df["date"] = df["date"].dt.strftime("%Y-%m-%d %H:%M:%S")

df = df[["call_id", "date", "exchange", "quarter", "ticker", "transcript"]]

print("Final columns:", df.columns)
print(df.head())
print(df.dtypes)

# Write to SQLite
conn = sqlite3.connect(DB_PATH)
df.to_sql("earnings_calls", conn, if_exists="replace", index=False)
conn.close()
print("Data saved to SQLite.")

# Create indexes
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.executescript("""
CREATE INDEX IF NOT EXISTS idx_calls_ticker_date ON earnings_calls(ticker, date);
CREATE INDEX IF NOT EXISTS idx_calls_exchange ON earnings_calls(exchange);
CREATE INDEX IF NOT EXISTS idx_calls_quarter ON earnings_calls(quarter);
""")
conn.commit()
conn.close()
print("Indexes created.")

# Build FTS safely (no duplicates)
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.executescript("""
DROP TABLE IF EXISTS earnings_calls_fts;
CREATE VIRTUAL TABLE earnings_calls_fts USING fts5(call_id, ticker, quarter, transcript);
INSERT INTO earnings_calls_fts (call_id, ticker, quarter, transcript)
SELECT call_id, ticker, quarter, transcript FROM earnings_calls;
""")
conn.commit()
conn.close()
print("FTS index built.")

# Sanity check 
conn = sqlite3.connect(DB_PATH)
print("rows:", conn.execute("SELECT COUNT(*) FROM earnings_calls").fetchone()[0])
print("sample:", conn.execute("""
  SELECT call_id, ticker, date, quarter, LENGTH(transcript) as n
  FROM earnings_calls
  ORDER BY date DESC
  LIMIT 5
""").fetchall())
conn.close()

print("df rows:", len(df))

