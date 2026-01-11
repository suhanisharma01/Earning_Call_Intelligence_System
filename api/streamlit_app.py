import requests
import streamlit as st
import yfinance as yf

@st.cache_data(ttl=7*24*3600)
def get_company_name(ticker: str) -> str:
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        return info.get("longName") or info.get("shortName") or ticker
    except Exception:
        return ticker


# -------------------------
# Config
# -------------------------
DEFAULT_API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Earnings Call Intelligence", page_icon="📈", layout="wide")
st.title("📈 Earnings Call Intelligence (Streamlit)")
st.caption("Browse calls → open transcript → generate intelligence report → cached in SQLite via FastAPI")


# Helpers

@st.cache_data(ttl=10)
def api_get(path: str, params=None):
    base = st.session_state.get("API_BASE", DEFAULT_API_BASE).rstrip("/")
    url = base + path
    r = requests.get(url, params=params, timeout=60)
    return r

def api_post(path: str, params=None):
    base = st.session_state.get("API_BASE", DEFAULT_API_BASE).rstrip("/")
    url = base + path
    r = requests.post(url, params=params, timeout=180)
    return r

def pretty_err(r: requests.Response) -> str:
    try:
        return r.json()
    except Exception:
        return r.text


# -------------------------
# Sidebar: connection + filters
# -------------------------
with st.sidebar:
    st.header("🔧 Connection")
    api_base = st.text_input("API Base URL", value=DEFAULT_API_BASE)
    st.session_state["API_BASE"] = api_base

    # quick health check
    if st.button("Check API"):
        try:
            r = api_get("/health")
            if r.ok:
                st.success(r.json())
            else:
                st.error(pretty_err(r))
        except Exception as e:
            st.error(str(e))

    st.divider()
    st.header("🔎 Browse filters")

    ticker = st.text_input("Ticker (exact)", value="")
    exchange = st.text_input("Exchange (exact)", value="")
    quarter = st.text_input("Quarter (exact)", value="")

    st.divider()
    st.header("🔍 Transcript search (FTS)")
    q = st.text_input("Search query (optional)", value="")
    st.caption("Uses GET /search. If you didn’t build FTS, this may fail — browse still works.")

    st.divider()
    st.header("📄 Pagination")
    limit = st.slider("Limit", min_value=10, max_value=200, value=50, step=10)
    offset = st.number_input("Offset", min_value=0, value=0, step=50)

    if st.button("Refresh list"):
        api_get.clear()  # clear cache


# -------------------------
# Main: load call list
# -------------------------

call_id = None
st.subheader("📚 Calls")

params = {"limit": int(limit), "offset": int(offset)}
if ticker.strip():
    params["ticker"] = ticker.strip()
if exchange.strip():
    params["exchange"] = exchange.strip()
if quarter.strip():
    params["quarter"] = quarter.strip()

rows = []
error = None



try:
    if q.strip():
        r = api_get("/search", params={"q": q.strip(), "limit": int(limit), "offset": int(offset)})
    else:
        r = api_get("/calls", params=params)

    if r.ok:
        rows = r.json()
    else:
        error = pretty_err(r)
except Exception as e:
    error = str(e)

if error:
    st.error(error)
else:
    st.caption(f"Showing {len(rows)} calls.")

    if rows:
        # Create a readable label for dropdown
        options = {}

        for r in rows:
            cid = r.get("call_id", "")
            ticker = r.get("ticker", "")
            quarter = r.get("quarter", "")

            company = get_company_name(ticker)

            label = f"{company} — {quarter}"
             # ensure uniqueness internally
            unique_label = f"{label}##{cid}"

            options[unique_label] = cid


        selected = st.selectbox(
            "Select a call",
            list(options.keys()),
            format_func=lambda x: x.split("##")[0]
            )

        call_id = options[selected]


    else:
           st.info("No calls returned. Try different filters / offset.")
    
        


# -------------------------
# Load selected call
# -------------------------
if call_id:
    st.divider()
    st.subheader("📄 Call Details")

    colA, colB = st.columns([1, 1])

    with colA:
        st.markdown("### Transcript")
        try:
            r = api_get(f"/calls/{call_id}")
            if not r.ok:
                st.error(pretty_err(r))
                call = None
            else:
                call = r.json()
        except Exception as e:
            st.error(str(e))
            call = None

        if call:
            st.write(
                f"**{call['ticker']}** · {call['quarter']} · {call['exchange']} · {call['date']}"
            )
            st.text_area("Transcript", value=call.get("transcript", ""), height=420)

    with colB:
        st.markdown("### Intelligence Report")

        # Buttons
        btn1, btn2, btn3 = st.columns([1, 1, 1])

        with btn1:
            if st.button("Generate report", use_container_width=True):
                with st.spinner("Analyzing & caching..."):
                    try:
                        rp = api_post(f"/calls/{call_id}/analyze")
                        if not rp.ok:
                            st.error(pretty_err(rp))
                        else:
                            st.success(rp.json())
                            api_get.clear()  # refresh cached /insights
                    except Exception as e:
                        st.error(str(e))

        

        # Insights display
        try:
            ri = api_get(f"/calls/{call_id}/insights")
            if not ri.ok:
                st.info("No insights yet. Click **Generate report**.")
                insights = None
            else:
                insights = ri.json()
        except Exception as e:
            st.error(str(e))
            insights = None

        if insights:
            st.markdown("#### Summary")
            st.write(insights.get("summary", ""))

            bullets = insights.get("summary_bullets") or []
            if bullets:
                st.markdown("#### Summary bullets")
                for b in bullets:
                    st.markdown(f"- {b}")

            themes = insights.get("themes") or []
            if themes:
                st.markdown("#### Themes")
                for t in themes:
                    st.markdown(f"**• {t.get('theme','')}**")
                    for ev in (t.get("evidence") or []):
                        st.markdown(f"> {ev}")

            risks = insights.get("risk_flags") or []
            if risks:
                st.markdown("#### Risk flags")
                for rsk in risks:
                    title = rsk.get("risk", "")
                    sev = rsk.get("severity")
                    st.markdown(f"**• {title}{f' ({sev})' if sev else ''}**")
                    for ev in (rsk.get("evidence") or []):
                        st.markdown(f"> {ev}")

            guidance = insights.get("guidance") or []
            if guidance:
                st.markdown("#### Guidance signals")
                for g in guidance:
                    st.markdown(f"**• {g.get('type','')}**")
                    for ev in (g.get("evidence") or []):
                        st.markdown(f"> {ev}")

          
