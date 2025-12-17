# app.py
import os
import pandas as pd
import streamlit as st

from orchestrator import orchestrate_query
from db_layer import connect_db, introspect_schema
from genai_layer import build_allowed_list
from rag_layer import ingest_pdfs_from_docs_dir, has_indexed_docs

st.set_page_config(page_title="Retail Data-to-Insight Assistant", layout="wide")

# ---- Minimal styling ----
st.markdown("""
<style>
/* Headline color + spacing */
h1, .stMarkdown h3 { margin-top: 0.4rem; }
/* Source badge style */
.badge {display:inline-block; padding:2px 8px; border-radius:12px; background:#eef3ff; margin:2px 6px 2px 0; font-size:0.85rem;}
.badge b {color:#3b5ed7;}
/* Section cards */
.card {padding:1rem; border:1px solid #e7e7e9; border-radius:12px; background:#fff;}
.section-title {font-weight:600; margin-bottom:0.4rem;}
.small-dim {color:#76839a; font-size:0.9rem;}
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align:center;color:#4B9CD3;'>🤖 Retail Data + Policy Assistant</h1>",
    unsafe_allow_html=True
)


# ---- Lightweight usage & health metrics ----
def load_agent_metrics(log_path: str = "agent_logs.jsonl", max_events: int = 500):
    if not os.path.exists(log_path):
        return {
            "sessions": 0,
            "avg_duration_ms": None,
            "avg_confidence": None,
            "success_rate": None,
        }
    sessions = 0
    durations = []
    confidences = []
    success_flags = []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_events:
                    break
                try:
                    ev = json.loads(line.strip())
                except Exception:
                    continue
                if ev.get("type") == "session_complete":
                    sessions += 1
                    if "duration_ms" in ev:
                        durations.append(float(ev.get("duration_ms", 0)))
                    if "confidence" in ev:
                        confidences.append(float(ev.get("confidence", 0)))
                    if "steps_total" in ev and ev.get("steps_total"):
                        success_flags.append(
                            float(ev.get("steps_completed", 0)) / float(ev.get("steps_total", 1))
                        )
    except Exception:
        return {
            "sessions": 0,
            "avg_duration_ms": None,
            "avg_confidence": None,
            "success_rate": None,
        }

    def _avg(lst):
        return (sum(lst) / len(lst)) if lst else None

    return {
        "sessions": sessions,
        "avg_duration_ms": _avg(durations),
        "avg_confidence": _avg(confidences),
        "success_rate": _avg(success_flags),
    }


def connection_health():
    """Basic stability checks for DB + RAG + LLM config (no external calls)."""
    db_ok = False
    schema_ok = False
    try:
        con = connect_db()
        _schema = introspect_schema(con)
        db_ok = True
        schema_ok = bool(_schema)
    except Exception:
        pass

    rag_ok = has_indexed_docs()
    llm_ok = bool(os.getenv("GROQ_API_KEY") or (hasattr(st, "secrets") and st.secrets.get("GROQ_API_KEY")))

    return {
        "db_ok": db_ok,
        "schema_ok": schema_ok,
        "rag_ok": rag_ok,
        "llm_ok": llm_ok,
    }

# ---- API key check ----
groq_key = os.getenv("GROQ_API_KEY") or (hasattr(st, "secrets") and st.secrets.get("GROQ_API_KEY"))

if not groq_key:
    st.error("Add GROQ_API_KEY in environment or .streamlit/secrets.toml")
    st.stop()


# ---- Session ----
if "schema" not in st.session_state:
    con = connect_db()
    st.session_state.schema = introspect_schema(con)
    st.session_state.allowed = build_allowed_list(st.session_state.schema)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "turns" not in st.session_state:
    st.session_state.turns = []
if "pending_q" not in st.session_state:
    st.session_state.pending_q = None
if "chunk_mode" not in st.session_state:
    st.session_state.chunk_mode = "sentence"

schema = st.session_state.schema
allowed = st.session_state.allowed

# Pre-compute health + usage insights for landing page and sidebar
health = connection_health()
metrics = load_agent_metrics()

# ---- Helpers ----
def short_title(text: str, max_words: int = 8) -> str:
    words = [w for w in text.replace("?", "").split() if w.strip()]
    return " ".join(words[:max_words]) + ("…" if len(words) > max_words else "")

def dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns
    new_cols, seen = [], {}
    for c in cols:
        if c not in seen:
            seen[c] = 0
            new_cols.append(c)
        else:
            seen[c] += 1
            new_cols.append(f"{c}_{seen[c]}")
    df.columns = new_cols
    return df

def render_doc_sources(sources):
    # Render as clean badges (no JSON)
    if not sources:
        st.caption("No sources.")
        return
    badges = []
    for i, s in enumerate(sources, 1):
        if s.get("type") == "doc":
            file = s.get("file", "unknown")
            idx = s.get("chunk_index", -1)
            score = s.get("score", 0.0)
            badges.append(f"<span class='badge'><b>[{i}]</b> {file} • chunk {idx} • score {score:.3f}</span>")
        else:
            src = s.get("source", "db")
            badges.append(f"<span class='badge'><b>SQL</b> {src}</span>")
    st.markdown(" ".join(badges), unsafe_allow_html=True)

def run_question(q: str):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    prev_turn = st.session_state.turns[0] if st.session_state.turns else None

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            result = orchestrate_query(q, schema, allowed, prev_turn)

        mode = result.get("type")
        assistant_summary = ""

        # --- SQL ONLY ---
        if mode == "sql":
            out = result.get("sql_output", {})
            if out.get("error"):
                st.error(out["error"]); return

            st.markdown("#### SQL")
            st.code(out["sql"], language="sql")

            df = dedupe_columns(out["df"])
            st.markdown("#### Results")
            st.dataframe(df.head(50), use_container_width=True)

            st.markdown("#### Insight")
            st.markdown(f"<div class='card'>{out['insight']}</div>", unsafe_allow_html=True)

            c1, c2 = st.columns([1,2])
            with c1:
                if out.get("export_path") and os.path.exists(out["export_path"]):
                    st.download_button("⬇️ Download Excel", open(out["export_path"], "rb"), "results.xlsx")
                else:
                    st.caption("Excel export unavailable")
            with c2:
                st.markdown("<div class='section-title'>Sources</div>", unsafe_allow_html=True)
                render_doc_sources(out.get("sources", []))
            assistant_summary = out.get("insight", "")

        # --- AGENT ---
        elif mode == "agent":
            ao = result.get("agent_output", {})
            st.markdown("#### Plan")
            steps = ao.get("plan", [])
            plan_lines = [f"- [{i+1}] {s.get('action','')} — {s.get('params',{})}" for i, s in enumerate(steps)]
            st.markdown("\n".join(plan_lines))

            st.markdown("#### Evidence")
            step_results = ao.get("steps", [])
            charts = ao.get("charts", {})
            for s in step_results:
                title = f"[{s.get('action','')}] {'✅' if s.get('success') else '❌'} ({int(s.get('duration_ms',0))} ms)"
                with st.expander(title, expanded=False):
                    ev = s.get("evidence", {})
                    if ev.get("sql"):
                        st.code(ev.get("sql",""), language="sql")
                    if s.get("action") == "check_trend":
                        ts = charts.get("timeseries", {})
                        if ts and ts.get("period") and ts.get("value"):
                            df_chart = pd.DataFrame({"period": ts.get("period", []), "value": ts.get("value", [])})
                            if not df_chart.empty:
                                st.line_chart(df_chart.set_index("period"))
                        st.json(ev.get("timeseries", {}))
                    if s.get("action") == "breakdown":
                        bd = ev.get("breakdown", {})
                        rows = bd.get("rows", [])
                        if rows:
                            st.dataframe(pd.DataFrame(rows))
                        else:
                            st.json(bd)
                    if s.get("action") == "read_policy":
                        ctx = ev.get("context", "")
                        if ctx:
                            st.code(ctx[:1000], language="text")
                        render_doc_sources(ev.get("sources", []))
                    if s.get("action") == "compare_promo":
                        st.write(f"Correlation sales vs promo: {round(ev.get('corr_sales_promo',0),2)}")
                    if s.get("action") == "what_if":
                        st.json(ev.get("what_if", {}))

            st.markdown("#### Conclusion")
            st.markdown(f"<div class='card'>{ao.get('conclusion','')}</div>", unsafe_allow_html=True)

            recs = ao.get("recommendations", "")
            if recs:
                st.markdown("#### Recommendations")
                st.markdown(f"<div class='card'>{recs}</div>", unsafe_allow_html=True)

            conf = float(ao.get("confidence", 0))
            st.markdown("#### Confidence")
            if conf < 0.55:
                st.warning(f"Low confidence: {conf}")
            else:
                st.success(f"Confidence: {conf}")

            st.markdown("<div class='section-title'>Citations</div>", unsafe_allow_html=True)
            render_doc_sources(ao.get("citations", []))

            rep = ao.get("report_md", "")
            if rep:
                st.download_button("📄 Download Report (MD)", rep.encode("utf-8"), "report.md", mime="text/markdown")
            log_path = "agent_logs.jsonl"
            if os.path.exists(log_path):
                st.download_button("🗂️ Download JSON Logs", open(log_path, "rb"), "agent_logs.jsonl")
            assistant_summary = (ao.get("conclusion","") + "\n" + ao.get("recommendations",""))

        # --- DOCS ONLY ---
        elif mode == "doc":
            out = result.get("doc_output", {})
            st.markdown("#### Document Answer")
            st.markdown(f"<div class='card'>{out.get('bullets','(No insights found)')}</div>", unsafe_allow_html=True)

            st.markdown("<div class='section-title'>Sources</div>", unsafe_allow_html=True)
            render_doc_sources(out.get("sources", []))

            with st.expander("Show retrieved evidence (optional)"):
                st.caption("These are the exact text snippets used for grounding.")
                st.code(out.get("context_text",""), language="text")

        # --- HYBRID ---
        else:
            sql_out = result.get("sql_output", {})
            doc_out = result.get("doc_output", {})

            st.markdown("#### SQL")
            st.code(sql_out.get("sql",""), language="sql")

            df = dedupe_columns(sql_out["df"])
            st.markdown("#### Results")
            st.dataframe(df.head(50), use_container_width=True)

            st.markdown("#### Document Insights")
            st.markdown(f"<div class='card'>{doc_out.get('bullets','')}</div>", unsafe_allow_html=True)

            st.markdown("#### Combined Narrative")
            st.markdown(f"<div class='card'>{result.get('merged','')}</div>", unsafe_allow_html=True)

            st.markdown("<div class='section-title'>Sources</div>", unsafe_allow_html=True)
            render_doc_sources(
                [{"type":"sql","source":"salesDw.db"}] +
                (doc_out.get("sources") or [])
            )

            with st.expander("Show retrieved evidence (optional)"):
                st.caption("These are the exact text snippets used for grounding.")
                st.code(doc_out.get("context_text",""), language="text")
            assistant_summary = result.get("merged", "")

        # Save turn
        st.session_state.turns.insert(0, {"q": q, "result": result})
        st.session_state.messages.append({"role": "assistant", "content": assistant_summary or "(See above)"})


# ---- Layout: Overview + Assistant ----
tab_overview, tab_assistant = st.tabs(["Overview & How to Use", "Assistant"])

with tab_overview:
    st.markdown("### What this tool does")
    st.markdown(
        """
**Retail Data + Policy Assistant** is a business-analytics copilot that:
- **Turns natural language questions into SQL** over your retail star-schema
- **Combines data + policy documents** using RAG to answer "why" and "how" questions
- **Runs an agentic root-cause investigation** for KPI drops (trend → breakdown → documents → what‑if)
- **Generates an executive summary and a downloadable report** with confidence scores and citations
        """
    )

    st.markdown("### Who should use it")
    st.markdown(
        """
- **Business analysts** who need quick, explainable KPI deep dives  
- **Revenue / sales leaders** tracking performance and promotions  
- **Operations / strategy teams** validating policies against actual results  
- **Data-curious PMs** who know the questions but not the SQL
        """
    )

    st.markdown("### Sample prompts")
    st.markdown(
        """
- **KPI diagnosis**:  
  - "Why did sales drop last quarter?"  
  - "Investigate the root cause of the orders decline by region and category."
- **What‑if & scenario planning**:  
  - "What if sales increased by 10% next month?"  
  - "How would revenue change if we reduced discounts by 5%?"
- **SQL-style questions** (data only):  
  - "Show top 10 products by sales."  
  - "List sales by customer segment and region."
- **Policy / docs questions**:  
  - "What is the return policy for electronics?"  
  - "Summarize our discount strategy."
        """
    )

    st.markdown("### FAQs & known limitations")
    st.markdown(
        """
- **Does it modify my database?**  
  - No. It only runs **read‑only `WITH`/`SELECT` queries** with strict safety checks.
- **What data does it see?**  
  - Only the connected SQLite DB (`salesDw.db`) and PDFs inside the `docs/` folder.
- **How reliable are answers?**  
  - Each agent run gets a **confidence score**; low scores show a warning and you should double‑check the evidence.
- **What about PII?**  
  - Logs and summaries apply **PII redaction** for emails, phones, SSNs, cards, and more, but you should avoid pasting raw sensitive data.
- **Model & latency**  
  - Uses a remote LLM (Groq); responses depend on network + API latency, typically a few seconds per full analysis.
        """
    )

    st.markdown("### System status & usage insights")
    cols = st.columns(4)
    with cols[0]:
        st.metric("DB connection", "OK" if health["db_ok"] else "Failed")
        st.caption("Checks `salesDw.db` and schema.")
    with cols[1]:
        st.metric("RAG index", "Ready" if health["rag_ok"] else "Not built")
        st.caption("PDFs indexed from `docs/`.")
    with cols[2]:
        st.metric("LLM configured", "Yes" if health["llm_ok"] else "Missing")
        st.caption("Requires `GROQ_API_KEY`.")
    with cols[3]:
        st.metric("Sessions (logged)", metrics["sessions"])
        avg_ms = metrics["avg_duration_ms"]
        if avg_ms:
            st.caption(f"Avg duration: {avg_ms/1000:.1f}s")

    if metrics["avg_confidence"] is not None or metrics["success_rate"] is not None:
        c1, c2 = st.columns(2)
        with c1:
            if metrics["avg_confidence"] is not None:
                st.metric("Avg confidence", f"{metrics['avg_confidence']*100:.1f}%")
        with c2:
            if metrics["success_rate"] is not None:
                st.metric("Avg tool success", f"{metrics['success_rate']*100:.1f}%")

    st.info(
        "When you're ready, switch to the **Assistant** tab, ask a question like "
        "\"Why did sales drop?\", and follow the plan/evidence/conclusion flow."
    )

with tab_assistant:
    # ---- Sidebar ----
    with st.sidebar:
        st.subheader("Conversation History")
        for i, t in enumerate(st.session_state.turns[:16], 1):
            label = short_title(t["q"])
            if st.button(label, key=f"hist_{i}"):
                st.session_state.pending_q = t["q"]
                st.rerun()

        st.markdown("---")
        st.subheader("RAG Index")
        st.caption("Place PDFs in ./docs/")
        st.write(f"Index status: {'✅ Ready' if has_indexed_docs() else '⚠️ Not built'}")

        mode = st.radio("Chunking", ["sentence", "paragraph"],
                        index=0 if st.session_state.chunk_mode == "sentence" else 1,
                        key="chunk_mode")
        if st.button("Rebuild Index"):
            with st.spinner("Indexing PDFs..."):
                info = ingest_pdfs_from_docs_dir(rebuild=True, mode=mode)
            st.success("Index rebuilt.")
            with st.expander("Index details"):
                st.write(info)

        st.markdown("---")
        st.subheader("Health snapshot")
        st.caption("Last app refresh")
        st.write(f"DB: {'✅' if health['db_ok'] else '⚠️'} | "
                 f"Schema: {'✅' if health['schema_ok'] else '⚠️'} | "
                 f"RAG: {'✅' if health['rag_ok'] else '⚠️'}")
        if metrics["avg_confidence"] is not None:
            st.write(f"Avg conf: {metrics['avg_confidence']*100:.0f}%")

        if st.button("Clear history"):
            st.session_state.messages.clear()
            st.session_state.turns.clear()
            st.success("History cleared.")

    # ---- Previous messages ----
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # ---- Run pending question from history ----
    if st.session_state.pending_q:
        q = st.session_state.pending_q
        st.session_state.pending_q = None
        run_question(q)

    # ---- Chat input ----
    user_q = st.chat_input("Ask about sales (SQL), policies or strategy (Docs), or combine both…")
    if user_q:
        run_question(user_q)
