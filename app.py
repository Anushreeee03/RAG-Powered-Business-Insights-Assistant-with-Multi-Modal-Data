# app.py
import os
import time
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
            
            # Enhanced evidence display with better charts
            for s in step_results:
                action_name = s.get('action', 'unknown')
                success = s.get('success', False)
                duration = int(s.get('duration_ms', 0))
                status_icon = "✅" if success else "❌"
                title = f"{status_icon} **{action_name.replace('_', ' ').title()}** ({duration} ms)"
                
                with st.expander(title, expanded=success):
                    ev = s.get("evidence", {})
                    
                    # SQL evidence
                    if ev.get("sql"):
                        st.markdown("**SQL Query:**")
                        st.code(ev.get("sql", ""), language="sql")
                    
                    # Time series analysis with enhanced chart
                    if action_name == "check_trend":
                        ts = ev.get("timeseries", {})
                        if ts and not ts.get("error"):
                            # Main chart
                            chart_data = charts.get("timeseries", {})
                            if chart_data and chart_data.get("period") and chart_data.get("value"):
                                df_chart = pd.DataFrame({
                                    "period": chart_data.get("period", []),
                                    "value": chart_data.get("value", [])
                                })
                                if not df_chart.empty:
                                    st.markdown("**Time Series Trend:**")
                                    st.line_chart(df_chart.set_index("period"), use_container_width=True)
                            
                            # Metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Trend", ts.get("trend", "unknown").title())
                            with col2:
                                change_pct = ts.get("change_last", 0.0) * 100
                                st.metric("Last Period Change", f"{change_pct:.2f}%")
                            with col3:
                                anomalies = ts.get("anomalies", [])
                                st.metric("Anomalies", len(anomalies))
                            
                            # Anomaly details
                            if anomalies:
                                st.markdown("**Detected Anomalies:**")
                                anomaly_df = pd.DataFrame(anomalies)
                                st.dataframe(anomaly_df[["period", "value", "z"]], use_container_width=True)
                        else:
                            st.warning(f"Time series analysis failed: {ts.get('error', 'Unknown error')}")
                    
                    # Breakdown analysis with bar chart
                    elif action_name == "breakdown":
                        bd = ev.get("breakdown", {})
                        if bd and not bd.get("error"):
                            rows = bd.get("rows", [])
                            if rows:
                                df_breakdown = pd.DataFrame(rows)
                                st.markdown("**Breakdown by Dimension:**")
                                
                                # Bar chart
                                if "value" in df_breakdown.columns:
                                    chart_col = bd.get("by", "category")
                                    if chart_col in df_breakdown.columns:
                                        st.bar_chart(df_breakdown.set_index(chart_col)["value"], use_container_width=True)
                                    else:
                                        # Fallback: use first non-numeric column as index
                                        index_col = [c for c in df_breakdown.columns if c != "value" and c != "contribution"][0] if len(df_breakdown.columns) > 1 else df_breakdown.columns[0]
                                        st.bar_chart(df_breakdown.set_index(index_col)["value"], use_container_width=True)
                                
                                # Table with contribution percentages
                                display_cols = [bd.get("by", "category"), "value"]
                                if "contribution" in df_breakdown.columns:
                                    display_cols.append("contribution")
                                # Filter to only existing columns
                                display_cols = [c for c in display_cols if c in df_breakdown.columns]
                                st.dataframe(df_breakdown[display_cols], use_container_width=True)
                            else:
                                st.json(bd)
                        else:
                            error_reason = bd.get("reason", bd.get("error", "Unknown error")) if bd else "No breakdown data"
                            st.warning(f"Breakdown failed: {error_reason}")
                            if ev.get("sql"):
                                st.code(ev.get("sql"), language="sql")
                    
                    # Policy/document retrieval
                    elif action_name == "read_policy":
                        ctx = ev.get("context", "")
                        sources = ev.get("sources", [])
                        if ctx:
                            st.markdown("**Retrieved Context:**")
                            st.text_area("", ctx[:2000], height=200, disabled=True, label_visibility="collapsed")
                        if sources:
                            st.markdown("**Document Sources:**")
                            render_doc_sources(sources)
                    
                    # Promo comparison
                    elif action_name == "compare_promo":
                        corr = ev.get("corr_sales_promo", 0.0)
                        st.metric("Sales-Promotion Correlation", f"{corr:.3f}")
                        if abs(corr) > 0.3:
                            st.info(f"Strong {'positive' if corr > 0 else 'negative'} correlation detected.")
                        else:
                            st.info("Weak correlation - promotions may not be a major factor.")
                    
                    # What-if scenario
                    elif action_name == "what_if":
                        wi = ev.get("what_if", {})
                        if wi:
                            base = wi.get("base", 0.0)
                            pct = wi.get("pct", 0.0) * 100
                            new_val = wi.get("new", 0.0)
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Base Value", f"{base:,.2f}")
                            with col2:
                                st.metric(f"With {pct:.1f}% Change", f"{new_val:,.2f}")
                            st.info(f"Counterfactual: If KPI changes by {pct:.1f}%, new value would be {new_val:,.2f}")
                    
                    # Error display
                    if not success and s.get("error"):
                        st.error(f"Error: {s.get('error')}")

            st.markdown("#### Conclusion")
            st.markdown(f"<div class='card'>{ao.get('conclusion','')}</div>", unsafe_allow_html=True)

            recs = ao.get("recommendations", "")
            if recs:
                st.markdown("#### Recommendations")
                st.markdown(f"<div class='card'>{recs}</div>", unsafe_allow_html=True)

            conf = float(ao.get("confidence", 0))
            st.markdown("#### Confidence Score")
            conf_threshold = 0.55
            conf_pct = conf * 100
            
            # Visual confidence indicator
            if conf < conf_threshold:
                st.warning(f"⚠️ **{conf_pct:.1f}%** - Below threshold ({conf_threshold*100:.0f}%)")
                st.caption("Review findings carefully. Consider additional analysis.")
            elif conf < 0.75:
                st.info(f"ℹ️ **{conf_pct:.1f}%** - Moderate confidence")
            else:
                st.success(f"✅ **{conf_pct:.1f}%** - High confidence")
            
            # Confidence breakdown (if available in logs)
            with st.expander("Confidence Details"):
                st.write(f"**Score:** {conf:.3f}")
                st.write(f"**Threshold:** {conf_threshold:.2f}")
                st.write(f"**Status:** {'Above threshold' if conf >= conf_threshold else 'Below threshold'}")
                st.caption("Confidence is calculated based on step success rate, evidence quality, and analysis depth.")

            st.markdown("<div class='section-title'>Citations</div>", unsafe_allow_html=True)
            render_doc_sources(ao.get("citations", []))

            # Enhanced report display
            rep = ao.get("report_md", "")
            if rep:
                st.markdown("---")
                st.markdown("#### 📄 Auto-Generated Report")
                with st.expander("View Full Report", expanded=False):
                    st.markdown(rep)
                st.download_button(
                    "📥 Download Report (Markdown)", 
                    rep.encode("utf-8"), 
                    f"kpi_report_{int(time.time())}.md", 
                    mime="text/markdown"
                )
            
            # Logs download
            log_path = "agent_logs.jsonl"
            if os.path.exists(log_path):
                with open(log_path, "rb") as f:
                    st.download_button(
                        "🗂️ Download JSON Logs", 
                        f.read(), 
                        f"agent_logs_{int(time.time())}.jsonl",
                        mime="application/jsonl"
                    )
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
