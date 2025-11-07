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
                st.download_button("⬇️ Download Excel", open(out["export_path"], "rb"), "results.xlsx")
            with c2:
                st.markdown("<div class='section-title'>Sources</div>", unsafe_allow_html=True)
                render_doc_sources(out.get("sources", []))

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

        # Save turn
        st.session_state.turns.insert(0, {"q": q, "result": result})
        # We store a short text, not the whole dict (keeps history clean)
        st.session_state.messages.append({"role": "assistant", "content": result.get("merged") or out.get("insight","") or "(See above)"})

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
