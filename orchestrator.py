# orchestrator.py

from db_layer import connect_db, run_sql as run_sql_db
from rag_layer import retrieve, build_rag_context, has_indexed_docs
from genai_layer import nl_to_sql_and_insight, build_insight_prompt, llm_summarize
import re
import pandas as pd

# ------------------- Intent Detection -------------------

def detect_intent(q: str):
    ql = q.lower()

    # Document/definition triggers
    doc_trigger = any(x in ql for x in [
        "what is", "meaning", "define", "stands for", "explain", "describe",
        "policy", "ethics", "strategy", "guideline", "responsible", "procedure",
        "principle", "governance", "compliance"
    ])

    # SQL/business analytics triggers
    sql_trigger = any(x in ql for x in [
        "sales","revenue","profit","quantity","orders","units",
        "top","best","highest","lowest","category","segment","customer","product",
        "trend","month","year","quarter","growth","compare","analysis"
    ])

    # BOTH → HYBRID
    if doc_trigger and sql_trigger:
        return "hybrid"

    # Only docs → DOC
    if doc_trigger and not sql_trigger:
        return "doc"

    # Only analytics → SQL
    if sql_trigger:
        return "sql"

    return "doc"  # default safe fallback

# ------------------- SQL Pipeline -------------------

def run_sql_pipeline(q, schema, allowed, prev):
    res = nl_to_sql_and_insight(q, schema, allowed, prev)
    sql = res.get("sql")

    if not sql:
        return {"error": "SQL generation failed."}

    con = connect_db()
    try:
        df = run_sql_db(con, sql)
    except Exception as e:
        return {"error": f"SQL execution failed: {str(e)}", "sql": sql}

    df = pd.DataFrame(df)

    insight_prompt = build_insight_prompt(q, sql, df)
    insight = llm_summarize(insight_prompt)

    export_path = "export.xlsx"
    try:
        df.to_excel(export_path, index=False)
    except:
        export_path = None

    return {
        "sql": sql,
        "df": df,
        "insight": insight,
        "export_path": export_path,
        "sources": [{"type": "sql", "source": "salesDw.db"}]
    }

# ------------------- RAG Pipeline -------------------

def run_docs_pipeline(q, k=5):
    docs = retrieve(q, top_k=k)
    if not docs:
        return {
            "bullets": "⚠️ No relevant document chunks found.",
            "sources": [],
            "context_text": ""
        }

    ctx = build_rag_context(docs)

    prompt = f"""
Answer ONLY using the context. If answer missing say:
"Information not found in the provided documents."

User question: {q}

Context:
{ctx}

Format:
✅ Final answer (3 bullet points max)
📎 Evidence (quoted lines from docs)
"""

    bullets = llm_summarize(prompt)

    sources = []
    for r in docs:
        m = r.get("metadata", {})
        sources.append({
            "type": "doc",
            "file": m.get("source","unknown"),
            "chunk_index": m.get("chunk_index", -1),
            "score": float(r.get("score", 0))
        })

    return {
        "bullets": bullets,
        "sources": sources,
        "context_text": ctx
    }

# ------------------- Main Orchestrator -------------------

def orchestrate_query(q, schema, allowed, prev=None):
    intent = detect_intent(q)

    # ----- SQL ONLY -----
    if intent == "sql":
        sql = run_sql_pipeline(q, schema, allowed, prev)
        if not sql.get("error"):
            return {"type": "sql", "sql_output": sql}
        return {"type": "doc", "doc_output": run_docs_pipeline(q)}

    # ----- DOC ONLY -----
    if intent == "doc":
        return {"type": "doc", "doc_output": run_docs_pipeline(q)}

    # ----- HYBRID -----
    sql = run_sql_pipeline(q, schema, allowed, prev)
    doc = run_docs_pipeline(q)

    if sql.get("error"):
        return {"type": "doc", "doc_output": doc}
    if not doc.get("sources"):
        return {"type": "sql", "sql_output": sql}

    merged = llm_summarize(f"""
User asked: {q}

SQL Insight:
{sql.get('insight')}

Doc Insight:
{doc.get('bullets')}

Create a 4-line executive summary combining both business data + policy insight.
Tone: business leadership. Action-focused.
""")

    return {
        "type": "hybrid",
        "sql_output": sql,
        "doc_output": doc,
        "merged": merged,
        "rag_index_ready": has_indexed_docs()
    }
