# orchestrator.py

from db_layer import connect_db, run_sql as run_sql_db
from rag_layer import retrieve, build_rag_context, has_indexed_docs
from genai_layer import nl_to_sql_and_insight, build_insight_prompt, llm_summarize

import pandas as pd
import re

# ---------------- INTENT DETECTION ---------------- #

def detect_intent(q: str) -> str:
    ql = q.lower().strip()
    
    print(f"Detecting intent for: {q}")
    
    # Check for explicit hybrid indicators first
    hybrid_indicators = [
        r"\b(and|also|plus|as well as|in addition)\b",
        r"\bshow.*and.*explain\b",
        r"\banalyze.*and.*describe\b",
        r"\bmetrics.*and.*policy\b",
        r"\bdata.*and.*guidelines\b"
    ]
    
    for indicator in hybrid_indicators:
        if re.search(indicator, ql):
            print(f"Hybrid intent detected: {indicator}")
            return "hybrid"
    
    # Strong SQL patterns (data/metrics focused)
    sql_patterns = [
        r"^(show|list|get|find|calculate|analyze)\s",
        r"\b(top|bottom)\s+\d+\b",
        r"\b(how many|how much|count of|total|sum|average)\b",
        r"\b(sales|revenue|profit|income|margin|growth)\b",
        r"\b(orders|customers|products|units|quantity)\b",
        r"\b(by|per|for)\s+(month|year|quarter|day|week)\b",
        r"\b(region|state|city|category|segment)\b",
        r"\b(trend|performance|ranking|distribution)\b"
    ]
    
    # Strong document patterns (explanation/policy focused)
    doc_patterns = [
        r"^(what is|explain|define|describe|tell me about)\s",
        r"\b(policy|policies|guideline|guidelines)\b",
        r"\b(procedure|process|method|methodology)\b",
        r"\b(compliance|ethical|ethics|legal|regulation)\b",
        r"\b(governance|audit|risk|security|privacy)\b",
        r"\b(strategy|strategic|mission|vision|objective)\b",
        r"\b(sustainable|sustainability|esg|responsibility)\b",
        r"\b(framework|standard|best practice)\b"
    ]
    
    sql_score = sum(1 for pattern in sql_patterns if re.search(pattern, ql))
    doc_score = sum(1 for pattern in doc_patterns if re.search(pattern, ql))
    
    print(f"SQL score: {sql_score}, Doc score: {doc_score}")
    
    # Decision logic
    if sql_score >= 2 and doc_score >= 1:
        print("Mixed query with strong SQL component → Hybrid")
        return "hybrid"
    elif doc_score >= 2 and sql_score >= 1:
        print("Mixed query with strong doc component → Hybrid") 
        return "hybrid"
    elif sql_score >= 2:
        print("Strong SQL patterns → SQL")
        return "sql"
    elif doc_score >= 2:
        print("Strong doc patterns → Document")
        return "doc"
    elif sql_score > doc_score:
        print("More SQL patterns → SQL")
        return "sql"
    elif doc_score > sql_score:
        print("More doc patterns → Document")
        return "doc"
    else:
        # For ambiguous queries, check for question type
        if any(word in ql for word in ["what", "how", "why", "explain", "define"]):
            print("Ambiguous but explanatory → Document")
            return "doc"
        else:
            print("Ambiguous default → SQL")
            return "sql"

def split_hybrid(q: str):
    # Enhanced splitting patterns for better hybrid query handling
    patterns = [
        r"\band\b",
        r"\balso\b",
        r"\bplus\b",
        r"&",
        r";",
        r"\bthen\b",
        r"\bas well as\b",
        r"\bin addition\b"
    ]
    
    for pattern in patterns:
        parts = re.split(pattern, q, maxsplit=1, flags=re.I)
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
    
    return q, ""

# ---------------- SQL PIPELINE ---------------- #

def run_sql_pipeline(q, schema, allowed, prev):
    print(f"Running SQL pipeline for: {q}")
    res = nl_to_sql_and_insight(q, schema, allowed, prev)
    sql = res.get("sql")
    
    if not sql:
        error_msg = res.get("error", "SQL generation failed")
        print(f"SQL generation failed: {error_msg}")
        return {
            "error": f"❌ **SQL Generation Failed**\n\n{error_msg}\n\n**Suggestions:**\n• Try rephrasing your question\n• Use simpler terms like 'show sales by category'\n• Check if the requested data exists in the database"
        }

    print(f"Generated SQL: {sql}")

    con = connect_db()
    try:
        df = run_sql_db(con, sql)
        print(f"SQL execution successful, returned {len(df)} rows")
        
        if len(df) == 0:
            print("Query returned no results")
            return {
                "sql": sql,
                "df": pd.DataFrame(),
                "insight": "📊 **No Data Found**\n\nThe query executed successfully but returned no results. This could mean:\n• No matching data exists\n• Date filters are too restrictive\n• Try broader search terms",
                "sources": [{"type":"sql","source":"salesDw.db"}],
                "export_path": None
            }
            
    except Exception as e:
        error_msg = str(e)
        print(f"SQL execution failed: {error_msg}")
        
        # Try to provide helpful error message
        if "no such table" in error_msg.lower():
            help_text = "The requested table doesn't exist in the database."
        elif "no such column" in error_msg.lower():
            help_text = "The requested column doesn't exist. Check column names in the schema."
        elif "syntax error" in error_msg.lower():
            help_text = "There was a syntax error in the generated SQL."
        else:
            help_text = "Database query failed."
            
        return {
            "error": f"❌ **SQL Execution Error**\n\n{help_text}\n\n**Error Details:** {error_msg}\n\n**Generated SQL:**\n```sql\n{sql}\n```",
            "sql": sql
        }

    df = pd.DataFrame(df)
    
    # Generate insight with better error handling
    try:
        insight_prompt = f"""
Analyze the following business data and provide 3-4 clear, actionable insights:

Question: {q}
Data: {df.head(10).to_string()}

Provide insights in this format:
• **Key Finding**: [Main observation]
• **Business Impact**: [What this means for the business]
• **Recommendation**: [Suggested action or next step]

Keep insights concise and business-focused.
"""
        insight = llm_summarize(insight_prompt)
        print(f"Insight generated: {insight[:100]}...")
    except Exception as e:
        print(f"Insight generation failed: {e}")
        insight = f"📊 **Data Summary**\n\nQuery returned {len(df)} rows.\n\nUnable to generate detailed insights due to API error. The raw data is available above."

    export_path = "export.xlsx"
    try:
        df.to_excel(export_path, index=False)
        print(f"Excel export created: {export_path}")
    except Exception as e:
        print(f"Excel export failed: {e}")
        export_path = None

    return {
        "sql": sql,
        "df": df,
        "insight": insight,
        "sources": [{"type":"sql","source":"salesDw.db"}],
        "export_path": export_path
    }

# ---------------- DOC PIPELINE ---------------- #

def run_docs_pipeline(q):
    print(f"Running document pipeline for: {q}")
    
    # Check if we have indexed documents
    if not has_indexed_docs():
        print("No indexed documents found")
        return {
            "bullets": "❌ **No documents indexed**\n\nPlease add PDF files to the 'docs' folder and click 'Rebuild Index' in the sidebar to enable document search.",
            "sources": [],
            "context_text": "No documents available for this query."
        }
    
    # Try multiple retrieval strategies
    docs = retrieve(q, top_k=6)
    print(f"Retrieved {len(docs)} documents")
    
    if not docs:
        print("No documents found for query")
        # Try with broader terms
        broader_q = re.sub(r'\b(state-wise|procedure|responsible)\b', '', q).strip()
        if broader_q != q:
            print(f"Trying broader query: {broader_q}")
            docs = retrieve(broader_q, top_k=6)
            print(f"Retrieved {len(docs)} documents with broader query")
    
    if not docs:
        return {
            "bullets": f"📄 **No relevant information found**\n\nI couldn't find specific information about '{q}' in the indexed documents.\n\n**Suggestions:**\n• Try different keywords\n• Check if relevant PDFs are in the docs folder\n• Rebuild the index if you recently added documents\n• Use more general terms",
            "sources": [],
            "context_text": "No relevant documents found."
        }

    ctx = build_rag_context(docs)
    print(f"Built context with {len(ctx)} characters")

    # Enhanced prompt for better answers
    prompt = f"""
You are a helpful business assistant. Use ONLY the provided context to answer the user's question comprehensively.

User Question: {q}

Context Information:
{ctx}

Instructions:
1. Answer the question based ONLY on the context provided
2. If the context doesn't contain the specific information, clearly state what IS available
3. Provide a clear, structured answer with bullet points if helpful
4. If information is partially available, provide what you can and indicate what's missing
5. Be specific and helpful

Answer:
"""

    try:
        bullets = llm_summarize(prompt)
        print(f"Generated response: {bullets[:100]}...")
        
        # If the response indicates no information, try a more flexible approach
        if "don't have enough information" in bullets.lower() or "not in the context" in bullets.lower():
            print("First attempt failed, trying more flexible approach")
            flexible_prompt = f"""
Based on the available context, provide the most helpful response possible:

User Question: {q}

Available Context:
{ctx}

Even if you can't answer the specific question, provide any related information from the context that might be helpful to the user.
"""
            bullets = llm_summarize(flexible_prompt)
            print(f"Flexible response: {bullets[:100]}...")
            
    except Exception as e:
        print(f"LLM summarization failed: {e}")
        bullets = f"❌ **Error processing question**\n\nI encountered an error while processing your question about '{q}'.\n\nPlease try again or check if the documents are properly indexed."

    src = []
    for r in docs:
        m = r.get("metadata", {})
        src.append({
            "file": m.get("source", "unknown"),
            "chunk_index": m.get("chunk_index", -1),
            "score": float(r.get("score", 0))
        })

    return {"bullets": bullets, "sources": src, "context_text": ctx}

# ---------------- MAIN ---------------- #

def orchestrate_query(q, schema, allowed, prev=None):
    print(f"\n=== ORCHESTRATING QUERY: {q} ===")
    
    intent = detect_intent(q)
    print(f"Detected intent: {intent}")

    if intent == "sql":
        print("→ Routing to SQL pipeline")
        return {"type": "sql", "sql_output": run_sql_pipeline(q, schema, allowed, prev)}

    if intent == "doc":
        print("→ Routing to document pipeline")
        return {"type": "doc", "doc_output": run_docs_pipeline(q)}

    # Hybrid case
    print("→ Routing to hybrid pipeline")
    sql_q, doc_q = split_hybrid(q)
    print(f"Split into SQL: '{sql_q}' and DOC: '{doc_q}'")
    
    sql = run_sql_pipeline(sql_q, schema, allowed, prev)
    doc = run_docs_pipeline(doc_q)

    if sql.get("error"):
        print("SQL part failed, returning document only")
        return {"type": "doc", "doc_output": doc}

    try:
        merged = llm_summarize(f"""
User asked: {q}
SQL insights: {sql.get('insight')}
Document insights: {doc.get('bullets')}
Provide 4 leadership summary lines combining both insights.
""")
        print("Generated merged summary")
    except Exception as e:
        print(f"Merge failed: {e}")
        merged = f"SQL Analysis: {sql.get('insight')}\nDocument Analysis: {doc.get('bullets')}"

    return {
        "type": "hybrid",
        "sql_output": sql,
        "doc_output": doc,
        "merged": merged,
        "rag_index_ready": has_indexed_docs()
    }
