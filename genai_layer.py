# nl2sql_advanced.py
import os, re, json, sqlite3
from typing import Dict, List, Tuple, Optional, Set, Iterable
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

try:
    import sqlparse
except ImportError:
    raise ImportError("sqlparse is required. Install with: pip install sqlparse")

try:
    import pandas as pd  # optional: used for insight prompts
except ImportError:
    raise ImportError("pandas is required. Install with: pip install pandas")

try:
    from groq import Groq
except ImportError:
    raise ImportError("groq is required. Install with: pip install groq")

# =========================
# ---- Groq Setup ----------
# =========================
def get_groq_client():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            pass
    if not key:
        raise RuntimeError("GROQ_API_KEY missing. Please set environment variable or Streamlit secret.")
    try:
        return Groq(api_key=key)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Groq client: {e}")

MODEL_NAME = "llama-3.3-70b-versatile"

# =========================
# ---- Safety / Rules -----
# =========================
# Allow WITH + SELECT only. Forbid mutations & dangerous pragmas.
DANGEROUS = [
    r"\bDROP\b", r"\bDELETE\b", r"\bUPDATE\b", r"\bINSERT\b", r"\bALTER\b",
    r"\bATTACH\b", r"\bDETACH\b", r"\bVACUUM\b", r"\bPRAGMA\b",
    r"\bREINDEX\b", r"\bANALYZE\b", r"\bCREATE\s+(?!VIEW\b)",  # allow CREATE VIEW? No.
]

# Canonical tables we *prefer* to mention in prompts if present
STAR_FAVORITES = [
    "FactSales", "DimCustomer", "DimProduct", "DimDate",
    "FactReturns", "FactMarketing", "SalesDataMart"
]

# =========================
# ---- Schema Helpers -----
# =========================
def normalize_ident(x: str) -> str:
    return re.sub(r"\W+", "_", x.strip())

def get_schema_from_sqlite(conn: sqlite3.Connection) -> Dict[str, List[str]]:
    """Introspect SQLite to a {table: [columns]} dict, excluding sqlite_* internals."""
    schema: Dict[str, List[str]] = {}
    cur = conn.cursor()
    tables = cur.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view') "
        "AND name NOT LIKE 'sqlite_%';"
    ).fetchall()
    for (tname,) in tables:
        cols = cur.execute(f"PRAGMA table_info('{tname}')").fetchall()
        schema[tname] = [c[1] for c in cols] if cols else []
    return schema

def build_allowed_list(schema: Dict[str, List[str]]) -> List[str]:
    present = [t for t in STAR_FAVORITES if t in schema]
    # add others in stable order
    the_rest = [t for t in sorted(schema.keys()) if t not in present]
    return present + the_rest

def table_columns(schema: Dict[str, List[str]]) -> Dict[str, Set[str]]:
    return {t: {c.lower() for c in cols} for t, cols in schema.items()}

def guess_fk_pairs(schema: Dict[str, List[str]]) -> List[Tuple[str, str, str]]:
    """
    Heuristics to guess joinable columns across tables.
    Returns tuples: (tableA.col, '=', tableB.col) normalized to 'Table.Column' strings.
    Rules:
    - Same column name in different tables (case-insensitive).
    - *_ID pattern (e.g., Customer_ID) shared across tables.
    - Common keys: Customer_ID, Product_ID, Date_ID, Order_ID, Campaign_ID, Return_ID, State, Region.
    """
    cols_by_name: Dict[str, List[Tuple[str, str]]] = {}
    for t, cols in schema.items():
        for c in cols:
            cols_by_name.setdefault(c.lower(), []).append((t, c))

    candidate_names = set(cols_by_name.keys())
    common_keys = {
        "customer_id", "product_id", "date_id", "order_id",
        "campaign_id", "return_id",
        "state", "region", "category", "sub_category"
    }
    # Include all *_id
    for name in list(candidate_names):
        if name.endswith("_id"):
            common_keys.add(name)

    joins: Set[Tuple[str, str, str]] = set()
    for key in common_keys:
        pairs = cols_by_name.get(key, [])
        if len(pairs) >= 2:
            # all pairwise combinations
            for i in range(len(pairs)):
                for j in range(i + 1, len(pairs)):
                    (t1, c1), (t2, c2) = pairs[i], pairs[j]
                    if t1 != t2:
                        left = f"{t1}.{c1}"
                        right = f"{t2}.{c2}"
                        # store normalized ordering (Fact* tends to the left)
                        if t1.startswith("Fact") and not t2.startswith("Fact"):
                            joins.add((left, "=", right))
                        elif t2.startswith("Fact") and not t1.startswith("Fact"):
                            joins.add((right, "=", left))
                        else:
                            joins.add((left, "=", right))
    # Optional: add Superstore-ish joins when both columns exist
    def add_if_both(t1, c1, t2, c2):
        if t1 in schema and t2 in schema and (c1 in schema[t1]) and (c2 in schema[t2]):
            joins.add((f"{t1}.{c1}", "=", f"{t2}.{c2}"))
    # Encourage star
    add_if_both("FactSales", "Customer_ID", "DimCustomer", "Customer_ID")
    add_if_both("FactSales", "Product_ID", "DimProduct", "Product_ID")
    add_if_both("FactSales", "Date_ID", "DimDate", "Date_ID")
    add_if_both("FactReturns", "Order_ID", "FactSales", "Order_ID")
    add_if_both("FactMarketing", "Campaign_ID", "FactSales", "Campaign_ID")

    return sorted(list(joins))

def format_join_hints(joins: List[Tuple[str, str, str]]) -> str:
    if not joins:
        return "Joins:\n- (No inferred joins. Use matching key columns like *_ID.)\n"
    lines = [f"- {a} {op} {b}" for a, op, b in joins]
    return "Joins:\n" + "\n".join(lines) + "\n"

# =========================
# ---- SQL Normalizers ----
# =========================
def allow_with_select_only(sql: str) -> Tuple[bool, str]:
    if not sql or not sql.strip():
        return False, "Empty SQL query"
    
    # Strip comments
    s = re.sub(r"--.*?$", "", sql, flags=re.M)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S).strip()
    
    # Remove extra whitespace
    s = re.sub(r'\s+', ' ', s)
    
    # Must be one statement
    stmts = [str(x).strip() for x in sqlparse.parse(s) if str(x).strip()]
    if len(stmts) != 1:
        return False, "Multiple SQL statements detected."

    # First token: WITH or SELECT
    if not re.match(r"^\s*(WITH|SELECT)\b", s, re.I):
        return False, "Only WITH/SELECT queries are allowed."

    # Forbid dangerous patterns
    for pat in DANGEROUS:
        if re.search(pat, s.upper()):
            return False, f"Forbidden keyword: {pat}"

    # Additional safety checks
    if re.search(r'--|/\*|\*/', s):
        return False, "Comments not allowed in SQL"
    
    # Check for basic SQL structure
    if not re.search(r'\bFROM\b', s, re.I):
        return False, "SQL must contain FROM clause"

    return True, "OK"

# Table name canonicalization & synonyms (bi-directional-ish)
def canonicalize_tables(sql: str, schema: Dict[str, List[str]]) -> str:
    mappings = {
        # facts
        "factsales": "FactSales",
        "fact_sales": "FactSales",
        "sales": "FactSales",
        "facts": "FactSales",
        "factreturns": "FactReturns",
        "fact_returns": "FactReturns",
        "returns": "FactReturns",
        "factmarketing": "FactMarketing",
        "fact_marketing": "FactMarketing",
        "marketing": "FactMarketing",
        # dims
        "dimproduct": "DimProduct",
        "products": "DimProduct",
        "product": "DimProduct",
        "dimcustomer": "DimCustomer",
        "customers": "DimCustomer",
        "customer": "DimCustomer",
        "dimdate": "DimDate",
        "date": "DimDate",
        "dates": "DimDate",
        # view(s)
        "salesdatamart": "SalesDataMart",
        "sales_data_mart": "SalesDataMart",
        "datamart": "SalesDataMart",
    }
    for src, target in mappings.items():
        sql = re.sub(rf"\b{src}\b", target, sql, flags=re.I)
    return sql

def parse_tables(sql: str) -> List[str]:
    s = re.sub(r"[\\\n\r]+", " ", sql)
    toks = re.split(r"\s+", s)
    out = []
    for i, tok in enumerate(toks):
        if tok.lower() in ("from", "join") and i + 1 < len(toks):
            tab = re.sub(r"[,();]", "", toks[i + 1])
            out.append(tab)
    return out

def schema_grounding_check(sql: str, allowed_present: Iterable[str]) -> Tuple[bool, str]:
    tables_in_sql = {t.lower() for t in parse_tables(sql)}
    allowed = {t.lower() for t in allowed_present}
    return (len(tables_in_sql & allowed) > 0, "OK" if (len(tables_in_sql & allowed) > 0) else "No known table referenced.")

def table_columns_lower(schema: Dict[str, List[str]]) -> Dict[str, Set[str]]:
    return {t: {c.lower() for c in cols} for t, cols in schema.items()}

def closest_column(name: str, candidates: Set[str]) -> Optional[str]:
    n = name.lower()
    if n in candidates:
        return n
    # fuzzy-ish containment
    for c in sorted(candidates, key=len):
        if n in c or c in n:
            return c
    return None

def rewrite_unknown_columns(sql: str, schema: Dict[str, List[str]]) -> str:
    alias_map = {
        "f": "FactSales", "p": "DimProduct", "c": "DimCustomer", "d": "DimDate",
        "r": "FactReturns", "m": "FactMarketing", "v": "SalesDataMart"
    }
    live_cols = table_columns_lower(schema)

    def repl(m):
        ident = m.group(0)  # e.g., f.something
        alias, col = ident.split(".")
        table = alias_map.get(alias.lower())
        if not table or table not in live_cols:
            return ident
        cand = closest_column(col, live_cols[table])
        return f"{alias}.{cand}" if cand else ident

    return re.sub(r"\b([fpcdrmv])\.[A-Za-z_][A-Za-z0-9_]*", repl, sql)

def fix_bad_aggregates(sql: str) -> str:
    for fn in ("SUM", "COUNT", "AVG", "MIN", "MAX"):
        sql = re.sub(rf"\b{fn}\s*\(\s*f\.\s*([A-Za-z_][A-Za-z0-9_]*)\s+f\s*\)", rf"{fn}(f.\1)", sql, flags=re.I)
        sql = re.sub(rf"\b{fn}\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s+f\s*\)", rf"{fn}(\1)", sql, flags=re.I)
    # Common sales synonyms → f.Sales
    sql = re.sub(r"\bSUM\s*\(\s*f\.\s*FactSales\s*\)", "SUM(f.Sales)", sql, flags=re.I)
    sql = re.sub(r"\bSUM\s*\(\s*FactSales\s*\)", "SUM(f.Sales)", sql, flags=re.I)
    sql = re.sub(r"\bSUM\s*\(\s*f\.(revenue|amount|sales_amount|sale)\s*\)", "SUM(f.Sales)", sql, flags=re.I)
    return sql

def enforce_aliases(sql: str, schema: Dict[str, List[str]]) -> str:
    # ensure common aliases exist when tables referenced without alias
    alias_pairs = {
        "FactSales": "f",
        "DimProduct": "p",
        "DimCustomer": "c",
        "DimDate": "d",
        "FactReturns": "r",
        "FactMarketing": "m",
        "SalesDataMart": "v",
    }
    for table, alias in alias_pairs.items():
        if table in schema:
            # If table name appears and not already aliased (table <alias>)
            sql = re.sub(rf"\b{table}\b(?!\s+[A-Za-z]\b)", f"{table} {alias}", sql)
    return sql

def extract_sql(text: str) -> str:
    if not text:
        return ""
    
    print(f"Attempting to extract SQL from: {text[:300]}...")
    
    # Method 1: Try to extract JSON with sql field
    try:
        # Look for JSON pattern more flexibly
        json_patterns = [
            r'\{[^{}]*"sql"[^{}]*\}',
            r'\{[^{}]*\'sql\'[^{}]*\}',
            r'\{"sql"\s*:\s*"[^"]*"\}',
            r'\{\'sql\':\s*\'[^\']*\'\}'
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                try:
                    obj = json.loads(match.group())
                    sql = obj.get("sql") or obj.get('sql')
                    if sql:
                        sql = str(sql).strip()
                        # Clean up quotes if present
                        sql = re.sub(r'^["\']|["\']$', '', sql)
                        print(f"Extracted from JSON: {sql}")
                        return sql
                except:
                    continue
    except Exception as e:
        print(f"JSON extraction failed: {e}")
    
    # Method 2: Direct SQL extraction - look for WITH or SELECT
    try:
        # Remove common artifacts
        clean_text = re.sub(r'[\{\}\[\]"\'`]', '', text)
        clean_text = re.sub(r'\\n', ' ', clean_text)
        clean_text = re.sub(r'\\', '', clean_text)
        
        # Look for SQL statements
        patterns = [
            r'((WITH|SELECT)[\s\S]*?)(?:;|$)',
            r'(WITH\s+[\s\S]*?SELECT[\s\S]*?)(?:;|$)',
            r'(SELECT[\s\S]*?FROM[\s\S]*?)(?:;|$)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, clean_text, re.IGNORECASE)
            if match:
                sql = match.group(1).strip()
                # Clean up extra whitespace
                sql = re.sub(r'\s+', ' ', sql)
                # Ensure it ends properly
                if not sql.endswith(';'):
                    sql += ';'
                print(f"Extracted directly: {sql}")
                return sql
    except Exception as e:
        print(f"Direct extraction failed: {e}")
    
    # Method 3: Very basic fallback
    try:
        # Just find anything that looks like SELECT
        if 'SELECT' in text.upper():
            # Extract everything from SELECT to the end or semicolon
            select_start = text.upper().find('SELECT')
            if select_start != -1:
                sql_part = text[select_start:]
                # Find the end
                end_patterns = [';', '\n\n', '"', '}']
                for end_pat in end_patterns:
                    end_pos = sql_part.find(end_pat)
                    if end_pos != -1:
                        sql_part = sql_part[:end_pos]
                        break
                sql = sql_part.strip()
                if sql and not sql.endswith(';'):
                    sql += ';'
                print(f"Basic extraction: {sql}")
                return sql
    except Exception as e:
        print(f"Basic extraction failed: {e}")
    
    print("No SQL could be extracted")
    return ""

# =========================
# ---- Prompting ----------
# =========================
def build_schema_prompt(
    schema: Dict[str, List[str]],
    user_q: str,
    allowed_present: List[str],
    join_hints_text: str,
    prev_turn: Optional[Dict] = None
) -> str:
    rules = (
        "You convert business questions into ONE valid SQLite WITH/SELECT query.\n"
        f"- Use ONLY these tables/views: {', '.join(allowed_present)}; never invent names.\n"
        f"- {join_hints_text}"
        "- Prefer aliases: FactSales f, DimProduct p, DimCustomer c, DimDate d, "
        "FactReturns r, FactMarketing m, SalesDataMart v.\n"
        "- If a pre-joined view (e.g., SalesDataMart) answers the question directly, you may query it alone.\n"
        "- Quarters: use DimDate.Order_Date with strftime for quarter math.\n"
        "- Add LIMIT unless user asks for full dataset.\n"
        "- Never invent columns. If unsure, choose the closest existing column.\n"
        "- **RETURNS RULE**: When the question mentions return/returns/returned, compute metrics from FactReturns (alias r). "
        "Use the returns amount column from FactReturns, not FactSales.Sales.\n"
        "- **TOP-N RULE**: For requests like 'top N customers/products by X', return a tabular result grouped by that entity, "
        "with an explicit ORDER BY the metric DESC and LIMIT N.\n"
        "- **DATE FILTERS**: For date-based queries, use proper date functions: "
        "strftime('%Y', DimDate.Order_Date) for year, "
        "strftime('%m', DimDate.Order_Date) for month, "
        "strftime('%Y-%m', DimDate.Order_Date) for year-month.\n"
        "- **AGGREGATION RULES**: Use appropriate aggregations - SUM for totals, AVG for averages, "
        "COUNT for counts, MAX/MIN for extremes. Always use proper column names.\n"
        "- Do NOT use JSON_OBJECT, JSON_ARRAY, JSON_GROUP_ARRAY, or JSON extraction.\n"
        "- Always return raw tabular rows using SELECT columns (rows & columns, not a JSON blob).\n"
        'Return ONLY JSON wrapper for SQL like: {"sql":"..."}'
    )

    # keep column map concise but real
    col_map = "Column map:\n" + "\n".join(
        [f"{t}: {', '.join(schema[t])}" for t in allowed_present if t in schema]
    )
    prev = f"Previous Q: {prev_turn.get('q','')}\nPrevious SQL: {prev_turn.get('sql','')}\n\n" if prev_turn else ""
    return f"{prev}{rules}\n\n{col_map}\n\nUser question: {user_q}"

def call_groq_chat(prompt: str) -> str:
    try:
        client = get_groq_client()
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Convert English to one safe SQLite WITH/SELECT; return JSON {\"sql\": \"...\"}."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=768,
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"Groq API error: {e}")
        return ""

FALLBACKS = [
    (r"\btop\s*\d+\s*products\b.*\bsales\b",
     "SELECT p.Product_Name, SUM(f.Sales) AS total_sales "
     "FROM FactSales f JOIN DimProduct p ON f.Product_ID = p.Product_ID "
     "GROUP BY p.Product_Name ORDER BY total_sales DESC LIMIT {k};"),
    (r"\breturns\b.*\b(by|per)\b.*\b(product|category)\b",
     "SELECT p.Category, COALESCE(COUNT(r.Return_ID),0) AS returns_count "
     "FROM DimProduct p "
     "LEFT JOIN FactSales f ON f.Product_ID = p.Product_ID "
     "LEFT JOIN FactReturns r ON r.Order_ID = f.Order_ID "
     "GROUP BY p.Category ORDER BY returns_count DESC LIMIT {k};"),
    (r"\bsales\s+by\s+(month|year|quarter)\b",
     "SELECT strftime('%Y-%m', d.Order_Date) AS period, SUM(f.Sales) AS total_sales "
     "FROM FactSales f JOIN DimDate d ON f.Date_ID = d.Date_ID "
     "GROUP BY period ORDER BY period DESC LIMIT 12;"),
    (r"\bcustomer\s+segmentation\b",
     "SELECT c.Customer_Segment, COUNT(DISTINCT c.Customer_ID) AS customer_count, SUM(f.Sales) AS total_sales "
     "FROM FactSales f JOIN DimCustomer c ON f.Customer_ID = c.Customer_ID "
     "GROUP BY c.Customer_Segment ORDER BY total_sales DESC;"),
    (r"\bprofit\s+margin\b",
     "SELECT p.Category, (SUM(f.Sales) - SUM(f.Cost)) / SUM(f.Sales) * 100 AS profit_margin "
     "FROM FactSales f JOIN DimProduct p ON f.Product_ID = p.Product_ID "
     "GROUP BY p.Category ORDER BY profit_margin DESC;"),
    (r"\bsales\s+by\s+category\b",
     "SELECT p.Category, SUM(f.Sales) AS total_sales "
     "FROM FactSales f JOIN DimProduct p ON f.Product_ID = p.Product_ID "
     "GROUP BY p.Category ORDER BY total_sales DESC;"),
    (r"\bsales\s+by\s+region\b",
     "SELECT c.Region, SUM(f.Sales) AS total_sales "
     "FROM FactSales f JOIN DimCustomer c ON f.Customer_ID = c.Customer_ID "
     "GROUP BY c.Region ORDER BY total_sales DESC;"),
    (r"\bcustomer\s+count\b",
     "SELECT COUNT(DISTINCT c.Customer_ID) AS customer_count "
     "FROM FactSales f JOIN DimCustomer c ON f.Customer_ID = c.Customer_ID;"),
    (r"\border\s+count\b",
     "SELECT COUNT(DISTINCT f.Order_ID) AS order_count "
     "FROM FactSales f;"),
]

def fallback_sql_for(q: str) -> Optional[str]:
    ql = q.lower()
    print(f"Trying fallback for: {q}")
    
    for pat, tmpl in FALLBACKS:
        if re.search(pat, ql):
            k = 10
            m = re.search(r"top\s*(\d+)", ql)
            if m:
                k = int(m.group(1))
            result = tmpl.format(k=k)
            print(f"Fallback matched pattern {pat}: {result}")
            return result
    
    # Additional generic fallbacks
    if "sales" in ql and "total" in ql:
        result = "SELECT SUM(f.Sales) AS total_sales FROM FactSales f LIMIT 1;"
        print(f"Generic sales fallback: {result}")
        return result
    
    if "count" in ql and ("orders" in ql or "order" in ql):
        result = "SELECT COUNT(DISTINCT f.Order_ID) AS order_count FROM FactSales f LIMIT 1;"
        print(f"Generic order count fallback: {result}")
        return result
        
    if "count" in ql and ("customers" in ql or "customer" in ql):
        result = "SELECT COUNT(DISTINCT c.Customer_ID) AS customer_count FROM FactSales f JOIN DimCustomer c ON f.Customer_ID = c.Customer_ID LIMIT 1;"
        print(f"Generic customer count fallback: {result}")
        return result
    
    if "average" in ql or "avg" in ql:
        if "sales" in ql:
            result = "SELECT AVG(f.Sales) AS avg_sales FROM FactSales f LIMIT 1;"
            print(f"Generic average sales fallback: {result}")
            return result
    
    # Last resort - basic sales query
    if "sales" in ql:
        result = "SELECT SUM(f.Sales) AS total_sales, COUNT(DISTINCT f.Order_ID) AS order_count FROM FactSales f LIMIT 1;"
        print(f"Last resort sales fallback: {result}")
        return result
    
    print("No fallback found")
    return None

# =========================
# ---- Insight Prompt -----
# =========================
def build_insight_prompt(user_q, sql, df, max_rows: int = 10) -> str:
    sample_df = df.head(max_rows)
    return (
        "Write 3 plain-English lines summarizing the numbers.\n\n"
        f"User question: {user_q}\nSQL: {sql}\nRows:\n{sample_df.to_json()}"
    )

def llm_summarize(prompt: str) -> str:
    try:
        client = get_groq_client()
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Summarize in 3 simple lines."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=128
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM summarization error: {e}")
        return "Unable to generate summary due to API error."

# =========================
# ---- NL→SQL Pipeline ----
# =========================
def nl_to_sql_and_insight(
    user_q: str,
    schema: Dict[str, List[str]],
    allowed_present: Optional[List[str]] = None,
    prev_turn: Optional[Dict] = None
):
    """
    Returns: dict(sql, df, summary, error)
    """
    result = {"sql": None, "df": None, "summary": None, "error": None}

    # allow list
    allowed_present = allowed_present or build_allowed_list(schema)

    # dynamic join hints
    join_pairs = guess_fk_pairs(schema)
    join_hints_text = format_join_hints(join_pairs)

    # Build prompt
    prompt = build_schema_prompt(schema, user_q, allowed_present, join_hints_text, prev_turn)

    # Call LLM
    raw = call_groq_chat(prompt)
    
    # Debug logging
    print(f"LLM Raw Response: {raw[:200]}...")
    
    sql = extract_sql(raw) or fallback_sql_for(user_q)
    if not sql:
        result["error"] = "Could not extract SQL from LLM response."
        print(f"Failed to extract SQL from: {raw}")
        return result

    print(f"Extracted SQL: {sql}")

    # Safety
    ok, msg = allow_with_select_only(sql)
    if not ok:
        result["error"] = f"Safety check failed: {msg}"
        result["sql"] = sql
        print(f"Safety check failed: {msg} for SQL: {sql}")
        return result

    # Normalize & repair
    sql = canonicalize_tables(sql, schema)
    ok2, _ = schema_grounding_check(sql, allowed_present)
    if not ok2:
        # one-shot repair asking the model to re-ground
        repair = (
            "Repair this SQL to use ONLY allowed tables/views.\n"
            f"Allowed: {', '.join(allowed_present)}\n"
            f"Original:\n{sql}"
        )
        raw2 = call_groq_chat(repair)
        sql2 = extract_sql(raw2)
        if sql2:
            sql = canonicalize_tables(sql2, schema)

    # make flat, enforce aliases, rewrite columns, clean aggs
    sql = re.sub(r"[\\\n\r]+", " ", sql).strip()
    sql = enforce_aliases(sql, schema)
    sql = rewrite_unknown_columns(sql, schema)
    sql = fix_bad_aggregates(sql)

    result["sql"] = sql
    return result

# =========================
# ---- Optional Runner ----
# =========================
def run_query_and_summarize(conn: sqlite3.Connection, user_q: str, prev_turn: Optional[Dict] = None):
    """
    Convenience: introspects schema, generates SQL, executes, and summarizes.
    """
    schema = get_schema_from_sqlite(conn)
    res = nl_to_sql_and_insight(user_q, schema, None, prev_turn)

    if res.get("error"):
        return res

    # Execute
    try:
        df = pd.read_sql_query(res["sql"], conn)
        res["df"] = df
    except Exception as e:
        res["error"] = f"Execution error: {e}"
        return res

    # Summarize
    try:
        prompt = build_insight_prompt(user_q, res["sql"], res["df"])
        res["summary"] = llm_summarize(prompt)
    except Exception as e:
        res["summary"] = None  # Not fatal

    return res

# =========================
# ---- Minimal Usage ------
# =========================
"""
# Example (with a live DB):
conn = sqlite3.connect("retail.sqlite")
answer = run_query_and_summarize(conn, "Show top 5 categories by sales and returns rate")
print(answer["sql"])
print(answer["summary"])
print(answer["df"].head())
"""
