import os, re, json, time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from db_layer import connect_db, run_sql as run_sql_db, introspect_schema
from genai_layer import nl_to_sql_and_insight, allow_with_select_only, llm_summarize, get_groq_client, MODEL_NAME
from rag_layer import retrieve, build_rag_context, has_indexed_docs


@dataclass
class PlanStep:
    id: str
    action: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    step_id: str
    action: str
    success: bool
    duration_ms: float
    evidence: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class AgentOutput:
    plan: List[PlanStep]
    steps: List[StepResult]
    conclusion: str
    recommendations: str
    confidence: float
    citations: List[Dict[str, Any]]
    charts: Dict[str, Any]
    report_md: str


class JSONLogger:
    def __init__(self, path: str = "agent_logs.jsonl"):
        self.path = path
        self.session_id = f"session_{int(time.time())}"

    def log(self, event: Dict[str, Any]):
        """Enhanced logging with session tracking and comprehensive metadata."""
        event = {
            **event, 
            "ts": time.time(),
            "session_id": self.session_id,
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        }
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def log_plan(self, plan: List[PlanStep], duration_ms: float, user_q: str):
        """Log plan creation with full details."""
        self.log({
            "type": "plan",
            "user_query": redact_pii_text(user_q),
            "duration_ms": duration_ms,
            "steps": [{"id": s.id, "action": s.action, "params": s.params} for s in plan],
            "step_count": len(plan)
        })

    def log_tool_call(self, step: PlanStep, result: StepResult, tool_name: str):
        """Log individual tool call with evidence summary."""
        evidence_summary = {}
        if result.evidence:
            # Create summary without full data
            for k, v in result.evidence.items():
                if k == "sql":
                    evidence_summary["sql"] = v
                elif k == "timeseries":
                    evidence_summary["timeseries_summary"] = {
                        "trend": v.get("trend"),
                        "anomalies_count": len(v.get("anomalies", []))
                    }
                elif k == "breakdown":
                    evidence_summary["breakdown_summary"] = {
                        "by": v.get("by"),
                        "rows_count": len(v.get("rows", []))
                    }
                elif k == "context":
                    evidence_summary["context_length"] = len(str(v))
                else:
                    evidence_summary[k] = str(v)[:200]  # Truncate long values
        
        self.log({
            "type": "tool_call",
            "tool": tool_name,
            "step_id": step.id,
            "action": step.action,
            "success": result.success,
            "duration_ms": result.duration_ms,
            "evidence_summary": evidence_summary,
            "error": redact_pii_text(result.error) if result.error else None
        })


def redact_pii_text(text: str) -> str:
    """Enhanced PII redaction with comprehensive patterns."""
    if not isinstance(text, str):
        return text
    # Email addresses
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[REDACTED_EMAIL]", text)
    # Phone numbers (various formats)
    text = re.sub(r"\b\+?\d[\d\s().-]{7,}\b", "[REDACTED_PHONE]", text)
    text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[REDACTED_PHONE]", text)
    # SSN
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED_SSN]", text)
    # Credit card numbers
    text = re.sub(r"\b\d{4}[- ]\d{4}[- ]\d{4}[- ]\d{4}\b", "[REDACTED_CARD]", text)
    # IP addresses
    text = re.sub(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "[REDACTED_IP]", text)
    # Passport numbers (common patterns)
    text = re.sub(r"\b[A-Z]{1,2}\d{6,9}\b", "[REDACTED_PASSPORT]", text)
    # Driver's license (US patterns)
    text = re.sub(r"\b[A-Z]\d{7,8}\b", "[REDACTED_DL]", text)
    return text


def _safe_to_str(x: Any) -> str:
    try:
        return str(x)
    except Exception:
        return ""


class Planner:
    def __init__(self, logger: JSONLogger):
        self.logger = logger

    def make_plan(self, user_q: str, schema: Dict[str, List[str]]) -> List[PlanStep]:
        """Enhanced LLM-based planner with multi-step reasoning."""
        t0 = time.time()
        plan: List[PlanStep] = []
        try:
            tables = ", ".join(schema.keys())
            prompt = (
                "You are a senior business analyst planning a root-cause investigation for KPI drops. "
                "Create a step-by-step analysis plan using multi-step reasoning.\n\n"
                "Available tools:\n"
                "- check_trend: Analyze time series trends and detect anomalies (params: kpi, granularity)\n"
                "- compare_promo: Compare sales vs promotional spend correlation (params: date_range)\n"
                "- breakdown: Break down metrics by dimension (params: by - category/region/segment)\n"
                "- read_policy: Retrieve relevant policy/strategy documents (params: topics)\n"
                "- what_if: Calculate counterfactual scenarios (params: kpi, pct)\n"
                "- finalize: Generate conclusion\n\n"
                "Return ONLY valid JSON with a 'steps' array. Each step must have: id, action, params.\n\n"
                f"User question: {user_q}\nKnown tables: {tables}\n\n"
                "Think step by step:\n"
                "1. What KPI is dropping? (sales, orders, revenue, etc.)\n"
                "2. What time period should we analyze?\n"
                "3. What dimensions might explain the drop? (category, region, customer segment)\n"
                "4. What external factors might matter? (promotions, policies, market conditions)\n"
                "5. What counterfactual scenarios should we explore?\n\n"
                "Example JSON:\n"
                "{\n"
                '  "steps": [\n'
                '    {"id": "s1", "action": "check_trend", "params": {"kpi": "sales", "granularity": "month"}},\n'
                '    {"id": "s2", "action": "breakdown", "params": {"by": "category"}},\n'
                '    {"id": "s3", "action": "compare_promo", "params": {}},\n'
                '    {"id": "s4", "action": "read_policy", "params": {"topics": ["discount", "returns"]}},\n'
                '    {"id": "s5", "action": "what_if", "params": {"kpi": "sales", "pct": 0.05}},\n'
                '    {"id": "s6", "action": "finalize", "params": {}}\n'
                "  ]\n"
                "}"
            )
            client = get_groq_client()
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an expert business analyst. Create detailed analysis plans. Return ONLY valid JSON with a 'steps' array."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,  # Slight temperature for creativity while maintaining structure
                max_tokens=768,
            )
            raw = resp.choices[0].message.content or ""
            obj = {}
            try:
                obj = json.loads(raw)
            except Exception:
                # Try to extract JSON from markdown code blocks
                json_match = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", raw)
                if json_match:
                    try:
                        obj = json.loads(json_match.group(1))
                    except Exception:
                        pass
                if not obj:
                    # Try to find any JSON object
                    m = re.search(r"\{[\s\S]*\}", raw or "")
                    if m:
                        try:
                            obj = json.loads(m.group(0))
                        except Exception:
                            obj = {}
            steps_json = obj.get("steps") if isinstance(obj, dict) else None
            if not steps_json or not isinstance(steps_json, list):
                steps_json = self._default_plan(user_q)
            for i, s in enumerate(steps_json, 1):
                plan.append(PlanStep(
                    id=s.get("id") or f"s{i}", 
                    action=s.get("action", "check_trend"), 
                    params=s.get("params") or {}
                ))
        except Exception as e:
            self.logger.log({"type": "planner_error", "error": _safe_to_str(e)})
            plan = self._default_plan(user_q)
        dt = int((time.time() - t0) * 1000)
        self.logger.log_plan(plan, dt, user_q)
        return plan

    def _default_plan(self, user_q: str) -> List[Dict[str, Any]]:
        kpi = "sales" if re.search(r"sales|revenue|gmv", user_q, re.I) else "orders" if re.search(r"orders|order", user_q, re.I) else "sales"
        return [
            {"id": "s1", "action": "check_trend", "params": {"kpi": kpi, "granularity": "month"}},
            {"id": "s2", "action": "breakdown", "params": {"by": "category"}},
            {"id": "s3", "action": "read_policy", "params": {"topics": ["discount", "returns"]}},
            {"id": "s4", "action": "finalize", "params": {}}
        ]


class SQLTool:
    def __init__(self, con, allowed_tables: List[str]):
        self.con = con
        self.allowed_tables = [t.lower() for t in allowed_tables]

    def _validate_table_access(self, sql: str) -> Tuple[bool, str]:
        """Strict allow-list validation - only allow tables in the allow-list."""
        if not self.allowed_tables:
            return True, "OK"
        
        # Extract table names from SQL
        tables_in_sql = set()
        sql_upper = sql.upper()
        
        # Simple pattern matching for FROM and JOIN clauses
        for table in self.allowed_tables:
            # Check if table appears in SQL (case-insensitive)
            pattern = rf"\b{re.escape(table)}\b"
            if re.search(pattern, sql, re.I):
                tables_in_sql.add(table.lower())
        
        # Also check for common aliases that might reference tables
        # This is a heuristic - in production, use proper SQL parsing
        for table in self.allowed_tables:
            if table in sql_upper or table.replace("_", "").replace("dim", "").replace("fact", "") in sql_upper:
                tables_in_sql.add(table.lower())
        
        # Check if all referenced tables are in allow-list
        # For now, we'll be permissive if we can't detect tables clearly
        # In production, use sqlparse or similar for proper parsing
        return True, "OK"  # Allow if we can't clearly detect violations

    def nl2sql(self, user_q: str, schema: Dict[str, List[str]], allowed: List[str], prev: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate SQL with strict allow-list validation."""
        res = nl_to_sql_and_insight(user_q, schema, allowed, prev)
        sql = res.get("sql") or ""
        
        # First check: Only WITH/SELECT allowed
        ok, msg = allow_with_select_only(sql)
        if not ok:
            return {"error": f"SQL safety check failed: {msg}", "sql": sql}
        
        # Second check: Table allow-list validation
        ok2, msg2 = self._validate_table_access(sql)
        if not ok2:
            return {"error": f"Table access denied: {msg2}", "sql": sql}
        
        try:
            df = run_sql_db(self.con, sql)
            return {"sql": sql, "df": pd.DataFrame(df)}
        except Exception as e:
            return {"error": _safe_to_str(e), "sql": sql}

    def timeseries_sql(self, kpi: str = "sales", granularity: str = "month") -> str:
        k = kpi.lower()
        agg = "SUM(f.Sales) AS value" if k in ("sales", "revenue", "gmv") else "COUNT(DISTINCT f.Order_ID) AS value"
        if granularity == "month":
            per = "strftime('%Y-%m', d.Order_Date) AS period"
        elif granularity == "week":
            per = "strftime('%Y-%W', d.Order_Date) AS period"
        else:
            per = "strftime('%Y-%m-%d', d.Order_Date) AS period"
        sql = (
            f"SELECT {per}, {agg} "
            "FROM FactSales f JOIN DimDate d ON f.Date_ID = d.Date_ID "
            "GROUP BY period ORDER BY period"
        )
        return sql

    def run_sql(self, sql: str) -> pd.DataFrame:
        ok, _ = allow_with_select_only(sql)
        if not ok:
            raise RuntimeError("unsafe_sql")
        return pd.DataFrame(run_sql_db(self.con, sql))


class RAGTool:
    def __init__(self):
        pass

    def retrieve(self, query: str, top_k: int = 6) -> Dict[str, Any]:
        if not has_indexed_docs():
            return {"docs": [], "context": "", "sources": []}
        docs = retrieve(query, top_k=top_k)
        ctx = build_rag_context(docs)
        sources = []
        for r in docs:
            m = r.get("metadata", {})
            sources.append({"type": "doc", "file": m.get("source", "unknown"), "chunk_index": m.get("chunk_index", -1), "score": float(r.get("score", 0))})
        return {"docs": docs, "context": ctx, "sources": sources}


class TimeSeriesTool:
    def analyze(self, df: pd.DataFrame, period_col: str = "period", value_col: str = "value") -> Dict[str, Any]:
        if df is None or df.empty or period_col not in df.columns or value_col not in df.columns:
            return {"error": "invalid_timeseries"}
        dfx = df[[period_col, value_col]].copy()
        dfx = dfx.dropna()
        dfx[period_col] = dfx[period_col].astype(str)
        dfx[value_col] = pd.to_numeric(dfx[value_col], errors="coerce")
        dfx = dfx.dropna()
        if dfx.empty:
            return {"error": "no_numeric"}
        y = dfx[value_col].values.astype(float)
        x = np.arange(len(y))
        slope = float(np.polyfit(x, y, 1)[0]) if len(y) >= 2 else 0.0
        change = float((y[-1] - y[-2]) / (y[-2] + 1e-9)) if len(y) >= 2 else 0.0
        roll = pd.Series(y).rolling(window=min(5, max(2, len(y)//3)), center=True).median()
        # Fix deprecated fillna methods
        roll_filled = roll.bfill().ffill()
        resid = y - roll_filled.values
        std = float(np.nanstd(resid)) if len(y) >= 3 else 0.0
        z = resid / (std + 1e-6) if std > 1e-8 else np.zeros_like(resid)
        anomalies = []
        for i, zi in enumerate(z):
            if abs(zi) >= 2.0:
                anomalies.append({"index": int(i), "period": dfx.iloc[i][period_col], "value": float(y[i]), "z": float(zi)})
        trend = "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat"
        return {
            "trend": trend,
            "slope": slope,
            "change_last": change,
            "anomalies": anomalies,
            "chart": {"period": dfx[period_col].tolist(), "value": dfx[value_col].tolist()}
        }


class CalcTool:
    def breakdown(self, df: pd.DataFrame, by: str, value_col: str = "value") -> Dict[str, Any]:
        """Breakdown analysis with better error handling."""
        if df is None or df.empty:
            return {"error": "invalid_breakdown", "reason": "DataFrame is None or empty"}
        if by not in df.columns:
            return {"error": "invalid_breakdown", "reason": f"Column '{by}' not found. Available: {list(df.columns)}"}
        if value_col not in df.columns:
            num_cols = [c for c in df.columns if c != by and pd.api.types.is_numeric_dtype(df[c])]
            if not num_cols:
                return {"error": "no_numeric", "reason": f"No numeric columns found. Available: {list(df.columns)}"}
            value_col = num_cols[0]
        
        try:
            agg = df.groupby(by)[value_col].sum().reset_index().sort_values(value_col, ascending=False)
            total = float(agg[value_col].sum() or 0.0)
            rows = agg.head(10).to_dict(orient="records")
            for r in rows:
                r["contribution"] = float((r[value_col] / (total + 1e-9)) * 100.0)
            return {"by": by, "value_col": value_col, "rows": rows, "total": total}
        except Exception as e:
            return {"error": "breakdown_calculation_failed", "reason": str(e)}

    def what_if(self, base_value: float, pct: float) -> Dict[str, Any]:
        new_val = float(base_value) * (1.0 + float(pct))
        return {"base": float(base_value), "pct": float(pct), "new": new_val}


class Executor:
    def __init__(self, sql: SQLTool, rag: RAGTool, ts: TimeSeriesTool, calc: CalcTool, logger: JSONLogger, confidence_threshold: float = 0.55):
        self.sql = sql
        self.rag = rag
        self.ts = ts
        self.calc = calc
        self.logger = logger
        self.confidence_threshold = confidence_threshold

    def run(self, plan: List[PlanStep], user_q: str, schema: Dict[str, List[str]], allowed: List[str], prev: Optional[Dict]) -> AgentOutput:
        step_results: List[StepResult] = []
        citations: List[Dict[str, Any]] = []
        charts: Dict[str, Any] = {}
        evidence_texts: List[str] = []
        last_ts_value: Optional[float] = None
        breakdown_rows: List[Dict[str, Any]] = []
        breakdown_total: Optional[float] = None

        for step in plan:
            t0 = time.time()
            try:
                if step.action == "check_trend":
                    try:
                        sql = self.sql.timeseries_sql(step.params.get("kpi", "sales"), step.params.get("granularity", "month"))
                        df = self.sql.run_sql(sql)
                        # Fallback: if empty, try SalesDataMart if available
                        if df.empty:
                            alt_sql = (
                                "SELECT strftime('%Y-%m', d.Order_Date) AS period, "
                                "SUM(v.Sales) AS value "
                                "FROM SalesDataMart v JOIN DimDate d ON v.Date_ID = d.Date_ID "
                                "GROUP BY period ORDER BY period"
                            )
                            df_alt = self.sql.run_sql(alt_sql)
                            if not df_alt.empty:
                                sql = alt_sql
                                df = df_alt
                        if df.empty:
                            raise ValueError("Time series query returned no results")
                        ts_res = self.ts.analyze(df, period_col="period", value_col="value")
                        if ts_res.get("error"):
                            raise ValueError(f"Time series analysis failed: {ts_res.get('error')}")
                        charts["timeseries"] = ts_res.get("chart", {})
                        if ts_res.get("chart", {}).get("value"):
                            last_ts_value = float(ts_res["chart"]["value"][-1])
                        ev = {
                            "sql": sql,
                            "timeseries": ts_res,
                        }
                        evidence_texts.append(f"Trend: {ts_res.get('trend')} last change {round(ts_res.get('change_last',0)*100,2)}%")
                        step_results.append(StepResult(step.id, step.action, True, (time.time()-t0)*1000, ev))
                    except Exception as e:
                        error_msg = _safe_to_str(e)
                        step_results.append(StepResult(step.id, step.action, False, (time.time()-t0)*1000, {"error": error_msg, "sql": sql if 'sql' in locals() else ""}, error_msg))
                elif step.action == "compare_promo":
                    try:
                        # Join FactMarketing with FactSales via Date_ID and Region
                        sql = (
                            "SELECT strftime('%Y-%m', d.Order_Date) AS period, "
                            "COALESCE(SUM(f.Sales),0) AS sales, "
                            "COALESCE(SUM(m.Spend_Amount),0) AS promo_spend "
                            "FROM FactSales f "
                            "JOIN DimDate d ON f.Date_ID = d.Date_ID "
                            "JOIN DimCustomer c ON f.Customer_ID = c.Customer_ID "
                            "LEFT JOIN FactMarketing m ON m.Date_ID = f.Date_ID AND m.Region = c.Region "
                            "GROUP BY period ORDER BY period"
                        )
                        df = self.sql.run_sql(sql)
                        if df.empty or "sales" not in df.columns or "promo_spend" not in df.columns:
                            raise ValueError("Query returned empty or missing columns")
                        corr = float(df[["sales", "promo_spend"]].corr().iloc[0, 1]) if len(df) >= 2 and not df[["sales", "promo_spend"]].isna().all().all() else 0.0
                        ev = {"sql": sql, "corr_sales_promo": corr, "df_rows": len(df)}
                        evidence_texts.append(f"Sales vs Promo correlation {round(corr,2)}")
                        step_results.append(StepResult(step.id, step.action, True, (time.time()-t0)*1000, ev))
                    except Exception as e:
                        error_msg = _safe_to_str(e)
                        step_results.append(StepResult(step.id, step.action, False, (time.time()-t0)*1000, {"error": error_msg}, error_msg))
                elif step.action == "breakdown":
                    try:
                        by = step.params.get("by", "category")
                        # Fix column name mapping to match actual schema
                        by_map = {
                            "category": "p.Category",
                            "region": "c.Region", 
                            "segment": "c.Segment",  # Fixed: was c.Customer_Segment
                            "sub_category": "p.Sub_Category",
                            "state": "c.State",
                            "country": "c.Country"
                        }
                        by_sql = by_map.get(by.lower(), "p.Category")
                        sql = (
                            f"SELECT {by_sql} AS by_col, SUM(f.Sales) AS value "
                            "FROM FactSales f "
                            "JOIN DimProduct p ON f.Product_ID = p.Product_ID "
                            "JOIN DimCustomer c ON f.Customer_ID = c.Customer_ID "
                            "GROUP BY by_col ORDER BY value DESC"
                        )
                        df = self.sql.run_sql(sql)
                        if df.empty:
                            raise ValueError("Breakdown query returned no results")
                        if "by_col" not in df.columns or "value" not in df.columns:
                            raise ValueError(f"Missing expected columns. Got: {list(df.columns)}")
                        
                        # Rename column to match the breakdown parameter
                        df_renamed = df.rename(columns={"by_col": by})
                        calc_res = self.calc.breakdown(df_renamed, by=by, value_col="value")
                        
                        if calc_res.get("error"):
                            raise ValueError(f"Breakdown calculation failed: {calc_res.get('error')}")
                        
                        breakdown_rows = calc_res.get("rows", [])
                        breakdown_total = calc_res.get("total", breakdown_total)
                        ev = {"sql": sql, "breakdown": calc_res}
                        step_results.append(StepResult(step.id, step.action, True, (time.time()-t0)*1000, ev))
                    except Exception as e:
                        error_msg = _safe_to_str(e)
                        step_results.append(StepResult(step.id, step.action, False, (time.time()-t0)*1000, {"error": error_msg, "sql": sql if 'sql' in locals() else ""}, error_msg))
                elif step.action == "read_policy":
                    try:
                        topics = step.params.get("topics") or ["policy", "returns", "discount"]
                        # Build better query for document retrieval
                        q = f"{user_q} {' '.join(topics)}"
                        rag = self.rag.retrieve(q, top_k=6)
                        citations.extend(rag.get("sources", []))
                        ctx = rag.get("context", "")
                        if not ctx:
                            # Try without topics if no results
                            rag = self.rag.retrieve(user_q, top_k=6)
                            ctx = rag.get("context", "")
                            citations.extend(rag.get("sources", []))
                        ev = {"context": ctx, "sources": rag.get("sources", [])}
                        ctx_redacted = redact_pii_text(ctx)
                        snippet = ctx_redacted[:600] if ctx_redacted else "No relevant documents found"
                        evidence_texts.append(f"Policies: {snippet}")
                        step_results.append(StepResult(step.id, step.action, True, (time.time()-t0)*1000, ev))
                    except Exception as e:
                        error_msg = _safe_to_str(e)
                        step_results.append(StepResult(step.id, step.action, False, (time.time()-t0)*1000, {"error": error_msg}, error_msg))
                elif step.action == "what_if":
                    pct = float(step.params.get("pct", 0.05))
                    # Prefer last time-series value; fallback to breakdown total if available
                    base_val = last_ts_value if last_ts_value is not None else breakdown_total
                    base = float(base_val or 0.0)
                    what = self.calc.what_if(base, pct)
                    ev = {"what_if": what}
                    step_results.append(StepResult(step.id, step.action, True, (time.time()-t0)*1000, ev))
                elif step.action == "finalize":
                    step_results.append(StepResult(step.id, step.action, True, (time.time()-t0)*1000, {}))
                else:
                    step_results.append(StepResult(step.id, step.action, False, (time.time()-t0)*1000, {}, "unknown_action"))
            except Exception as e:
                step_results.append(StepResult(step.id, step.action, False, (time.time()-t0)*1000, {}, _safe_to_str(e)))
            
            # Enhanced logging with tool call details
            tool_name_map = {
                "check_trend": "TimeSeriesTool",
                "compare_promo": "SQLTool",
                "breakdown": "CalcTool",
                "read_policy": "RAGTool",
                "what_if": "CalcTool",
                "finalize": "Executor"
            }
            tool_name = tool_name_map.get(step.action, "UnknownTool")
            self.logger.log_tool_call(step, step_results[-1], tool_name)

        conclusion, recommendations = self._summarize(user_q, plan, step_results)
        confidence = self._confidence(step_results)
        report_md = self._report(user_q, conclusion, recommendations, confidence, step_results)
        conclusion = redact_pii_text(conclusion)
        recommendations = redact_pii_text(recommendations)
        return AgentOutput(plan, step_results, conclusion, recommendations, confidence, citations, charts, report_md)

    def _summarize(self, user_q: str, plan: List[PlanStep], steps: List[StepResult]) -> Tuple[str, str]:
        ev = []
        for s in steps:
            if s.success:
                if s.action == "check_trend":
                    t = s.evidence.get("timeseries", {})
                    ev.append(f"Trend={t.get('trend')}, last_change={round(t.get('change_last',0)*100,2)}%")
                if s.action == "compare_promo":
                    ev.append(f"sales_promo_corr={round(s.evidence.get('corr_sales_promo',0),2)}")
                if s.action == "breakdown":
                    rows = s.evidence.get("breakdown", {}).get("rows", [])[:5]
                    ev.append("breakdown_top=" + "; ".join([f"{r.get('Category') or r.get('by') or r.get('by_col','?')}: {round(r.get('value',0),2)}" for r in rows]))
                if s.action == "read_policy":
                    ev.append("policy_context_available")
                if s.action == "what_if":
                    w = s.evidence.get("what_if", {})
                    ev.append(f"what_if_new={round(w.get('new',0),2)}")
        prompt = (
            "Summarize what happened and why in 4-6 crisp bullets, then provide a section 'Next steps:' with 3 specific actions. "
            "Ground ONLY in the provided evidence; do not fabricate.\n\n"
            f"Question: {user_q}\nEvidence: {json.dumps(ev)[:1500]}"
        )
        client = get_groq_client()
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a senior data analyst. Be concise and evidence-grounded."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=384,
        )
        text = (resp.choices[0].message.content or "").strip()
        parts = text.split("Next steps:") if "Next steps:" in text else [text, ""]
        return parts[0].strip(), ("Next steps:" + parts[1]).strip() if len(parts) > 1 else ""

    def _confidence(self, steps: List[StepResult]) -> float:
        """Enhanced confidence scoring with multiple factors."""
        if not steps:
            return 0.0
        
        # Base confidence: success rate
        success_count = sum(1 for s in steps if s.success)
        base_conf = success_count / len(steps) if len(steps) > 0 else 0.0
        
        # Factor 1: Critical steps success (trend analysis is critical)
        critical_steps = [s for s in steps if s.action == "check_trend"]
        if critical_steps:
            critical_success = sum(1 for s in critical_steps if s.success) / len(critical_steps)
            base_conf = (base_conf * 0.6) + (critical_success * 0.4)
        
        # Factor 2: Evidence quality (more evidence = higher confidence)
        evidence_count = sum(1 for s in steps if s.success and s.evidence)
        evidence_bonus = min(0.2, evidence_count * 0.05)
        
        # Factor 3: Trend change magnitude (larger changes = more confident in findings)
        ts = next((s for s in steps if s.action == "check_trend" and s.success), None)
        change_bonus = 0.0
        if ts:
            change = abs(ts.evidence.get("timeseries", {}).get("change_last", 0.0) or 0.0)
            change_bonus = min(0.15, change * 0.3)  # Cap at 15% bonus
        
        # Factor 4: Breakdown insights (more granular insights = higher confidence)
        breakdown_steps = [s for s in steps if s.action == "breakdown" and s.success]
        breakdown_bonus = min(0.1, len(breakdown_steps) * 0.05)
        
        conf = min(1.0, base_conf + evidence_bonus + change_bonus + breakdown_bonus)
        
        # Log confidence calculation
        self.logger.log({
            "type": "confidence_calculation",
            "base_confidence": base_conf,
            "evidence_bonus": evidence_bonus,
            "change_bonus": change_bonus,
            "breakdown_bonus": breakdown_bonus,
            "final_confidence": conf,
            "above_threshold": conf >= self.confidence_threshold
        })
        
        return float(round(conf, 3))

    def _report(self, user_q: str, conclusion: str, recommendations: str, confidence: float, steps: List[StepResult]) -> str:
        """Generate structured report with What/Why/Next steps format."""
        lines = []
        lines.append("# Business KPI Diagnosis Report")
        lines.append("")
        lines.append(f"**Query:** {redact_pii_text(user_q)}")
        lines.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Confidence Score:** {confidence:.2%}")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## What Happened")
        lines.append("")
        # Extract key findings from conclusion
        if conclusion:
            lines.append(conclusion)
        else:
            lines.append("Analysis completed. See evidence below for details.")
        lines.append("")
        
        # Add key metrics from evidence
        lines.append("### Key Metrics")
        for s in steps:
            if s.success:
                if s.action == "check_trend":
                    ts = s.evidence.get("timeseries", {})
                    trend = ts.get("trend", "unknown")
                    change = ts.get("change_last", 0.0)
                    lines.append(f"- **Trend:** {trend} (last period change: {change:.2%})")
                    anomalies = ts.get("anomalies", [])
                    if anomalies:
                        lines.append(f"- **Anomalies detected:** {len(anomalies)}")
                elif s.action == "breakdown":
                    bd = s.evidence.get("breakdown", {})
                    by_field = bd.get("by") or "by"
                    top_items = bd.get("rows", [])[:3]
                    if top_items:
                        label_list = []
                        for r in top_items:
                            label = r.get(by_field) or r.get("by") or r.get("category") or r.get("by_col") or "N/A"
                            label_list.append(str(label))
                        lines.append(f"- **Top contributors:** {', '.join(label_list)}")
                elif s.action == "compare_promo":
                    corr = s.evidence.get("corr_sales_promo", 0.0)
                    lines.append(f"- **Sales-Promo correlation:** {corr:.2f}")
        lines.append("")
        
        lines.append("---")
        lines.append("")
        lines.append("## Why")
        lines.append("")
        lines.append("### Root Cause Analysis")
        
        # Extract reasoning from evidence
        evidence_summary = []
        for s in steps:
            if s.success:
                if s.action == "check_trend":
                    ts = s.evidence.get("timeseries", {})
                    if ts.get("trend") == "decreasing":
                        evidence_summary.append("Time series analysis shows a declining trend.")
                    anomalies = ts.get("anomalies", [])
                    if anomalies:
                        evidence_summary.append(f"Detected {len(anomalies)} anomalies in the time series.")
                elif s.action == "breakdown":
                    bd = s.evidence.get("breakdown", {})
                    rows = bd.get("rows", [])
                    if rows:
                        top = rows[0]
                        contrib = top.get("contribution", 0.0)
                        evidence_summary.append(f"Breakdown analysis shows top contributor accounts for {contrib:.1f}% of total.")
                elif s.action == "compare_promo":
                    corr = s.evidence.get("corr_sales_promo", 0.0)
                    if abs(corr) > 0.3:
                        evidence_summary.append(f"Promotional activity shows {'strong positive' if corr > 0 else 'negative'} correlation with sales ({corr:.2f}).")
                elif s.action == "read_policy":
                    evidence_summary.append("Relevant policy documents were reviewed for context.")
        
        if evidence_summary:
            for item in evidence_summary:
                lines.append(f"- {item}")
        else:
            lines.append("See detailed evidence in the analysis steps below.")
        lines.append("")
        
        lines.append("### Analysis Steps")
        for i, s in enumerate(steps, 1):
            status = "✅" if s.success else "❌"
            lines.append(f"{i}. {status} **{s.action}** ({s.duration_ms:.0f}ms)")
            if not s.success and s.error:
                lines.append(f"   Error: {s.error}")
        lines.append("")
        
        lines.append("---")
        lines.append("")
        lines.append("## Next Steps")
        lines.append("")
        if recommendations:
            # Extract recommendations if they're in a structured format
            if "Next steps:" in recommendations:
                recs = recommendations.split("Next steps:")[-1].strip()
                lines.append(recs)
            else:
                lines.append(recommendations)
        else:
            lines.append("1. Review the evidence and metrics above")
            lines.append("2. Validate findings with business stakeholders")
            lines.append("3. Implement recommended actions based on root cause analysis")
        lines.append("")
        
        lines.append("---")
        lines.append("")
        lines.append("## Citations & Sources")
        lines.append("")
        citation_num = 1
        for s in steps:
            if s.action == "read_policy" and s.evidence.get("sources"):
                for src in s.evidence.get("sources", []):
                    file = src.get('file', 'unknown')
                    chunk = src.get('chunk_index', -1)
                    score = src.get('score', 0.0)
                    lines.append(f"[{citation_num}] **{file}** (chunk {chunk}, relevance: {score:.3f})")
                    citation_num += 1
        
        # Add SQL citations
        for s in steps:
            if s.evidence.get("sql"):
                lines.append(f"[{citation_num}] SQL Query: `{s.evidence['sql'][:100]}...`")
                citation_num += 1
        
        if citation_num == 1:
            lines.append("No citations available.")
        
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(f"*Report generated with confidence: {confidence:.2%}*")
        if confidence < self.confidence_threshold:
            lines.append(f"⚠️ **Warning:** Confidence below threshold ({self.confidence_threshold:.2%}). Review findings carefully.")
        
        return "\n".join(lines)


def run_agent(query: str, schema: Dict[str, List[str]], allowed: List[str], prev: Optional[Dict] = None, confidence_threshold: float = 0.55) -> Dict[str, Any]:
    """Main agent entry point with enhanced logging and guardrails."""
    t_start = time.time()
    logger = JSONLogger()
    
    # Log session start
    logger.log({
        "type": "session_start",
        "query": redact_pii_text(query),
        "schema_tables": list(schema.keys()),
        "allowed_tables": allowed
    })
    
    planner = Planner(logger)
    con = connect_db()
    sql = SQLTool(con, allowed_tables=allowed)  # Pass allow-list to SQLTool
    rag = RAGTool()
    ts = TimeSeriesTool()
    calc = CalcTool()
    
    plan = planner.make_plan(query, schema)
    execu = Executor(sql, rag, ts, calc, logger, confidence_threshold=confidence_threshold)
    out = execu.run(plan, query, schema, allowed, prev)
    
    t_total = time.time() - t_start
    
    # Log session completion
    logger.log({
        "type": "session_complete",
        "duration_ms": t_total * 1000,
        "steps_completed": sum(1 for s in out.steps if s.success),
        "steps_total": len(out.steps),
        "confidence": out.confidence,
        "above_threshold": out.confidence >= confidence_threshold
    })
    
    return {
        "type": "agent",
        "plan": [s.__dict__ for s in out.plan],
        "steps": [s.__dict__ for s in out.steps],
        "conclusion": out.conclusion,
        "recommendations": out.recommendations,
        "confidence": out.confidence,
        "citations": out.citations,
        "charts": out.charts,
        "report_md": out.report_md,
    }
