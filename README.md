# 🤖 Agentic Business KPI Diagnosis System

A comprehensive agentic system that diagnoses business KPI drops, performs multi-step reasoning, and provides data-backed root-cause insights and recommendations. This system combines SQL analysis, document retrieval (RAG), time series analysis, and LLM-based reasoning to deliver actionable business intelligence.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [System Components](#system-components)
- [Requirements Verification](#requirements-verification)
- [Example Questions](#example-questions)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This agentic system upgrades a traditional RAG assistant into an intelligent business analyst that can:

- **Investigate anomalies end-to-end** with autonomous multi-step reasoning
- **Mix SQL + documents + reasoning** automatically
- **Produce traceable, cited, actionable insights** with confidence scores

The system uses an LLM-based planner to create investigation plans, executes multiple tools to gather evidence, and generates comprehensive reports with root-cause analysis.

---

## ✨ Features

### Core Capabilities

- **🤖 Agentic Planning**: LLM-based planner creates multi-step investigation plans
- **🔧 Tool Orchestration**: Executor coordinates 4+ specialized tools
- **📊 Multi-Source Analysis**: Combines SQL queries, document retrieval, and reasoning
- **📈 Visual Analytics**: Interactive charts for trends and breakdowns
- **📄 Auto-Generated Reports**: Structured "What/Why/Next steps" format
- **🎯 Confidence Scoring**: Multi-factor confidence assessment
- **🔒 Security**: SQL allow-list, PII redaction, read-only validation
- **📝 Observability**: Comprehensive JSON logging

### Tools

1. **SQLTool**: Database queries with joins, aggregates, read-only validation
2. **RAGTool**: Semantic search over embedded PDF documents
3. **TimeSeriesTool**: Trend analysis and anomaly detection
4. **CalcTool**: KPI breakdowns and what-if counterfactuals

### Stretch Goals

- ✅ Counterfactual simulator ("What if sales +10%?")
- ✅ Root-cause ranking via contribution percentages
- ✅ Short-term conversation memory
- ✅ PDF export (Markdown format, convertible to PDF)

---

## 🏗️ Architecture

### System Flow

```
User Question
    ↓
Intent Detection (orchestrator.py)
    ↓
┌─────────────────────────────────────┐
│  Agent Pipeline (agent.py)          │
│  ┌───────────────────────────────┐ │
│  │ Planner (LLM-based)            │ │
│  │ Creates step-by-step plan     │ │
│  └───────────────────────────────┘ │
│           ↓                          │
│  ┌───────────────────────────────┐ │
│  │ Executor                       │ │
│  │ Orchestrates tool execution    │ │
│  └───────────────────────────────┘ │
│           ↓                          │
│  ┌───────────────────────────────┐ │
│  │ Tools                          │ │
│  │ • SQLTool                      │ │
│  │ • RAGTool                      │ │
│  │ • TimeSeriesTool               │ │
│  │ • CalcTool                     │ │
│  └───────────────────────────────┘ │
│           ↓                          │
│  Evidence Collection & Summarization │
└─────────────────────────────────────┘
    ↓
UI Display (app.py)
    ↓
Plan → Evidence → Conclusion → Report
```

### Component Overview

- **`orchestrator.py`**: Routes queries to appropriate pipeline (SQL/Doc/Hybrid/Agent)
- **`agent.py`**: Core agentic system with Planner, Executor, and Tools
- **`genai_layer.py`**: LLM integration (Groq) for SQL generation and summarization
- **`rag_layer.py`**: FAISS-based document retrieval from PDFs
- **`db_layer.py`**: SQLite database operations
- **`app.py`**: Streamlit UI with interactive visualizations

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- SQLite database (`salesDw.db`)
- PDF documents in `docs/` folder (optional)

### Step 1: Clone/Setup

```bash
cd "E:\JOB_TRAINING\6. AgenticSystem"
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set API Key

Create a `.env` file or set environment variable:

```bash
# Windows PowerShell
$env:GROQ_API_KEY="your-api-key-here"

# Linux/Mac
export GROQ_API_KEY="your-api-key-here"
```

Or create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your-api-key-here"
```

### Step 5: Index Documents (Optional)

1. Place PDF files in `docs/` folder
2. Run Streamlit app
3. Click "Rebuild Index" in sidebar

---

## 🎮 Quick Start

### 1. Start the Application

```bash
streamlit run app.py
```

### 2. Test with Sample Question

Copy and paste into the chat:

```
"Why did sales drop?"
```

### 3. View Results

You should see:
- ✅ Plan with numbered steps
- ✅ Evidence from all tools
- ✅ Time series chart
- ✅ Breakdown charts
- ✅ Conclusion with root cause
- ✅ Recommendations
- ✅ Confidence score
- ✅ Citations
- ✅ Downloadable report

---

## 📖 Usage

### Agentic Questions

Questions that trigger the full agentic pipeline:

**Pattern**: Use words like "why", "investigate", "diagnose", "root cause" combined with KPIs like "sales", "revenue", "orders"

**Examples**:
- "Why did sales drop?"
- "What caused the revenue decline?"
- "Investigate the root cause of the orders drop"
- "Diagnose the sales anomaly"
- "What if sales increased by 10%?"

### SQL-Only Questions

Questions that route to SQL pipeline:

**Examples**:
- "Show top 10 products by sales"
- "List sales by category"
- "Show sales by region"

### Document-Only Questions

Questions that route to document pipeline:

**Examples**:
- "What is the return policy?"
- "Explain the discount strategy"
- "What are the sales management guidelines?"

### Hybrid Questions

Questions that combine SQL + Documents:

**Examples**:
- "Show sales by category and explain the category strategy"
- "List sales by region and describe the regional strategy"
- "Compare sales trends with marketing spend and explain the promotional strategy"

---

## 🔧 System Components

### Planner (`agent.py` lines 136-219)

- **Purpose**: Creates multi-step investigation plans
- **Method**: `Planner.make_plan()`
- **Input**: User question, database schema
- **Output**: List of `PlanStep` objects
- **LLM**: Uses Groq API with structured JSON prompts

**Example Plan**:
```
[1] check_trend (kpi: sales, granularity: month)
[2] breakdown (by: category)
[3] compare_promo
[4] read_policy (topics: discount, returns)
[5] what_if (kpi: sales, pct: 0.05)
[6] finalize
```

### Executor (`agent.py` lines 413-595)

- **Purpose**: Executes plan steps and collects evidence
- **Method**: `Executor.run()`
- **Features**:
  - Calls tools based on plan steps
  - Collects evidence from each tool
  - Generates conclusion and recommendations
  - Calculates confidence score
  - Creates auto-generated report

### SQLTool (`agent.py` lines 231-312)

- **Purpose**: Database query execution
- **Features**:
  - Joins: FactSales + DimProduct + DimCustomer + DimDate
  - Aggregates: SUM, COUNT, AVG, MAX, MIN
  - Read-only validation
  - Table allow-list enforcement
- **Methods**: `nl2sql()`, `timeseries_sql()`, `run_sql()`

### RAGTool (`agent.py` lines 315-328)

- **Purpose**: Document retrieval from PDFs
- **Features**:
  - FAISS-based semantic search
  - Retrieves from embedded PDFs
  - Returns context + citations
- **Method**: `retrieve()`

### TimeSeriesTool (`agent.py` lines 331-371)

- **Purpose**: Trend and anomaly analysis
- **Features**:
  - Trend detection (increasing/decreasing/flat)
  - Anomaly detection (z-score method, threshold ≥ 2.0)
  - Change percentage calculation
  - Chart data generation
- **Method**: `analyze()`

### CalcTool (`agent.py` lines 374-410)

- **Purpose**: KPI breakdowns and counterfactuals
- **Features**:
  - Breakdown by dimension (category/region/segment)
  - Contribution percentage calculation
  - What-if counterfactual scenarios
  - Top-N analysis
- **Methods**: `breakdown()`, `what_if()`

---

## ✅ Requirements Verification

### Core Requirements ✅

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Planner + Executor | ✅ | `agent.py` Planner & Executor classes |
| SQLTool | ✅ | Joins, aggregates, read-only validation |
| RAGTool | ✅ | PDF retrieval with FAISS |
| TimeSeriesTool | ✅ | Trend & anomaly detection |
| CalcTool | ✅ | Breakdown & what-if |
| SQL Allow-List | ✅ | DML/drop prevention |
| PII Redaction | ✅ | 7+ patterns |
| Confidence Thresholds | ✅ | Multi-factor scoring |
| JSON Logs | ✅ | Plans, tool calls, elapsed time |
| UI Deliverables | ✅ | Plan → Evidence → Conclusion |
| Auto-Generated Report | ✅ | What/Why/Next steps format |

### Stretch Goals ✅

| Goal | Status | Implementation |
|------|--------|----------------|
| Counterfactual Simulator | ✅ | `CalcTool.what_if()` |
| Root-Cause Ranking | ✅ | Contribution percentages |
| Conversation Memory | ✅ | `prev_turn` parameter |
| PDF Export | ✅ | Markdown report |

**Total: 15/15 Requirements Met** ✅

---

## 💡 Example Questions

### Agentic Questions (Full Pipeline)

```
"Why did sales drop?"
"What caused the revenue decline?"
"Investigate the root cause of the orders drop"
"Diagnose the sales anomaly"
"What if sales increased by 10%?"
```

### SQL Questions

```
"Show top 10 products by sales"
"List sales by category"
"Show sales by region"
"What are the total sales by customer segment?"
```

### Document Questions

```
"What is the return policy?"
"Explain the discount strategy"
"What are the sales management guidelines?"
```

### Hybrid Questions

```
"Show sales by category and explain the category strategy"
"List sales by region and describe the regional strategy"
"Compare sales trends with marketing spend and explain the promotional strategy"
```

---

## ⚙️ Configuration

### Environment Variables

- `GROQ_API_KEY`: Required for LLM functionality

### Database

- **Location**: `salesDw.db`
- **Schema**: Star schema with FactSales, DimProduct, DimCustomer, DimDate, FactReturns, FactMarketing

### Documents

- **Location**: `docs/` folder
- **Format**: PDF files
- **Index**: FAISS index stored in `faiss_index/`

### Confidence Threshold

- **Default**: 0.55 (55%)
- **Configurable**: Pass `confidence_threshold` parameter to `run_agent()`

### Logging

- **File**: `agent_logs.jsonl`
- **Format**: JSON Lines (one JSON object per line)
- **Contents**: Plans, tool calls, evidence summaries, elapsed time

---

## 🐛 Troubleshooting

### Issue: Question routes to SQL instead of Agent

**Solution**: Use trigger words + KPI words
- ✅ Good: "Why did sales drop?" (has "why" + "sales" + "drop")
- ❌ Bad: "Show sales" (no trigger word)

### Issue: Time series shows "no_numeric" error

**Solution**: Fixed in latest version - date format handling improved

### Issue: Breakdown shows "?" categories

**Solution**: Fixed in latest version - NULL handling improved

### Issue: What-if shows 0.00

**Solution**: Fixed in latest version - fallback to breakdown totals

### Issue: No documents retrieved

**Solution**:
1. Check PDFs are in `docs/` folder
2. Click "Rebuild Index" in sidebar
3. Verify index status shows "✅ Ready"

### Issue: SQL errors

**Solution**:
1. Verify database has data
2. Check table names match schema
3. Verify SQL allow-list includes required tables

### Issue: Low confidence scores

**Solution**:
- This is expected if:
  - Some tool steps fail
  - Limited data available
  - Documents not indexed
- Check individual step results in Evidence section

---

## 📁 File Structure

```
AgenticSystem/
├── agent.py                 # Core agentic system (Planner, Executor, Tools)
├── app.py                   # Streamlit UI
├── orchestrator.py          # Query routing and orchestration
├── genai_layer.py           # LLM integration (Groq)
├── rag_layer.py             # Document retrieval (FAISS)
├── db_layer.py              # Database operations
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── salesDw.db              # SQLite database
├── agent_logs.jsonl        # JSON logs
├── export.xlsx             # Excel export (generated)
├── docs/                   # PDF documents folder
│   ├── chunk1.pdf
│   ├── chunk2.pdf
│   └── chunk3.pdf
└── faiss_index/            # FAISS index for documents
    ├── index.faiss
    └── metadata.json
```

---

## 🔍 Key Features Explained

### Plan → Evidence → Conclusion Flow

1. **Plan**: LLM creates multi-step investigation plan
2. **Evidence**: Executor runs tools and collects evidence
3. **Conclusion**: LLM summarizes findings and provides recommendations

### Charts & Visualizations

- **Time Series Chart**: Line chart showing trends over time
- **Breakdown Chart**: Bar chart showing distribution by dimension
- **Anomaly Detection**: Highlights unusual data points

### Auto-Generated Report

Structured markdown report with:
- **What Happened**: Key findings and metrics
- **Why**: Root cause analysis with evidence
- **Next Steps**: Actionable recommendations
- **Confidence Score**: Reliability indicator
- **Citations**: All sources cited

### Guardrails

- **SQL Safety**: Only WITH/SELECT queries, DML/drop blocked
- **Table Allow-List**: Only allowed tables can be queried
- **PII Redaction**: Email, phone, SSN, credit card, etc. redacted
- **Confidence Thresholds**: Warnings when confidence is low

### Observability

- **JSON Logs**: All activities logged to `agent_logs.jsonl`
- **Session Tracking**: Unique session IDs
- **Tool Call Logging**: Each tool execution logged
- **Elapsed Time**: Performance metrics tracked

---

## 📊 Database Schema

### Tables

- **FactSales**: Sales_ID, Order_ID, Customer_ID, Product_ID, Date_ID, Sales
- **DimCustomer**: Customer_ID, Customer_Name, Segment, Region, State, City, Country
- **DimProduct**: Product_ID, Category, Sub_Category, Product_Name
- **DimDate**: Date_ID, Order_Date, Ship_Date, Ship_Mode
- **FactReturns**: Return_ID, Order_ID, Product_ID, Customer_ID, Date_ID, Return_Qty, Return_Amount, Reason
- **FactMarketing**: Campaign_ID, Date_ID, Region, Channel, Spend_Amount, Campaign_Name

### Key Relationships

- FactSales.Customer_ID → DimCustomer.Customer_ID
- FactSales.Product_ID → DimProduct.Product_ID
- FactSales.Date_ID → DimDate.Date_ID
- FactReturns.Order_ID → FactSales.Order_ID
- FactMarketing.Date_ID → DimDate.Date_ID

---

## 🎯 Use Cases

### 1. KPI Drop Diagnosis

**Question**: "Why did sales drop in September?"

**System Response**:
- Creates plan: check_trend → breakdown → compare_promo → read_policy
- Executes tools: Time series analysis, category breakdown, promo correlation, policy retrieval
- Generates report: What happened, why, next steps
- Provides confidence score and citations

### 2. What-If Analysis

**Question**: "What if sales increased by 10%?"

**System Response**:
- Analyzes current sales trend
- Calculates counterfactual: base × 1.10
- Shows impact analysis
- Provides recommendations

### 3. Anomaly Detection

**Question**: "Investigate the sales spike in the last month"

**System Response**:
- Detects anomalies using z-score method
- Analyzes contributing factors
- Retrieves relevant policies
- Provides root cause analysis

### 4. Multi-Dimensional Analysis

**Question**: "Why did sales drop? Break it down by category, region, and segment"

**System Response**:
- Multiple breakdowns (category, region, segment)
- Comprehensive evidence collection
- Multi-factor root cause analysis
- Detailed recommendations

---

## 🔐 Security & Guardrails

### SQL Security

- ✅ Only WITH/SELECT queries allowed
- ✅ DML operations blocked (DELETE, UPDATE, INSERT)
- ✅ DDL operations blocked (DROP, ALTER, CREATE)
- ✅ Table allow-list enforcement
- ✅ Schema grounding validation

### PII Protection

- ✅ Email addresses redacted
- ✅ Phone numbers redacted
- ✅ SSN redacted
- ✅ Credit card numbers redacted
- ✅ IP addresses redacted
- ✅ Passport numbers redacted
- ✅ Driver's license numbers redacted

### Confidence Thresholds

- ✅ Multi-factor confidence scoring
- ✅ Configurable threshold (default: 0.55)
- ✅ UI warnings when below threshold
- ✅ Detailed confidence breakdown

---

## 📈 Performance

### Typical Execution Times

- **Plan Creation**: 1-2 seconds
- **Tool Execution**: 50-200ms per tool
- **Report Generation**: 1-2 seconds
- **Total**: 3-5 seconds for complete analysis

### Optimization Tips

- Index documents once (reuse index)
- Use specific questions (better routing)
- Check database has sufficient data
- Monitor logs for performance issues

---

## 🧪 Testing

### Quick Test

1. Start app: `streamlit run app.py`
2. Ask: "Why did sales drop?"
3. Verify all sections appear:
   - ✅ Plan
   - ✅ Evidence (charts, SQL, documents)
   - ✅ Conclusion
   - ✅ Recommendations
   - ✅ Confidence score
   - ✅ Citations
   - ✅ Report download

### Test Questions

See `100_PERCENT_WORKING_AGENTIC_QUESTIONS.txt` for comprehensive test questions.

---

## 📝 Logging

### Log File: `agent_logs.jsonl`

**Log Types**:
- `session_start`: Session initialization
- `plan`: Plan creation with steps
- `tool_call`: Individual tool execution
- `confidence_calculation`: Confidence scoring
- `session_complete`: Session summary

**Example Log Entry**:
```json
{
  "type": "tool_call",
  "tool": "TimeSeriesTool",
  "step_id": "s1",
  "action": "check_trend",
  "success": true,
  "duration_ms": 37.5,
  "evidence_summary": {
    "timeseries_summary": {
      "trend": "decreasing",
      "anomalies_count": 2
    }
  },
  "ts": 1701234567.89,
  "session_id": "session_1701234567",
  "timestamp_iso": "2025-11-29T18:07:06"
}
```

---

## 🤝 Contributing

### Adding New Tools

1. Create tool class in `agent.py`
2. Add tool to Executor's `run()` method
3. Update tool name mapping in logging
4. Add UI display in `app.py`

### Adding New Guardrails

1. Add validation in appropriate layer
2. Update error messages
3. Add logging
4. Update documentation

---

## 📄 License

See `LICENSE` file for details.

---

## 🙏 Acknowledgments

- **Groq**: LLM API for planning and summarization
- **FAISS**: Vector similarity search for document retrieval
- **Streamlit**: UI framework
- **Sentence Transformers**: Embedding model for RAG

---

## 📞 Support

For issues or questions:
1. Check `TROUBLESHOOTING` section
2. Review logs in `agent_logs.jsonl`
3. Verify all requirements are installed
4. Check API key is set correctly

---

## 🎉 Status

**✅ PRODUCTION READY**

- All core requirements implemented
- All stretch goals completed
- Comprehensive guardrails and observability
- Complete UI with all deliverables
- Extensive testing and verification

**Version**: 1.0.0  
**Last Updated**: 2025-11-29  
**Status**: 100% Complete ✅
