# 🤖 RAG-Powered Business Insights Assistant with Multi-Modal Data

## 📘 Overview
This is a **sophisticated retail analytics assistant** that combines **SQL query generation** with **RAG (Retrieval-Augmented Generation)** for comprehensive business insights. The system intelligently routes queries between data analytics and document retrieval, providing a unified interface for both structured and unstructured business intelligence.

---

## ⚙️ Tech Stack
- **Frontend:** Streamlit  
- **Backend / Logic:** Python  
- **Database:** SQLite (salesDw.db)  
- **AI Model:** Groq Llama 3.3 70B (via `genai_layer.py`)  
- **RAG Components:** Sentence-Transformers, FAISS, NLTK  
- **Libraries:** pandas, numpy, sqlparse, pypdf, openpyxl  

---

## 🏗️ Architecture Overview
| Layer | File | Description |
|-------|------|--------------|
| **Frontend** | `app.py` | Streamlit-based UI with chat interface, source tracking, and export capabilities |
| **Orchestrator** | `orchestrator.py` | Smart query routing, intent detection, and pipeline coordination |
| **GenAI Layer** | `genai_layer.py` | LLM integration for SQL generation, validation, and insight summarization |
| **RAG Layer** | `rag_layer.py` | Document processing, embedding, indexing, and intelligent retrieval |
| **Database Layer** | `db_layer.py` | Database connection, schema introspection, and query execution |

---

## 🧩 Enhanced Workflow
1. **User Input:** Natural language question (SQL analytics, document search, or hybrid)
2. **Intent Detection:** Smart routing based on query content and context
3. **Processing Paths:**
   - **SQL Analytics:** NL → SQL → Execution → Insight Generation
   - **Document Search:** Query → Embedding → Retrieval → Contextual Answer
   - **Hybrid:** Combines both data analytics and document insights
4. **Validation & Safety:** Comprehensive checks for SQL safety and content grounding
5. **Display:** Results with source attribution, confidence scores, and export options

---

## 🧱 Database Schema (Star Schema)
| Table | Key Columns | Description |
|--------|--------------|-------------|
| **FactSales** | Sale_ID, Product_ID, Customer_ID, Date_ID, Quantity, Sales | Transaction-level facts |
| **DimProduct** | Product_ID, Product_Name, Category, Sub_Category | Product details |
| **DimCustomer** | Customer_ID, Customer_Name, Segment, Region | Customer information |
| **DimDate** | Date_ID, Order_Date, Year, Quarter, Month | Date dimensions |
| **FactReturns** | Return_ID, Order_ID, Return_Amount | Return data |
| **FactMarketing** | Campaign_ID, Campaign_Name, Spend | Marketing data |

---

## 🚀 Key Improvements Made

### 🔧 Critical Fixes
- ✅ Fixed database connection management (removed problematic global connection)
- ✅ Added proper Excel export handling with error checking
- ✅ Fixed missing context text in document pipeline output
- ✅ Enhanced error handling for all API calls and dependencies

### 🎯 Performance Enhancements
- **Improved RAG Accuracy:** 
  - Optimized chunking strategy (500 chars with 100 overlap)
  - Enhanced query embedding with business context
  - Implemented re-ranking with length-based scoring
- **Better SQL Generation:**
  - Enhanced schema prompting with date functions and aggregation rules
  - Comprehensive fallback queries for common business patterns
  - Improved table and column canonicalization
- **Smart Intent Detection:**
  - Scoring-based query routing with confidence thresholds
  - Enhanced business term recognition (40+ SQL terms, 30+ document terms)
  - Better hybrid query splitting with 8 different patterns

### 🛡️ Robustness & Safety
- Comprehensive dependency checking with clear error messages
- Graceful API failure handling with fallback responses
- Automatic NLTK data download on first run
- SQL injection prevention and query validation
- Warning suppression for cleaner user experience  

---
## 🚀 Setup Instructions

### 1️⃣ Prerequisites
- Python 3.10 or above  
- SQLite3 (installed by default with Python)  
- Groq API key (free at https://console.groq.com/)

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Set Up API Key
```bash
# Option 1: Environment variable
export GROQ_API_KEY="your-groq-api-key"

# Option 2: Streamlit secrets
# Create .streamlit/secrets.toml and add:
# GROQ_API_KEY = "your-groq-api-key"
```

### 4️⃣ Prepare Your Data
- Place your SQLite database as `salesDw.db` in the project root
- Add PDF documents to the `docs/` directory for RAG functionality

### 5️⃣ Run the App
```bash
streamlit run app.py
```

---

## 💬 Example Queries to Try

### SQL Analytics
- "Show total sales by category"
- "Which sub-category generated the highest sales?"  
- "Top 10 products by total sales amount"
- "Show total sales for each region"
- "Sales trends for the last 12 months"
- "Customer segmentation analysis"

### Document Search
- "What is our return policy?"
- "Explain our sustainability initiatives"
- "What are the compliance guidelines?"
- "Describe our strategic framework"

### Hybrid Queries
- "Show sales trends and explain our sustainability policy"
- "Top performing products and relevant business guidelines"

---

## 🎯 Advanced Features

### RAG Index Management
- **Sentence Chunking**: Better for precise information retrieval
- **Paragraph Chunking**: Better for contextual understanding
- **Rebuild Index**: Updates after adding new documents
- **Source Tracking**: Shows which documents provided answers

### Query Safety & Validation
- SQL injection prevention
- Schema grounding to prevent hallucination
- Comprehensive error handling
- Fallback queries for common patterns

### Export & Sharing
- Excel export for SQL results
- Source attribution for transparency
- Chat history for context retention

---

## 🔧 Configuration Options

### Environment Variables
- `GROQ_API_KEY`: Required for LLM functionality

### Chunking Strategies
- **Sentence Mode**: 500-char chunks with 100-char overlap
- **Paragraph Mode**: Semantic paragraph boundaries

### Performance Tuning
- Adjust `top_k` parameter for retrieval precision
- Modify chunk sizes based on document types
- Use date filters for better SQL performance

---

## 🛠️ Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   - All dependencies are checked on startup with helpful error messages

2. **API Key Issues**
   - Ensure GROQ_API_KEY is set correctly
   - Check API key validity and quota

3. **RAG Index Issues**
   - Ensure PDF files are in the `docs/` directory
   - Click "Rebuild Index" after adding new documents
   - Verify documents contain extractable text

4. **Database Connection**
   - Verify `salesDw.db` exists in project root
   - Check database file permissions
   - Ensure required tables are present

---

## 📈 Performance Tips

- Use sentence chunking for precise queries
- Use paragraph chunking for broader context needs
- Limit queries to specific time ranges for better performance
- Rebuild index after adding significant new documents
- Clear chat history periodically for better performance

---

## 🧠 Technical Deep Dive

### Intent Detection Algorithm
- Scoring-based system with 40+ SQL terms and 30+ document terms
- Confidence thresholds for reliable routing
- Hybrid query support with 8 splitting patterns

### RAG Enhancement Strategy
- Query enhancement with business context
- Re-ranking based on chunk length and relevance
- Cosine similarity with normalized embeddings

### SQL Safety Framework
- Multi-layer validation (syntax, schema, safety)
- Canonical table and column mapping
- Comprehensive fallback system

---

## 📄 License

This project is licensed under the MIT License.

---

## 👩‍💻 Developed For

RAG-Powered Business Insights Assistant Project
Advanced Multi-Modal Analytics with GenAI Integration

**Author:** Anushree Sathyan  
**Tools Used:** Streamlit, Groq Llama 3.3, FAISS, Sentence-Transformers, SQLite, Python