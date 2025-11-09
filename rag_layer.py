# rag_layer.py
import os
import json
from pathlib import Path
from typing import List, Dict, Any
import warnings
import re

# Suppress warnings
warnings.filterwarnings("ignore")

try:
    import numpy as np
except ImportError:
    raise ImportError("numpy is required. Install with: pip install numpy")

try:
    from pypdf import PdfReader
except ImportError:
    raise ImportError("pypdf is required. Install with: pip install pypdf")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")

try:
    import faiss
except ImportError:
    raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu")

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
except ImportError:
    raise ImportError("nltk is required. Install with: pip install nltk")

DOCS_DIR = Path("./docs")
INDEX_DIR = Path("./faiss_index")
INDEX_FILE = INDEX_DIR / "index.faiss"
METADATA_FILE = INDEX_DIR / "metadata.json"
EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
MIN_CHUNK_SIZE = 50

_model = None
_index = None
_metadata = None
_dim = None

def _ensure_dirs():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- PDF -> Text ----------
def pdf_to_text(pdf_path: str) -> str:
    try:
        reader = PdfReader(pdf_path)
        pages = []
        for p in reader.pages:
            try:
                t = p.extract_text() or ""
                if t.strip():
                    pages.append(t)
            except Exception:
                continue
        return "\n".join(pages)
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return ""

# ---------- Chunking ----------
def split_into_paragraphs(text: str):
    paras = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paras if p.strip()]

def chunk_text(text: str, mode: str = "sentence") -> List[str]:
    if not text:
        return []
    
    # Clean text first
    text = re.sub(r'\s+', ' ', text.strip())
    
    if mode == "sentence":
        sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
        # Group sentences into optimal chunks
        chunks = []
        current_chunk = ""
        for sent in sentences:
            if len(current_chunk) + len(sent) + 1 <= CHUNK_SIZE:
                current_chunk += (" " if current_chunk else "") + sent
            else:
                if current_chunk and len(current_chunk) >= MIN_CHUNK_SIZE:
                    chunks.append(current_chunk.strip())
                current_chunk = sent
        if current_chunk and len(current_chunk) >= MIN_CHUNK_SIZE:
            chunks.append(current_chunk.strip())
        return chunks
    if mode == "paragraph":
        paras = split_into_paragraphs(text)
        chunks = []
        for p in paras:
            if len(p) <= CHUNK_SIZE:
                chunks.append(p)
            else:
                # fallback: break long paragraphs
                start = 0
                while start < len(p):
                    end = min(start + CHUNK_SIZE, len(p))
                    chunks.append(p[start:end].strip())
                    start = max(end - CHUNK_OVERLAP, end)  # no infinite loop
        return [c for c in chunks if c]
    # legacy fallback: single big chunk
    return [text]

# ---------- Embedding model ----------
def get_embedding_model():
    global _model, _dim
    if _model is None:
        try:
            _model = SentenceTransformer(EMBED_MODEL_NAME)
            test = _model.encode("hello", convert_to_numpy=True)
            _dim = int(test.shape[-1])
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model: {e}")
    return _model

# ---------- FAISS ----------
def init_faiss_index(dim: int):
    # Inner product + normalized vectors == cosine similarity
    return faiss.IndexFlatIP(dim)

def save_index(index, path: Path):
    faiss.write_index(index, str(path))

def load_index(path: Path):
    return faiss.read_index(str(path))

def persist_metadata(metadata: List[Dict[str, Any]], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def load_metadata(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------- Load/Init ----------
def _ensure_index_and_meta():
    global _index, _metadata, _dim
    _ensure_dirs()
    get_embedding_model()
    if _index is None:
        if INDEX_FILE.exists() and METADATA_FILE.exists():
            _index = load_index(INDEX_FILE)
            _metadata = load_metadata(METADATA_FILE)
        else:
            _index = init_faiss_index(_dim)
            _metadata = []
    return _index, _metadata

# ---------- Ingest ----------
def ingest_pdfs_from_docs_dir(rebuild: bool = False, mode: str = "sentence") -> Dict[str, Any]:
    _ensure_dirs()
    model = get_embedding_model()
    index, metadata = _ensure_index_and_meta()

    files = sorted([p for p in DOCS_DIR.glob("*.pdf")])
    if not files:
        return {"status": "no_files", "added": 0, "details": []}

    if rebuild:
        index = init_faiss_index(_dim)
        metadata = []

    next_id = len(metadata)
    added = 0
    details = []

    for path in files:
        text = pdf_to_text(str(path))
        chunks = chunk_text(text, mode=mode)
        if not chunks:
            details.append({"file": path.name, "chunks": 0})
            continue

        # Embed in batches
        embs = []
        for i in range(0, len(chunks), 64):
            batch = chunks[i:i+64]
            vecs = model.encode(batch, convert_to_numpy=True)
            embs.append(vecs)
        emb_array = np.vstack(embs).astype("float32")

        # Normalize for cosine
        faiss.normalize_L2(emb_array)
        index.add(emb_array)

        # Record metadata
        for i, chunk in enumerate(chunks):
            metadata.append({
                "id": next_id,
                "source": path.name,
                "chunk_index": i,
                "text": chunk
            })
            next_id += 1

        added += len(chunks)
        details.append({"file": path.name, "chunks": len(chunks)})

    # Persist
    save_index(index, INDEX_FILE)
    persist_metadata(metadata, METADATA_FILE)

    # Update globals
    global _index, _metadata
    _index, _metadata = index, metadata

    return {"status": "ok", "added": added, "details": details}

# ---------- Retrieval ----------
def has_indexed_docs() -> bool:
    _ensure_dirs()
    return INDEX_FILE.exists() and METADATA_FILE.exists()

def retrieve(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    if not has_indexed_docs():
        return []

    model = get_embedding_model()
    index, metadata = _ensure_index_and_meta()

    # Enhance query with context
    enhanced_query = f"{query} retail business analytics"
    
    q_emb = model.encode([enhanced_query], convert_to_numpy=True)  # 2D shape (1,d)
    if q_emb.ndim == 1:
        q_emb = q_emb.reshape(1, -1)
    q_emb = q_emb.astype("float32")
    faiss.normalize_L2(q_emb)

    # Retrieve more candidates for re-ranking
    candidates = min(top_k * 3, len(metadata))
    D, I = index.search(q_emb, candidates)
    
    results = []
    for score, idx in zip(D[0], I[0]):
        if 0 <= int(idx) < len(metadata):
            entry = metadata[int(idx)]
            # Apply relevance scoring based on text length and content
            text_length = len(entry.get("text", ""))
            length_penalty = 1.0 if text_length > MIN_CHUNK_SIZE else 0.5
            adjusted_score = float(score) * length_penalty
            
            results.append({
                "id": entry["id"],
                "document": entry["text"],
                "metadata": {"source": entry["source"], "chunk_index": entry["chunk_index"]},
                "score": adjusted_score,
            })
    
    # Sort by adjusted score and return top_k
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

def build_rag_context(retrieved: List[Dict[str, Any]], max_chars: int = 2000) -> str:
    parts = []
    for r in retrieved:
        src = r.get("metadata", {}).get("source", "unknown")
        idx = r.get("metadata", {}).get("chunk_index", -1)
        doc = r.get("document", "").strip()
        parts.append(f"[{src} #chunk {idx}] {doc}\n---\n")
    text = "".join(parts)
    return text if len(text) <= max_chars else text[:max_chars] + "\n[TRUNCATED]"
