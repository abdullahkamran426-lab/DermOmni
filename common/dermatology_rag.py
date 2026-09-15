"""
Dermatology Vector Store (RAG) module for DermOmni.
Provides semantic retrieval of evidence-based clinical guidelines and literature.
Supports ChromaDB with an automatic zero-dependency fallback engine.
"""

import os
import re
import json
import math
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

KB_FILE_PATH = os.path.join("data", "dermatology_kb", "clinical_guidelines.json")


def _tokenize(text: str) -> List[str]:
    """Tokenize and normalize text into lowercase terms."""
    return re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())


class FallbackVectorEngine:
    """
    Lightweight TF-IDF / Cosine Similarity vector engine for zero-dependency RAG.
    """
    def __init__(self, documents: List[Dict[str, Any]]):
        self.documents = documents
        self.vocabulary: Dict[str, int] = {}
        self.doc_vectors: List[Dict[str, float]] = []
        self._build_index()

    def _build_index(self):
        doc_count = len(self.documents)
        df: Dict[str, int] = {}

        # First pass: calculate document frequencies
        doc_tokens_list = []
        for doc in self.documents:
            full_text = f"{doc.get('title', '')} {doc.get('category', '')} {doc.get('content', '')}"
            tokens = _tokenize(full_text)
            unique_tokens = set(tokens)
            for t in unique_tokens:
                df[t] = df.get(t, 0) + 1
            doc_tokens_list.append(tokens)

        # Build vocabulary & IDF
        idf: Dict[str, float] = {}
        for token, count in df.items():
            idf[token] = math.log((doc_count + 1) / (count + 1)) + 1.0

        # Second pass: calculate normalized TF-IDF doc vectors
        for tokens in doc_tokens_list:
            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            
            vector: Dict[str, float] = {}
            norm_sq = 0.0
            for t, count in tf.items():
                tfidf_val = count * idf.get(t, 1.0)
                vector[t] = tfidf_val
                norm_sq += tfidf_val ** 2
            
            norm = math.sqrt(norm_sq) if norm_sq > 0 else 1.0
            for t in vector:
                vector[t] /= norm

            self.doc_vectors.append(vector)
        self.idf = idf

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return self.documents[:top_k]

        tf: Dict[str, int] = {}
        for t in query_tokens:
            tf[t] = tf.get(t, 0) + 1

        query_vec: Dict[str, float] = {}
        norm_sq = 0.0
        for t, count in tf.items():
            tfidf_val = count * self.idf.get(t, 1.0)
            query_vec[t] = tfidf_val
            norm_sq += tfidf_val ** 2

        norm = math.sqrt(norm_sq) if norm_sq > 0 else 1.0
        for t in query_vec:
            query_vec[t] /= norm

        # Calculate cosine similarity against all docs
        scores = []
        for idx, doc_vec in enumerate(self.doc_vectors):
            dot_product = 0.0
            for t, val in query_vec.items():
                if t in doc_vec:
                    dot_product += val * doc_vec[t]
            scores.append((dot_product, idx))

        scores.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, idx in scores[:top_k]:
            doc_copy = dict(self.documents[idx])
            doc_copy["relevance_score"] = round(float(score), 4)
            results.append(doc_copy)

        return results


class DermatologyVectorStore:
    """
    RAG vector store manager for clinical dermatology literature.
    Attempts to initialize ChromaDB if available, otherwise falls back to FallbackVectorEngine.
    """
    def __init__(self, kb_path: str = KB_FILE_PATH):
        self.kb_path = kb_path
        self.documents: List[Dict[str, Any]] = []
        self.chroma_collection = None
        self.fallback_engine: Optional[FallbackVectorEngine] = None
        self._load_and_initialize()

    def _load_and_initialize(self):
        if os.path.exists(self.kb_path):
            try:
                with open(self.kb_path, "r", encoding="utf-8") as f:
                    self.documents = json.load(f)
                logger.info(f"Loaded {len(self.documents)} dermatology clinical records from {self.kb_path}")
            except Exception as e:
                logger.error(f"Failed to load dermatology KB JSON: {e}")
                self.documents = []
        else:
            logger.warning(f"KB file not found at {self.kb_path}")

        # Attempt ChromaDB initialization
        try:
            import chromadb
            client = chromadb.Client()
            self.chroma_collection = client.create_collection("dermatology_guidelines")
            
            # Unpack document metadata and text explicitly for student readability
            ids = []
            documents_text = []
            metadatas = []

            for doc in self.documents:
                doc_id = doc.get("id", "")
                title = doc.get("title", "")
                content = doc.get("content", "")
                category = doc.get("category", "")

                ids.append(doc_id)
                documents_text.append(f"{title} - {content}")
                metadatas.append({"category": category})

            self.chroma_collection.add(ids=ids, documents=documents_text, metadatas=metadatas)
            logger.info("Successfully initialized ChromaDB vector store for dermatology RAG.")
        except Exception as e:
            logger.info(f"ChromaDB unavailable ({e}). Using native TF-IDF vector retrieval engine.")
            if self.documents:
                self.fallback_engine = FallbackVectorEngine(self.documents)

    def query(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Query the dermatology vector store for relevant clinical evidence.
        """
        if not self.documents:
            return []

        if self.chroma_collection:
            try:
                chroma_res = self.chroma_collection.query(query_texts=[query_text], n_results=top_k)
                retrieved = []
                if chroma_res and "ids" in chroma_res and chroma_res["ids"]:
                    matched_ids = chroma_res["ids"][0]
                    for doc_id in matched_ids:
                        for original_doc in self.documents:
                            if original_doc["id"] == doc_id:
                                retrieved.append(original_doc)
                                break
                    return retrieved
            except Exception as e:
                logger.warning(f"ChromaDB query failed: {e}. Falling back to native vector engine.")

        if self.fallback_engine:
            return self.fallback_engine.search(query_text, top_k=top_k)

        return self.documents[:top_k]


# Singleton instance
_rag_store_instance: Optional[DermatologyVectorStore] = None

def get_dermatology_rag() -> DermatologyVectorStore:
    global _rag_store_instance
    if _rag_store_instance is None:
        _rag_store_instance = DermatologyVectorStore()
    return _rag_store_instance
