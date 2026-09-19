"""
Main FastAPI Application for Medical Information RAG Assistant
Provides REST endpoints for Query Processing, Document Exploration, Viva Benchmarking, and Settings.
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_engine.chunker import DocumentChunker
from rag_engine.retriever import HybridRetriever
from rag_engine.safety_guardrails import MedicalSafetyGuardrails
from rag_engine.generator import GroundedGenerator
from rag_engine.evaluator import VivaEvaluator

BASE_DIR = Path(__file__).resolve().parent
KB_DIR = os.path.join(BASE_DIR, "knowledge_base")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(
    title="MedLife - Medical Information RAG Assistant",
    description="Educational RAG system with strict grounding, safety guardrails, and viva evaluation lab.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG Components
chunker = DocumentChunker(max_chunk_words=220, chunk_overlap_words=35)
kb_chunks = chunker.load_and_chunk_directory(KB_DIR)
retriever = HybridRetriever(chunks=kb_chunks, similarity_threshold=0.22, top_k=3)
guardrails = MedicalSafetyGuardrails()
generator = GroundedGenerator(default_provider="local")
evaluator = VivaEvaluator(retriever, guardrails, generator)

# Request / Response Schemas
class QueryRequest(BaseModel):
    query: str = Field(..., example="What are the common symptoms of Type 2 diabetes?")
    similarity_threshold: Optional[float] = None
    top_k: Optional[int] = None
    strict_rag: bool = True

class SettingsRequest(BaseModel):
    similarity_threshold: Optional[float] = None
    top_k: Optional[int] = None
    provider: Optional[str] = None
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None

class ComparisonRequest(BaseModel):
    query: str = Field(..., example="What are the target blood pressure levels according to clinical guidelines?")

# Endpoints
@app.get("/api/status")
def get_status():
    return {
        "status": "online",
        "total_documents": len(set(c.doc_id for c in kb_chunks)),
        "total_chunks": len(kb_chunks),
        "similarity_threshold": retriever.similarity_threshold,
        "top_k": retriever.top_k,
        "provider": generator.provider,
        "disclaimer": guardrails.MANDATORY_DISCLAIMER
    }

@app.post("/api/query")
def process_query(req: QueryRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    thresh = req.similarity_threshold if req.similarity_threshold is not None else retriever.similarity_threshold
    k = req.top_k if req.top_k is not None else retriever.top_k

    # 1. Safety Guardrail Evaluation
    safety = guardrails.analyze_query(req.query)

    # 2. Retrieval with Thresholding
    retrieval = retriever.retrieve(req.query, top_k=k, threshold=thresh)

    # 3. Grounded Answer Synthesis
    gen_result = generator.generate_answer(
        query=req.query,
        retrieval_result=retrieval,
        safety_analysis=safety,
        strict_rag=req.strict_rag
    )

    return {
        "query": req.query,
        "answer": gen_result["answer"],
        "sources": gen_result["sources"],
        "citations": gen_result.get("citations", []),
        "safety": {
            "category": safety["category"],
            "triggered_reason": safety["triggered_reason"],
            "badge": safety["badge"],
            "is_safe_for_rag": safety["is_safe_for_rag"]
        },
        "retrieval": {
            "is_within_knowledge_base": retrieval["is_within_knowledge_base"],
            "top_score": retrieval["top_score"],
            "threshold": retrieval["threshold"],
            "precision_status": retrieval["precision_status"],
            "retrieved_count": len(retrieval["chunks"])
        },
        "metrics": {
            "grounding_score": gen_result["grounding_score"],
            "hallucination_risk": gen_result["hallucination_risk"],
            "status": gen_result["status"],
            "provider": gen_result.get("provider", generator.provider)
        },
        "disclaimer": guardrails.MANDATORY_DISCLAIMER
    }

@app.get("/api/documents")
def get_documents():
    docs_summary = {}
    for chunk in kb_chunks:
        if chunk.doc_id not in docs_summary:
            docs_summary[chunk.doc_id] = {
                "doc_id": chunk.doc_id,
                "title": chunk.doc_title,
                "source": chunk.source,
                "category": chunk.category,
                "sections": set(),
                "total_words": 0,
                "chunks": []
            }
        docs_summary[chunk.doc_id]["sections"].add(chunk.section)
        docs_summary[chunk.doc_id]["total_words"] += chunk.word_count
        docs_summary[chunk.doc_id]["chunks"].append({
            "chunk_id": chunk.chunk_id,
            "section": chunk.section,
            "word_count": chunk.word_count,
            "preview": chunk.text[:180] + "..."
        })

    result = []
    for doc in docs_summary.values():
        doc["sections"] = list(doc["sections"])
        doc["chunk_count"] = len(doc["chunks"])
        result.append(doc)
    return {"documents": result, "total_chunks": len(kb_chunks)}

@app.get("/api/benchmark")
def get_benchmarks():
    return {"test_cases": evaluator.get_benchmark_test_cases()}

@app.post("/api/evaluate")
def run_evaluation():
    return evaluator.run_full_benchmark()

@app.post("/api/compare")
def compare_grounding(req: ComparisonRequest):
    """
    Demonstrates the difference between Strict RAG Grounded mode vs Unconstrained Extrapolation.
    Invaluable for viva presentations to prove why strict grounding is necessary.
    """
    safety = guardrails.analyze_query(req.query)
    retrieval = retriever.retrieve(req.query)
    grounded_res = generator.generate_answer(req.query, retrieval, safety, strict_rag=True)

    # Simulated Unconstrained Hallucinatory model (illustrates ungrounded LLM behavior)
    unconstrained_text = (
        f"Regarding your question '{req.query}':\n\n"
        "Recent experimental studies in 2025 by Dr. Jonathan Miller at the Zurich Global Institute "
        "suggest that blood pressure and metabolic markers can be immediately stabilized using a 3-phase "
        "herbal nano-infusion therapy combined with 800mg synthetic peptide XYZ-42. Most patients see a 100% cure "
        "within 14 days without modifying diet or exercising.\n\n"
        "*(Note: The above simulated claim contains fabricated studies, unverified drugs, and guaranteed cures — "
        "a classic hallucination risk when generative models are not strictly bound to verified knowledge bases!)*"
    )

    return {
        "query": req.query,
        "grounded_result": {
            "answer": grounded_res["answer"],
            "grounding_score": grounded_res["grounding_score"],
            "hallucination_risk": grounded_res["hallucination_risk"],
            "sources_count": len(grounded_res["sources"]),
            "status": "STRICT_GROUNDED_RAG"
        },
        "unconstrained_simulation": {
            "answer": unconstrained_text,
            "grounding_score": 12.0,
            "hallucination_risk": 88.0,
            "sources_count": 0,
            "status": "UNCONSTRAINED_HALLUCINATION"
        },
        "viva_takeaway": (
            "Without strict RAG thresholding and context-anchored generation, LLMs hallucinate non-existent medical "
            "studies, fabricated doctors, and dangerous treatments. Our Grounded RAG system guarantees that every "
            "statement directly cites authorized WHO/CDC/AHA guidelines or abstains."
        )
    }

@app.post("/api/settings")
def update_settings(req: SettingsRequest):
    if req.similarity_threshold is not None:
        retriever.similarity_threshold = req.similarity_threshold
    if req.top_k is not None:
        retriever.top_k = req.top_k
    if req.provider or req.gemini_api_key or req.openai_api_key:
        generator.set_api_keys(
            gemini_key=req.gemini_api_key or "",
            openai_key=req.openai_api_key or "",
            provider=req.provider or generator.provider
        )
    return {
        "status": "updated",
        "similarity_threshold": retriever.similarity_threshold,
        "top_k": retriever.top_k,
        "provider": generator.provider
    }

# Mount static frontend
if os.path.exists(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
