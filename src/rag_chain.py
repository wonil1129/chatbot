import re
import json
import os
from functools import lru_cache
from typing import Any

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from src.config import CHROMA_DIR, DEFAULT_OLLAMA_MODEL, EMBEDDING_MODEL, RERANKER_MODEL, BASE_DIR
from src.prompts import build_qa_prompt

FORMS_PATH = os.path.join(BASE_DIR, "src/forms.json")

@lru_cache(maxsize=1)
def _load_forms() -> dict:
    if os.path.exists(FORMS_PATH):
        with open(FORMS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def create_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "mps"},
    )

def create_vectorstore(embeddings: HuggingFaceEmbeddings) -> Chroma:
    persist_path = str(CHROMA_DIR)

    # chromadb versions differ in PersistentClient signature.
    # Try tenant/database-aware init first, then fall back.
    try:
        client = chromadb.PersistentClient(
            path=persist_path,
            tenant="default_tenant",
            database="default_database",
        )
    except TypeError:
        client = chromadb.PersistentClient(path=persist_path)

    return Chroma(
        client=client,
        persist_directory=persist_path,
        embedding_function=embeddings,
    )

@lru_cache(maxsize=1)
def _get_reranker() -> Any:
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(RERANKER_MODEL, device="mps")
    except Exception as exc:
        raise RuntimeError(f"BGE 리랭커 로드 실패: {exc}")

def _bge_rerank_scores(query: str, doc_texts: list[str]) -> list[float]:
    reranker = _get_reranker()
    pairs = [[query, text] for text in doc_texts]
    return [float(score) for score in reranker.predict(pairs)]

def create_llm() -> object:
    try:
        return ChatOllama(model=DEFAULT_OLLAMA_MODEL, temperature=0)
    except Exception as exc:
        raise RuntimeError(f"Ollama 연결 실패: {exc}")

def _extract_keywords(text: str) -> list[str]:
    return [tok for tok in re.findall(r"[가-힣A-Za-z0-9]+", text) if len(tok) >= 2]

def _score_doc(query: str, doc_text: str, metadata: dict = None) -> int:
    score = 0
    years = re.findall(r"(19\d{2}|20\d{2})", query)
    keywords = _extract_keywords(query)
    
    # 연도 매칭 (매우 중요)
    for year in years:
        if year in doc_text: score += 10
        if metadata and metadata.get("year") == year: score += 15

    # 핵심 키워드 매칭
    boosted = ["학점", "졸업", "휴학", "전공", "등록", "수강", "장학", "편입"]
    for kw in keywords:
        if kw in doc_text: score += 2
        if kw in boosted and kw in doc_text: score += 3
    return score

def check_forms(query: str) -> str:
    forms = _load_forms()
    query_clean = query.replace(" ", "").lower()
    best_match = None
    
    for form_name, page in forms.items():
        if len(form_name) < 4 or "<" in form_name or "개정" in form_name: continue
        name_clean = form_name.replace(" ", "").lower()
        if name_clean == query_clean or query_clean in name_clean:
            return f"요청하신 양식 '**{form_name}**'은(는) 학사규정 문서 **{page}페이지**에 있습니다."
            
        # 키워드 교집합 매칭
        if len(set(_extract_keywords(query)) & set(_extract_keywords(form_name))) >= 2:
            best_match = (form_name, page)

    return f"가장 유사한 양식 '**{best_match[0]}**'은(는) **{best_match[1]}페이지**에 있습니다." if best_match else ""

def retrieve_ranked_context(query: str, vectorstore: Chroma, top_k: int = 5) -> str:
    candidates = vectorstore.similarity_search(f"query: {query}", k=15)
    doc_texts = [doc.page_content for doc in candidates]
    
    try:
        bge_scores = _bge_rerank_scores(query, doc_texts)
        h_scores = [_score_doc(query, text, doc.metadata) for doc, text in zip(candidates, doc_texts)]
        scored = sorted(zip(candidates, bge_scores, h_scores), key=lambda x: (x[1] * 10 + x[2]), reverse=True)
        selected = [item[0] for item in scored[:top_k]]
    except:
        selected = sorted(candidates, key=lambda d: _score_doc(query, d.page_content, d.metadata), reverse=True)[:top_k]

    return "\n\n=====\n\n".join([f"[정보 | p.{d.metadata.get('page','?')}]\n{d.page_content.strip()}" for d in selected])

def answer_question(query: str, llm: object, vectorstore: Chroma):
    form_msg = check_forms(query)
    if form_msg:
        yield form_msg
        return
    context = retrieve_ranked_context(query, vectorstore)
    for chunk in llm.stream(build_qa_prompt().format(context=context, question=query)):
        yield getattr(chunk, "content", str(chunk))
