import os
import shutil
import pdfplumber
import json
import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- Configuration ---
PDF_PATH = "rule.pdf"
CHROMA_DIR = "./chroma_db_koe5"
FORMS_JSON = "src/forms.json"
CUTOFF_PAGE = 79  # Based on investigation, actual forms start around page 79-80

def clean_text(text: str) -> str:
    return " ".join((text or "").replace("\u00a0", " ").split())

def table_to_row_docs(table: list, context_prefix: str, metadata: dict) -> list:
    """표의 각 행을 독립 Document로 변환 (병합 셀 forward-fill 적용)"""
    if not table: return []
    cleaned_table = [[clean_text(cell) if cell else "" for cell in row] for row in table]
    cleaned_table = [row for row in cleaned_table if any(row)]
    if not cleaned_table: return []

    max_cols = max(len(row) for row in cleaned_table)
    normalized = [row + [""] * (max_cols - len(row)) for row in cleaned_table]

    # 병합 셀 처리: 빈 셀을 위 행 값으로 채움
    prev_row = [""] * max_cols
    filled = []
    for row in normalized:
        filled_row = [cell if cell else prev_row[i] for i, cell in enumerate(row)]
        filled.append(filled_row)
        prev_row = filled_row

    header = filled[0]
    row_docs = []
    for row in filled[1:]:
        if not any(row): continue
        # 헤더-값 쌍으로 구성된 자연어 형식으로 저장
        row_text = " | ".join(
            f"{h}: {v}" for h, v in zip(header, row) if v
        )
        if row_text:
            row_docs.append(Document(
                page_content=context_prefix + "[표 행 데이터]\n" + row_text,
                metadata={**metadata, "type": "table_row"},
            ))
    return row_docs

def extract_year_from_text(text: str) -> str:
    match = re.search(r"(\d{4})학년도", text)
    return match.group(1) if match else ""

print("🚀 Starting Ingestion Process...")
docs = []
forms = {}

with pdfplumber.open(PDF_PATH) as pdf:
    for i in range(min(CUTOFF_PAGE, len(pdf.pages))):
        page = pdf.pages[i]
        page_num = i + 1
        text = page.extract_text()
        entrance_year = extract_year_from_text(text or "")
        context_prefix = f"passage: [학사규정 {page_num}페이지"
        if entrance_year: context_prefix += f" | {entrance_year}학년도 입학생 관련"
        context_prefix += "]\n"

        if text:
            docs.append(Document(
                page_content=context_prefix + clean_text(text),
                metadata={"page": page_num, "year": entrance_year, "type": "text"}
            ))

        for j, table in enumerate(page.extract_tables()):
            row_docs = table_to_row_docs(
                table, context_prefix,
                {"page": page_num, "year": entrance_year}
            )
            docs.extend(row_docs)

    for i in range(CUTOFF_PAGE, len(pdf.pages)):
        text = pdf.pages[i].extract_text()
        if text:
            match = re.search(r"<(?:서식|별지)\s*(?:제?\d+호)?\s*서식>\s*\n?([^\n]+)", text)
            if match: forms[match.group(1).strip()] = i + 1

with open(FORMS_JSON, "w", encoding="utf-8") as f:
    json.dump(forms, f, ensure_ascii=False, indent=2)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

text_docs = [d for d in docs if d.metadata.get("type") == "text"]
table_row_docs = [d for d in docs if d.metadata.get("type") == "table_row"]

text_splits = text_splitter.split_documents(text_docs)

# 분할된 청크에 passage: prefix 누락 시 재추가
for doc in text_splits:
    if not doc.page_content.startswith("passage:"):
        page = doc.metadata.get("page", "?")
        year = doc.metadata.get("year", "")
        prefix = f"passage: [학사규정 {page}페이지"
        if year:
            prefix += f" | {year}학년도 입학생 관련"
        prefix += "]\n"
        doc.page_content = prefix + doc.page_content

# 표 행은 이미 행 단위로 분리되어 있으므로 추가 분할 없이 합산
final_splits = text_splits + table_row_docs

if os.path.exists(CHROMA_DIR): shutil.rmtree(CHROMA_DIR)

print("🧠 Embedding with KoE5...")
embeddings = HuggingFaceEmbeddings(model_name="nlpai-lab/KoE5", model_kwargs={"device": "mps"})
vectorstore = Chroma.from_documents(documents=final_splits, embedding=embeddings, persist_directory=CHROMA_DIR)
print("✨ Ingestion Complete!")
