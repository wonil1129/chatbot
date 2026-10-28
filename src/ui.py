from pathlib import Path

import streamlit as st

from src.config import APP_HEADER, APP_ICON, APP_LOGO, APP_TITLE, apply_runtime_env
from src.rag_chain import answer_question, create_embeddings, create_llm, create_vectorstore


BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"
TOP_LOGO_PATH = ASSETS_DIR / APP_LOGO

# 라벨: 실제 전송 질문
QUICK_REPLIES = {
    "졸업학점 확인": "졸업학점이 몇 학점인가요?",
    "휴학 신청 방법": "휴학 신청 방법은 어떻게 되나요?",
    "수강신청 기간": "수강신청 기간은 언제인가요?"
}

CATEGORIES = {
    "📚 수강": [
        "수강신청 기간은 언제인가요?",
        "최대 수강 학점은 몇 학점인가요?",
        "수강 취소 방법은?",
    ],
    "🎓 졸업": [
        "졸업학점이 몇 학점인가요?",
        "졸업 요건이 무엇인가요?",
        "조기졸업 조건은?",
    ],
    "🏫 휴학/복학": [
        "휴학 신청 방법은?",
        "최대 휴학 기간은?",
        "복학 절차는?",
    ],
    "📋 서식 찾기": [
        "휴학원 서식은 몇 페이지인가요?",
        "졸업신청서는 어디있나요?",
    ],
}

QUICK_REPLY_CSS = """
<style>
div[data-testid="column"] .stButton > button {
    height: 60px;
    white-space: normal;
    word-wrap: break-word;
}
</style>
"""


def _render_center_logo(path: Path) -> None:
    if not path.exists():
        return
    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.image(str(path))


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 카테고리별 질문")
        for category, questions in CATEGORIES.items():
            with st.expander(category):
                for q in questions:
                    if st.button(q, key=f"cat_{q}"):
                        st.session_state.pending_query = q


def _stream_response(generator) -> tuple[str, str | None]:
    full_text = ""
    placeholder = st.empty()
    for chunk in generator:
        full_text += chunk
        placeholder.markdown(full_text + "▌")
    placeholder.empty()

    if "출처:" in full_text:
        parts = full_text.rsplit("출처:", 1)
        body = parts[0].strip()
        source = parts[1].strip()
    else:
        body = full_text
        source = None

    return body, source


def _render_answer(body: str, source: str | None) -> None:
    st.markdown(body)
    if source:
        with st.expander("📄 출처 보기"):
            st.markdown(f"출처: {source}")


@st.cache_resource
def init_bot():
    apply_runtime_env()
    embeddings = create_embeddings()
    vectorstore = create_vectorstore(embeddings)
    llm = create_llm()
    return llm, vectorstore


def render_page() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON)
    st.markdown(QUICK_REPLY_CSS, unsafe_allow_html=True)
    _render_center_logo(TOP_LOGO_PATH)
    st.markdown(f"<h1 style='text-align: center'>{APP_HEADER}</h1>", unsafe_allow_html=True)
    st.markdown("---")

    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None
    if "qa_store" not in st.session_state:
        st.session_state.qa_store = None

    _render_sidebar()
    llm, vectorstore = init_bot()

    st.markdown("##### 자주 묻는 질문")
    cols = st.columns(len(QUICK_REPLIES))
    for col, (label, query) in zip(cols, QUICK_REPLIES.items()):
        if col.button(label, use_container_width=True):
            st.session_state.pending_query = query

    st.markdown("")

    prompt = st.chat_input("무엇이 궁금하신가요?")
    if not prompt and st.session_state.pending_query:
        prompt = st.session_state.pending_query
        st.session_state.pending_query = None

    qa_area = st.empty()

    if prompt:
        st.session_state.qa_store = None
        with qa_area.container():
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                try:
                    body, source = _stream_response(answer_question(prompt, llm, vectorstore))
                    _render_answer(body, source)
                    st.session_state.qa_store = {"question": prompt, "body": body, "source": source}
                except Exception as exc:
                    st.markdown(f"답변 생성 중 오류가 발생했습니다: {exc}")
    elif st.session_state.qa_store:
        qa = st.session_state.qa_store
        with qa_area.container():
            with st.chat_message("user"):
                st.markdown(qa["question"])
            with st.chat_message("assistant"):
                _render_answer(qa["body"], qa["source"])
