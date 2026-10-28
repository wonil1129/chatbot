# 학사규정 RAG 챗봇

대학 학사규정 PDF를 근거로 답하는 로컬 LLM 기반 RAG 챗봇 (Streamlit + Ollama + Chroma)

이 프로젝트를 다른 사람에게 전달할 때는 소스 코드만 보내는 것이 아니라, 실행에 필요한 의존성과 생성 파일을 구분해서 보내는 것이 좋습니다.

## 보내도 되는 것

- `app.py`
- `ingest.py`
- `src/`
- `.streamlit/config.toml`
- `requirements.txt`

## 보통 빼는 것

- `.venv/`
- `__pycache__/`
- `*.pyc`
- `.claude/`
- `chroma_db/`, `chroma_db_*/` 같은 생성된 벡터 DB

## 준비물 (저장소에 포함되지 않음)

- `rule.pdf`: 학사규정 PDF. 학교 홈페이지에서 받아 프로젝트 루트에 넣는다.
- `assets/logo.png`: 상단 로고 (선택). 없으면 로고 없이 실행된다. 파일명은 환경변수 `APP_LOGO`로 바꿀 수 있다.

## 실행 방법

1. Python 3.11 환경을 만든다.
2. `pip install -r requirements.txt` 를 실행한다.
3. 벡터 DB가 없으면 `python ingest.py` 를 먼저 실행한다.
4. `streamlit run app.py` 로 실행한다.

## 참고

현재 `src/config.py` 는 `chroma_db_koe5` 폴더가 있으면 그걸 우선 사용합니다. 즉, 이미 생성된 벡터 DB까지 같이 보내면 바로 실행할 수 있고, 보내지 않으면 상대방이 `ingest.py` 를 먼저 돌려야 합니다.