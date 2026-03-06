# Langchain 활용 RAG Insights Tool

이 프로젝트는 Langchain 라이브러리를 활용하여 PDF 문서를 분석하고 핵심 인사이트를 도출하는 RAG 기반의 AI 시스템입니다. Google Gemini API와 LangChain, FAISS를 활용하여 제작되었습니다. (아직 주요 에러 수정 중입니다.)

## 🚀 주요 기능

- **PDF 추출 및 정제**: 업로드된 PDF에서 텍스트를 추출하고 불필요한 특수문자, 공백 등을 정제합니다.
- **지능형 문서 청킹**: 문맥 손실을 최소화하기 위해 Semantic 단위와 일정 길이의 Overlap을 두고 문서를 청크로 분할합니다.
- **벡터 인덱싱 (FAISS)**: Google의 `models/gemini-embedding-001` 모델을 사용하여 문서 조각들을 벡터화하고 로컬에 상태를 유지하며 인덱싱합니다. Google API의 무료 티어 제한(Rate Limit)을 고려하여 60초 대기 로직이 적용되어 있습니다.
- **Insight 및 Q&A 생성**: 문서에서 중요한 비즈니스 인사이트 상위 3가지를 자동 추출하며, Streamlit을 통한 직관적인 질의응답 인터페이스를 제공합니다.

## ⚙️ 요구 사항 및 설치 (Prerequisites)

1. **Python 3.10+** (제공된 가상환경 `venv` 사용 권장)
2. **필수 라이브러리 설치**:
   ```bash
   pip install streamlit langchain langchain-core langchain-classic langchain-google-genai langchain-community faiss-cpu pypdf
   ```
3. **Google API Key**: 시스템을 이용하려면 `Gemini API Key`가 필요합니다. (실행 시 사이드바에서 입력)

## 💻 실행 방법 (How to Run)

아래 명령어를 통해 Streamlit 애플리케이션을 실행합니다.

```bash
streamlit run app.py
```

브라우저 창이 열리면 사이드바에 **Gemini API Key**를 입력한 후 PDF 파일을 업로드하면 분석이 진행됩니다.

## 📁 주요 코드 구조 설명

- **`engine.py`**:
  - **`DocumentProcessor`**: 원본 리소스에서 텍스트를 가져오고 자연어 처리 시스템이 이해하기 쉽도록 전처리(`preprocess_text`) 및 분할(`split_documents`)하는 파이프라인입니다.
  - **`RAGEngine`**: FAISS를 사용해 Document List를 인덱싱합니다. Rate Limit을 회피하기 위해 `merge_from`으로 배치를 쪼개어 API 요청을 전송합니다.
- **`app.py`**: Streamlit을 활용하여 업로드/상황 모니터링 UI 및 Interactive Chat 창을 제공하며, `engine.py`의 모듈을 호출하여 동작합니다.
