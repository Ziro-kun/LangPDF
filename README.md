# LangPDF — RAG Insights Tool (Flask + Vercel)

Google Gemini API, LangChain, FAISS를 활용해 PDF 문서를 분석하고 핵심 인사이트를 도출하는 RAG 기반 AI 시스템입니다.  
백엔드는 **Flask REST API**로 구성되어 **Vercel**에 배포됩니다.

## 🚀 주요 기능

- **PDF 추출 및 정제**: 업로드된 PDF에서 텍스트를 추출하고 불필요한 특수문자·공백을 정제합니다.
- **지능형 문서 청킹**: 문맥 손실을 최소화하기 위해 Semantic 단위와 일정 길이의 Overlap을 두고 문서를 청크로 분할합니다.
- **벡터 인덱싱 (FAISS)**: `models/gemini-embedding-001` 모델로 문서를 벡터화·인덱싱합니다. 무료 티어 Rate Limit 대응을 위해 60초 대기 로직이 적용되어 있습니다.
- **Insight 및 Q&A 생성**: 핵심 비즈니스 인사이트 3가지 자동 추출 및 자유 질의응답 REST API를 제공합니다.

## 📡 API 엔드포인트

| Method | Path            | 설명                                    |
| ------ | --------------- | --------------------------------------- |
| `POST` | `/api/upload`   | PDF 업로드 + 인덱싱 → `session_id` 반환 |
| `POST` | `/api/insights` | 핵심 인사이트 3가지 도출                |
| `POST` | `/api/query`    | 자유 질의응답                           |

### 예시

```bash
# 1. PDF 업로드
curl -X POST https://<your-domain>/api/upload \
  -F "file=@document.pdf" \
  -F "api_key=YOUR_GEMINI_KEY"
# → { "session_id": "...", "chunks": 42 }

# 2. 인사이트 도출
curl -X POST https://<your-domain>/api/insights \
  -H "Content-Type: application/json" \
  -d '{"session_id": "..."}'

# 3. 질의응답
curl -X POST https://<your-domain>/api/query \
  -H "Content-Type: application/json" \
  -d '{"session_id": "...", "question": "이 문서의 주요 리스크는?"}'
```

## ⚙️ 요구 사항 및 설치

1. **Python 3.10+**
2. **라이브러리 설치**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Google API Key**: 요청마다 `api_key` 파라미터로 전달합니다.

## 💻 로컬 실행

```bash
python app.py
```

기본 포트 `http://localhost:5000` 에서 실행됩니다.

## ☁️ Vercel 배포

```bash
vercel deploy
```

> **주의**: Vercel은 서버리스(stateless) 환경이므로 함수 재시작 시 메모리 내 세션이 초기화됩니다.  
> 영구적인 인덱스 유지가 필요하다면 FAISS 인덱스를 외부 스토리지(예: Redis, S3)에 저장하는 방식으로 확장해야 합니다.

## 📁 코드 구조

- **`engine.py`**
  - `DocumentProcessor`: PDF 로드 → 텍스트 정제 → 청킹 파이프라인
  - `RAGEngine`: FAISS 인덱싱, Rate Limit 대응 배치 처리, QA 체인 구성
- **`app.py`**: Flask REST API 서버. 업로드·인사이트·질의응답 엔드포인트 제공
- **`vercel.json`**: `@vercel/python` 런타임으로 Flask 배포 설정
