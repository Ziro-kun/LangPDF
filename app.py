import os
import uuid
import tempfile
from flask import Flask, request, jsonify, send_from_directory
from engine import DocumentProcessor, RAGEngine

app = Flask(__name__, static_folder="static", static_url_path="")


# ── 정적 파일 서빙 ────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# ── 1. PDF 업로드 & 인덱싱 ───────────────────────────────────────────────────
@app.route("/api/upload", methods=["POST"])
def upload():
    """
    Form-data:
      - file   : PDF 파일
      - api_key: Gemini API Key
    Response:
      { "session_id": "...", "chunks": N }

    Vercel 서버리스는 인스턴스간 메모리를 공유하지 않으므로
    FAISS 인덱스를 /tmp/{session_id}/ 에 파일로 저장합니다.
    """
    api_key = request.form.get("api_key", "").strip()
    if not api_key:
        return jsonify({"error": "api_key가 필요합니다."}), 400

    uploaded = request.files.get("file")
    if not uploaded or not uploaded.filename.endswith(".pdf"):
        return jsonify({"error": "PDF 파일을 업로드해주세요."}), 400

    session_id = str(uuid.uuid4())
    index_path = f"/tmp/{session_id}"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        uploaded.save(tmp.name)
        tmp_path = tmp.name

    try:
        processor = DocumentProcessor(chunk_size=1000, chunk_overlap=150)
        chunks = processor.process_pdf(tmp_path)

        engine = RAGEngine(api_key=api_key)
        engine.build_index(chunks, save_path=index_path)

        return jsonify({
            "session_id": session_id,
            "chunks": len(chunks),
            "message": f"{len(chunks)}개의 청크가 인덱싱되었습니다."
        })
    finally:
        os.remove(tmp_path)


def _load_engine(session_id: str, api_key: str):
    """세션 ID로 저장된 FAISS 인덱스를 로드해 RAGEngine을 반환합니다."""
    index_path = f"/tmp/{session_id}"
    if not os.path.exists(index_path):
        return None
    engine = RAGEngine(api_key=api_key)
    engine.load_index(save_path=index_path)
    return engine


# ── 2. 인사이트 도출 ─────────────────────────────────────────────────────────
@app.route("/api/insights", methods=["POST"])
def insights():
    """
    JSON body: { "session_id": "...", "api_key": "..." }
    """
    data = request.get_json(force=True)
    session_id = data.get("session_id", "")
    api_key = data.get("api_key", "").strip()

    if not api_key:
        return jsonify({"error": "api_key가 필요합니다."}), 400

    engine = _load_engine(session_id, api_key)
    if not engine:
        return jsonify({"error": "세션을 찾을 수 없습니다. PDF를 다시 업로드해주세요."}), 404

    qa_chain = engine.get_qa_chain()
    response = qa_chain.invoke(
        "이 문서의 핵심 요약을 수행하고, 우리가 반드시 알아야 할 비즈니스 인사이트 3가지를 도출해줘."
    )
    sources = [
        {"page": doc.metadata.get("page", "N/A"), "content": doc.page_content[:300]}
        for doc in response["source_documents"]
    ]
    return jsonify({"result": response["result"], "sources": sources})


# ── 3. 질의응답 ───────────────────────────────────────────────────────────────
@app.route("/api/query", methods=["POST"])
def query():
    """
    JSON body: { "session_id": "...", "api_key": "...", "question": "..." }
    """
    data = request.get_json(force=True)
    session_id = data.get("session_id", "")
    api_key = data.get("api_key", "").strip()
    question = data.get("question", "").strip()

    if not api_key:
        return jsonify({"error": "api_key가 필요합니다."}), 400
    if not question:
        return jsonify({"error": "질문을 입력해주세요."}), 400

    engine = _load_engine(session_id, api_key)
    if not engine:
        return jsonify({"error": "세션을 찾을 수 없습니다. PDF를 다시 업로드해주세요."}), 404

    qa_chain = engine.get_qa_chain(k=5)
    res = qa_chain.invoke(question)
    sources = [
        {"page": doc.metadata.get("page", "N/A"), "content": doc.page_content[:300]}
        for doc in res["source_documents"]
    ]
    return jsonify({"result": res["result"], "sources": sources})


if __name__ == "__main__":
    app.run(debug=True)