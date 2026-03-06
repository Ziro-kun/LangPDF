import os
import tempfile
import streamlit as st

from engine import DocumentProcessor, RAGEngine

def main():
    st.set_page_config(page_title="Enterprise RAG Solution", layout="wide")
    st.title("🚀 Enterprise RAG Insights Tool")

    # API Key 설정
    with st.sidebar:
        api_key = st.text_input("Gemini API Key", type="password")
        st.info("이 도구는 PDF 정제, 지능형 청킹, FAISS 인덱싱을 거쳐 답변을 생성합니다.")

    if not api_key:
        st.warning("API Key를 입력해주세요.")
        st.stop()

    # 객체 초기화
    if api_key:
        processor = DocumentProcessor(chunk_size=1000, chunk_overlap=150)
        engine = RAGEngine(api_key=api_key)
    else:
        processor = None
        engine = None

    uploaded_file = st.file_uploader("분석할 PDF 파일을 업로드하세요", type="pdf")

    if uploaded_file:
        # 파일 처리 프로세스
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        with st.status("데이터 처리 중...", expanded=True) as status:
            st.write("1. 텍스트 추출 및 정제 중...")
            chunks = processor.process_pdf(tmp_path)
            
            st.write(f"2. {len(chunks)}개의 청크 생성 완료. 벡터화 진행 중...")
            engine.build_index(chunks)
            
            status.update(label="인덱싱 완료!", state="complete", expanded=False)
        
        # 인사이트 도출 버튼
        if st.button("핵심 인사이트 도출"):
            qa_chain = engine.get_qa_chain()
            with st.spinner("AI 분석 중..."):
                response = qa_chain.invoke("이 문서의 핵심 요약을 수행하고, 우리가 반드시 알아야 할 비즈니스 인사이트 3가지를 도출해줘.")
                st.subheader("💡 도출된 인사이트")
                st.markdown(response["result"])
                
                with st.expander("사용한 참조 문맥 확인"):
                    for i, doc in enumerate(response["source_documents"]):
                        st.caption(f"Source {i+1}: Page {doc.metadata.get('page', 'N/A')}")
                        st.write(doc.page_content[:300] + "...")

        # 질의응답 Input
        st.divider()
        query = st.text_input("문서에 대해 구체적인 질문을 입력하세요:")
        if query:
            qa_chain = engine.get_qa_chain(k=5)
            with st.spinner("답변 생성 중..."):
                res = qa_chain.invoke(query)
                st.markdown(f"**A:** {res['result']}")

        os.remove(tmp_path)

if __name__ == "__main__":
    main()