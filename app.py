import os
import re
import tempfile
from typing import List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# 데이터 전처리 및 청킹 모듈
class DocumentProcessor:
    """
    [자료구조 변화]
    PDF Bytes -> Document Objects -> Cleaned Text -> Text Chunks
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
            length_function=len
        )

    def preprocess_text(self, text: str) -> str:
        """불필요한 공백, 특수문자 제거 및 텍스트 정규화"""
        # 1. 연속된 공백 및 줄바꿈 정리
        text = re.sub(r'\s+', ' ', text)
        # 2. 특수 기호 정제 (문맥 유지 범위 내)
        text = re.sub(r'[^\w\s\.\?\!\,\(\)가-힣]', '', text)
        return text.strip()

    def process_pdf(self, file_path: str) -> List[Document]:
        # Step A: 로드 (File Path -> List[Document])
        loader = PyPDFLoader(file_path)
        raw_docs = loader.load()
        
        # Step B: 정제 (Document.page_content 정문화)
        for doc in raw_docs:
            doc.page_content = self.preprocess_text(doc.page_content)
            
        # Step C: 지능형 청킹 (List[Document] -> List[Split Documents])
        # 문맥 유지를 위해 overlap을 활용하며 의미 단위(세퍼레이터)로 분할
        chunks = self.splitter.split_documents(raw_docs)
        return chunks

# 벡터 인덱싱 및 검색 엔진
class RAGEngine:
    """
    [자료구조 변화]
    Text Chunks -> Vector Embeddings -> FAISS Index (Local Persistence)
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=self.api_key
        )
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro", 
            google_api_key=self.api_key,
            temperature=0
        )
        self.vector_db = None

    def build_index(self, chunks: List[Document], save_path: str = "faiss_index"):
        """벡터 스토어 생성 및 로컬 저장 (API Rate Limit 대응)"""
        import time
        import streamlit as st

        batch_size = 80 # 무료 티어 분당 100건 제한 대응
        
        # 첫 번째 배치로 벡터 DB 초기화
        first_batch = chunks[:batch_size]
        self.vector_db = FAISS.from_documents(first_batch, self.embeddings)

        # 나머지 배치를 순차적으로 추가
        for i in range(batch_size, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            st.toast(f"API Rate Limit 방지를 위해 60초 대기 중... ({i}/{len(chunks)} 처리 완료)")
            time.sleep(60) # 60초 대기
            
            temp_db = FAISS.from_documents(batch, self.embeddings)
            self.vector_db.merge_from(temp_db)

        self.vector_db.save_local(save_path)
        return self.vector_db

    def load_index(self, save_path: str = "faiss_index"):
        """저장된 벡터 스토어 로드"""
        if os.path.exists(save_path):
            self.vector_db = FAISS.load_local(
                save_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            return True
        return False

    def get_qa_chain(self, k: int = 4):
        """검색 및 추론 체인 구성 (Top-K 검색)"""
        if not self.vector_db:
            raise ValueError("벡터 데이터베이스가 초기화되지 않았습니다.")

        # 오직 제공된 Context만 사용하도록 제한하는 프롬프트
        template = """당신은 인공지능 분석가입니다. 
        아래 제공된 [Context]를 바탕으로만 질문에 답하세요. 
        답을 모른다면 솔직하게 모른다고 하세요. 
        답변은 친절하고 논리적인 한국어로 작성하세요.

        [Context]
        {context}

        질문: {question}
        답변:"""
        
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff", # 텍스트를 한데 모아 전달
            retriever=self.vector_db.as_retriever(search_kwargs={"k": k}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True # 답변 근거 문서 포함
        )
        return chain

import streamlit as st

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
                response = qa_chain.invoke("이 문서의 핵심 요약과 우리가 반드시 알아야 할 비즈니스 인사이트 3가지를 도출해줘.")
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