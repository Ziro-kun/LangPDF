import os
import re
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
        # Step A: 로드 
        loader = PyPDFLoader(file_path)
        raw_docs = loader.load()
        
        # Step B: 정제 
        for doc in raw_docs:
            doc.page_content = self.preprocess_text(doc.page_content)
            
        # Step C: 지능형 청킹 
        # 문맥 유지를 위해 overlap을 활용하며 의미 단위로 분할
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

        batch_size = 80 # 무료 티어 에러 해결을 위해 분당 100건 제한 대응
        
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

        # 제공된 Context만 사용하도록 제한하는 프롬프트
        template = """당신은 인공지능 분석가입니다. 
        아래 제공된 [Context]를 바탕으로만 질문에 답하세요. 
        답을 모른다면 솔직하게 모른다고 하세요. 
        답변은 친절하고 논리적으로 한국어로 작성하세요.
        한국어로 답변 시 인코딩은 UTF-8로 해주세요.

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
