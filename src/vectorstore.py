import os
from typing import List, Dict, Any
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class VectorStoreManager:
    def __init__(
        self, 
        persist_directory: str = "./chroma_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    base_url="https://gms.ssafy.io/gmsapi/api.openai.com/v1/",
    api_key=os.environ.get("GMS_KEY")
)
        
        # 청크 크기와 중복 길이를 설정값으로 관리 (F2005)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def get_meta_store(self) -> Chroma:
        """항목 메타데이터 전용 컬렉션 (RAG 1)"""
        return Chroma(
            collection_name="domain_meta",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

    def get_doc_store(self) -> Chroma:
        """외부 문서 청크 전용 컬렉션 (RAG 2)"""
        return Chroma(
            collection_name="external_docs",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

    def index_metadata_items(self, items: List[Dict[str, Any]], create_text_func) -> int:
        """
        도메인 항목 메타데이터를 Vector DB에 색인합니다. (F2001, F2002)
        - 영화 ID(또는 고유 ID)를 식별자로 사용하여 중복 저장(덮어쓰기) 방지
        """
        store = self.get_meta_store()
        documents = []
        
        for item in items:
            item_id = str(item.get("id", ""))
            text = create_text_func(item)
            
            # 메타데이터 필드 구성
            metadata = {
                "item_id": item_id,
                "title": item.get("title", ""),
                "source_type": "metadata"
            }
            
            documents.append(Document(page_content=text, metadata=metadata))

        # 기존 데이터를 초기화하거나 고유 ID 기반으로 업서트 수행
        # ChromaDB에서 동일한 ID 관리용 텍스트 추가 색인
        if documents:
            # ID를 명시적으로 부여하여 중복 색인 방지
            ids = [doc.metadata["item_id"] for doc in documents]
            store.add_documents(documents=documents, ids=ids)
            
        return len(documents)

    def index_external_documents(self, raw_docs: List[Dict[str, Any]]) -> int:
        """
        수집한 외부 문서를 청킹하여 외부 문서 컬렉션에 색인합니다. (F2005)
        - 청크 식별자 규칙: "항목ID:문서ID:순번"
        """
        store = self.get_doc_store()
        all_splits = []

        for doc_info in raw_docs:
            item_id = str(doc_info.get("item_id", ""))
            doc_id = str(doc_info.get("doc_id", "doc"))
            title = doc_info.get("title", "")
            url = doc_info.get("url", "")
            content = doc_info.get("content", "")

            # 텍스트 청킹 분할
            doc_obj = Document(
                page_content=content,
                metadata={
                    "item_id": item_id,
                    "doc_title": title,
                    "source_url": url,
                    "source_type": "external_doc"
                }
            )
            splits = self.text_splitter.split_documents([doc_obj])

            # 청크 식별자 부여 (덮어쓰기 및 중복 방지)
            for i, split in enumerate(splits):
                split.metadata["chunk_id"] = f"{item_id}:{doc_id}:{i}"
                all_splits.append(split)

        if all_splits:
            ids = [split.metadata["chunk_id"] for split in all_splits]
            store.add_documents(documents=all_splits, ids=ids)

        return len(all_splits)