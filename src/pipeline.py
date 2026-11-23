import os
from typing import List, Dict, Any, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from src.vectorstore import VectorStoreManager

# LangGraph 상태 정의
class RAGState(TypedDict):
    item_id: str
    item_title: str
    question: str
    retrieved_docs: List[Document]
    answer: str
    sources: List[Dict[str, str]]

class RAGPipeline:
    def __init__(self, vectorstore_manager: VectorStoreManager, top_k: int = 4):
        self.vs_manager = vectorstore_manager
        self.top_k = top_k
        self.llm = ChatOpenAI(
    model="gpt-5.4-nano",
    base_url="https://gms.ssafy.io/gmsapi/api.openai.com/v1/",
    api_key=os.environ.get("GMS_KEY"),
    temperature=0.1
)
        self.graph = self._build_graph()

    def _retrieve_node(self, state: RAGState) -> RAGState:
        """
        [Retrieve 노드] 메타데이터 컬렉션과 외부 문서 컬렉션을 검색하여 병합합니다. (F2006)
        - 선택한 항목의 메타데이터는 항상 포함
        - 의미 유사도로 외부 문서 청크 검색 (Top-k 설정 반영)
        """
        item_id = state["item_id"]
        question = state["question"]
        
        meta_store = self.vs_manager.get_meta_store()
        doc_store = self.vs_manager.get_doc_store()

        retrieved_docs = []

        # 1. 항목 메타데이터는 항상 검색 없이/필터링으로 무조건 포함
        meta_results = meta_store.similarity_search(
            query=state["item_title"], 
            k=1, 
            filter={"item_id": item_id}
        )
        retrieved_docs.extend(meta_results)

        # 2. 외부 문서 청크는 사용자 질문과 의미 유사도로 검색 (Top-k 반영, 해당 item_id 필터링)
        ext_results = doc_store.similarity_search(
            query=question, 
            k=self.top_k, 
            filter={"item_id": item_id}
        )
        retrieved_docs.extend(ext_results)

        # 중복 청크 제거 (content 기준)
        unique_docs = []
        seen_contents = set()
        for doc in retrieved_docs:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                unique_docs.append(doc)

        state["retrieved_docs"] = unique_docs
        return state

    def _generate_node(self, state: RAGState) -> RAGState:
        """
        [Generate 노드] [프롬프트 3] 검색된 문서만을 근거로 답변을 생성하고 근거 번호를 부여합니다. (F2007, F2008)
        """
        docs = state["retrieved_docs"]
        question = state["question"]
        item_title = state["item_title"]

        if not docs:
            state["answer"] = "제공된 문서 및 메타데이터에서 해당 질문에 대한 근거를 확인할 수 없습니다."
            state["sources"] = []
            return state

        # 근거 문서 컨텍스트 구성 및 출처 매핑
        context_texts = []
        sources = []
        for i, doc in enumerate(docs, 1):
            source_type = "항목 메타데이터" if doc.metadata.get("source_type") == "metadata" else "외부 문서(위키백과)"
            title = doc.metadata.get("title") or doc.metadata.get("doc_title", item_title)
            url = doc.metadata.get("source_url", "")

            context_texts.append(f"[근거 {i}] ({source_type}) {title}\n{doc.page_content}")
            sources.append({
                "index": i,
                "source_type": source_type,
                "title": title,
                "url": url
            })

        context_str = "\n\n".join(context_texts)

        # 답변 생성 프롬프트 3
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "당신은 신뢰할 수 있는 도메인 지식 챗봇입니다.\n"
             "반드시 아래에 제공된 [근거] 내용만을 기반으로 사용자의 질문에 답하세요.\n"
             "근거 문서에 없는 내용은 추측하거나 지어내지 말고, '제공된 문서에서 확인되지 않습니다'라고 답변하세요.\n"
             "답변의 주요 주장이나 사실마다 어떤 근거를 참고했는지 [근거 번호]를 반드시 표기해주세요."
            ),
            ("user", "대상 항목: {item_title}\n\n참고 근거 문서:\n{context_str}\n\n사용자 질문: {question}")
        ])

        chain = prompt | self.llm
        try:
            response = chain.invoke({
                "item_title": item_title,
                "context_str": context_str,
                "question": question
            })
            state["answer"] = response.content
        except Exception as e:
            state["answer"] = f"답변 생성 중 오류가 발생했습니다: {str(e)}"

        state["sources"] = sources
        return state

    def _build_graph(self) -> StateGraph:
        """LangGraph를 retrieve와 generate 두 노드로 구성합니다. (F2007)"""
        workflow = StateGraph(RAGState)

        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("generate", self._generate_node)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()

    def run(self, item_id: str, item_title: str, question: str) -> Dict[str, Any]:
        """파이프라인을 실행하고 결과(답변 및 근거 목록)를 반환합니다."""
        initial_state = {
            "item_id": item_id,
            "item_title": item_title,
            "question": question,
            "retrieved_docs": [],
            "answer": "",
            "sources": []
        }
        result = self.graph.invoke(initial_state)
        return {
            "answer": result["answer"],
            "sources": result["sources"]
        }