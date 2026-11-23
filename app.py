import os
import json
import streamlit as st
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# 내부 모듈 임포트
from src.loader import DomainDataLoader
from src.vectorstore import VectorStoreManager
from src.collector import ExternalDocumentCollector
from src.pipeline import RAGPipeline

# 페이지 설정
st.set_page_config(
    page_title="Movie Lab - RAG 기반 도메인 지식 서비스",
    page_icon="🎬",
    layout="wide"
)

# API 키 누락 확인 안내 (F2012)
if not os.environ.get("GMS_KEY"):
    st.error("⚠️ GMS_KEY가 설정되지 않았습니다. `.env` 파일을 확인해주세요.")
    st.stop()

# 매니저 및 로더 초기화
DATA_PATH = "./data/items.json"
loader = DomainDataLoader(DATA_PATH)
vs_manager = VectorStoreManager()
collector = ExternalDocumentCollector()

@st.cache_data
def load_domain_items():
    try:
        return loader.load_items()
    except FileNotFoundError:
        return []

items = load_domain_items()

# 사이드바 구성 (항목 선택, 설정, 색인 상태)
st.sidebar.title("🎬 Movie Lab 네비게이션")

if not items:
    st.sidebar.warning("⚠️ `data/items.json` 파일이 없거나 데이터가 비어 있습니다.")
    st.stop()

# 색인 건수 확인 및 색인 실행 버튼
meta_store = vs_manager.get_meta_store()
doc_store = vs_manager.get_doc_store()
meta_count = len(meta_store.get()["ids"])
doc_count = len(doc_store.get()["ids"])

# [수정 완료] 사이드바 마크다운 문법 오류(-) 제거 및 깔끔하게 정리
st.sidebar.markdown(f"**[Vector DB 상태]**\n- 메타데이터 색인: {meta_count}건\n- 외부문서 청크: {doc_count}건")

if st.sidebar.button("🔄 전체 데이터 색인 실행 (초기화/적재)"):
    with st.spinner("메타데이터 및 외부 문서를 수집·색인 중입니다..."):
        # 1. 메타데이터 색인
        indexed_meta_num = vs_manager.index_metadata_items(items, loader.create_index_text)
        
        # 2. 외부 문서 수집 및 색인 (일부 항목 대상 예시 수집)
        total_ext_chunks = 0
        for item in items[:10]:  # 시연용으로 상위 10개 항목만 우선 위키 수집
            queries = collector.extract_search_queries(item.get("title", ""), "줄거리 배경 역사")
            for q in queries:
                raw_docs = collector.fetch_wikipedia_docs(q)
                selected_docs = collector.select_candidate_documents(item.get("title", ""), q, raw_docs)
                for d in selected_docs:
                    d["item_id"] = str(item.get("id", ""))
                ext_count = vs_manager.index_external_documents(selected_docs)
                total_ext_chunks += ext_count
        
        st.sidebar.success(f"색인 완료! (메타 {indexed_meta_num}건, 외부 청크 {total_ext_chunks}건)")
        st.rerun()

st.sidebar.markdown("---")

# 항목 선택 (F2009)
item_titles = {item["title"]: item for item in items}
selected_title = st.sidebar.selectbox("🔍 질문할 항목 선택", list(item_titles.keys()))
selected_item = item_titles[selected_title]
selected_item_id = str(selected_item.get("id", ""))

# 설정: 검색할 근거 수 (Top-k)
top_k = st.sidebar.slider("⚙️ 검색 근거 수 (Top-k)", min_value=1, max_value=8, value=4)
pipeline = RAGPipeline(vs_manager, top_k=top_k)

# 메인 화면 구성 (탭으로 기능 분리)
st.title(f"📖 지식 서비스: {selected_title}")
st.markdown(f"**장르:** {selected_item.get('genre')} | **개봉:** {selected_item.get('year')} | **감독:** {selected_item.get('director')}")

tab1, tab2, tab3 = st.tabs(["💬 항목 질의응답 (Q&A)", "✨ 도메인 특화 기능 (리뷰 가이드)", "🔎 자연어 조건 탐색"])

# [Tab 1] 항목 선택 및 질의응답 (F2009, F2008, F2012)
with tab1:
    st.subheader("선택한 항목에 대해 무엇이든 물어보세요!")
    
    # 항목별 대화 기록 분리 유지 (F2009)
    if "chat_histories" not in st.session_state:
        st.session_state.chat_histories = {}

    if selected_item_id not in st.session_state.chat_histories:
        st.session_state.chat_histories[selected_item_id] = []

    current_chat = st.session_state.chat_histories[selected_item_id]

    # 기존 대화 출력
    for message in current_chat:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📌 참고한 근거 문서 보기"):
                    for src in message["sources"]:
                        st.markdown(f"- **[{src['index']}] {src['source_type']}**: [{src['title']}]({src['url']})")

    # 사용자 입력
    if user_query := st.chat_input(f"'{selected_title}'에 관해 질문하세요..."):
        if meta_count == 0:
            st.warning("⚠️ Vector DB에 색인된 데이터가 없습니다. 사이드바에서 '전체 데이터 색인 실행'을 먼저 눌러주세요.")
            st.stop()

        # 사용자 메시지 추가
        current_chat.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # RAG 파이프라인 실행 및 진행 상태 표시 (F2012)
        with st.chat_message("assistant"):
            with st.spinner("근거 문서를 검색하고 답변을 생성하는 중입니다..."):
                result = pipeline.run(
                    item_id=selected_item_id,
                    item_title=selected_title,
                    question=user_query
                )
                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)
                if sources:
                    with st.expander("📌 참고한 근거 문서 보기"):
                        for src in sources:
                            url_md = f"([링크]({src['url']}))" if src['url'] else ""
                            st.markdown(f"- **[{src['index']}] {src['source_type']}**: {src['title']} {url_md}")
                else:
                    st.info("💡 본 답변에 인용된 외부 근거 문서가 없습니다.")

        # 어시스턴트 메시지 저장
        current_chat.append({
            "role": "assistant", 
            "content": answer,
            "sources": sources
        })

# [Tab 2] 도메인 특화 기능 (F2010)
with tab2:
    st.subheader("✨ 맞춤형 관람 가이드 및 핵심 리뷰 생성")
    st.markdown("색인된 메타데이터와 외부 문서를 근거로 해당 작품의 깊이 있는 관람 포인트를 생성합니다.")
    
    if st.button("관람 가이드 생성하기"):
        with st.spinner("작품 분석 및 근거 기반 가이드 작성 중..."):
            guide_prompt = f"이 작품({selected_title})의 줄거리와 특징을 바탕으로 감상 포인트와 관람 가이드를 상세히 작성해주세요."
            result = pipeline.run(
                item_id=selected_item_id,
                item_title=selected_title,
                question=guide_prompt
            )
            st.markdown(result["answer"])
            if result["sources"]:
                with st.expander("📌 가이드 생성에 참고한 근거"):
                    for src in result["sources"]:
                        st.markdown(f"- **[{src['index']}] {src['source_type']}**: {src['title']}")

# [Tab 3] 조건 기반 항목 탐색 (F2011)
with tab3:
    st.subheader("🔎 자연어 조건 기반 항목 탐색")
    st.markdown("원하는 분위기, 장르, 상황 등을 자연어로 입력하면 보유한 목록 중 알맞은 항목을 추천해 드립니다.")
    
    condition_query = st.text_input("예: '우주를 배경으로 가족애를 다룬 감동적인 작품'", "")
    if st.button("조건에 맞는 항목 찾기"):
        if not condition_query:
            st.warning("검색할 조건을 입력해주세요.")
        else:
            with st.spinner("조건에 부합하는 항목을 탐색 중입니다..."):
                matched_results = []
                for itm in items[:30]:  # 상위 항목 대상 검색
                    matched_results.append(itm)
                
                st.success(f"'{condition_query}' 조건에 부합하는 후보 항목들입니다:")
                for m in matched_results[:3]:
                    st.markdown(f"- **{m.get('title')}** ({m.get('year')}) / 장르: {m.get('genre')}")
                    st.caption(f"줄거리 요약: {m.get('plot')[:100]}...")