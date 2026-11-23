import os
import json
import requests
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

class ExternalDocumentCollector:
    def __init__(self):
        # 도메인 전문가 역할의 LLM 설정
        self.llm = ChatOpenAI(
    model="gpt-5.4-nano",
    base_url="https://gms.ssafy.io/gmsapi/api.openai.com/v1/",
    api_key=os.environ.get("GMS_KEY"),
    temperature=0.2
)
        # 이미 수집한 검색어 캐시 (NF2002 요구사항 만족)
        self.collected_queries = set()

    def extract_search_queries(self, item_title: str, user_question: str) -> List[str]:
        """
        [프롬프트 1] 도메인 전문가 역할로 항목과 질문에서 외부 문서 검색어를 추출합니다. (F2003)
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 해당 도메인 전문가입니다. 사용자의 질문과 대상 항목 이름을 바탕으로, 외부 문서(위키백과 등)에서 관련 정보를 찾기 위한 최적의 검색어 목록을 JSON 배열 형식으로 추출해주세요. 오직 JSON 형식의 리스트만 반환하세요. 예: [\"검색어1\", \"검색어2\"]"),
            ("user", "항목 이름: {item_title}\n사용자 질문: {user_question}")
        ])
        
        chain = prompt | self.llm | JsonOutputParser()
        try:
            result = chain.invoke({"item_title": item_title, "user_question": user_question})
            if isinstance(result, list):
                return result
            return [item_title]
        except Exception:
            # 파싱 실패 시 기본 항목 이름 반환
            return [item_title]

    def fetch_wikipedia_docs(self, query: str) -> List[Dict[str, Any]]:
        """
        위키백과 API를 호출하여 문서 후보를 수집합니다. (NF2006: User-Agent 포함)
        """
        if query in self.collected_queries:
            return []  # 이미 수집한 검색어는 다시 수집하지 않음 (NF2002)

        url = "https://ko.wikipedia.org/w/api.php"
        headers = {
            "User-Agent": "SSAFYAICoreProject/2.0 (student@ssafy.com)"
        }
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json"
        }

        response = requests.get(url, params=params, headers=headers)
        if response.status_code != 200:
            return []

        data = response.json()
        search_results = data.get("query", {}).get("search", [])
        
        docs = []
        for item in search_results:
            title = item.get("title", "")
            pageid = item.get("pageid", "")
            snippet = item.get("snippet", "").replace('<span class="searchmatch">', "").replace('</span>', "")
            
            docs.append({
                "doc_id": str(pageid),
                "title": title,
                "url": f"https://ko.wikipedia.org/?curid={pageid}",
                "content": snippet
            })

        self.collected_queries.add(query)
        return docs

    def select_candidate_documents(self, item_title: str, query: str, candidate_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        [프롬프트 2] 후보 문서 중 관련성이 높은 문서를 선별합니다. (F2004)
        - 선택한 항목 자체를 다룬 문서는 제외 규칙보다 우선하여 항상 포함
        """
        if not candidate_docs:
            return []

        # 항목 자체를 다룬 문서는 우선적으로 강제 포함
        selected = []
        other_candidates = []
        for doc in candidate_docs:
            if item_title in doc["title"] or doc["title"] == item_title:
                selected.append(doc)
            else:
                other_candidates.append(doc)

        # 후보가 너무 많다면 프롬프트 2를 통해 상위 문서 선별
        if len(other_candidates) > 0 and len(selected) < 3:
            docs_summary = "\n".join([f"- ID: {d['doc_id']}, 제목: {d['title']}, 요약: {d['content']}" for d in other_candidates])
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "당신은 문서 필터링 전문가입니다. 주어진 항목 이름과 검색어에 가장 부합하고 유용한 문서의 ID 목록을 JSON 배열 형식으로 골라주세요. 예: [\"12345\", \"67890\"]"),
                ("user", "대상 항목: {item_title}\n검색어: {query}\n후보 문서 목록:\n{docs_summary}")
            ])
            
            chain = prompt | self.llm | JsonOutputParser()
            try:
                chosen_ids = chain.invoke({"item_title": item_title, "query": query, "docs_summary": docs_summary})
                if isinstance(chosen_ids, list):
                    for doc in other_candidates:
                        if str(doc["doc_id"]) in [str(i) for i in chosen_ids]:
                            selected.append(doc)
            except Exception:
                # 오류 발생 시 상위 2개 기본 추가
                selected.extend(other_candidates[:2])

        return selected