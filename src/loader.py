import json
import os
from typing import Dict, List

class DomainDataLoader:
    def __init__(self, json_path: str):
        self.json_path = json_path

    def load_items(self) -> List[Dict]:
        """도메인 메타데이터 JSON 파일을 로드합니다. (약 100건 이상)"""
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"메타데이터 파일을 찾을 수 없습니다: {self.json_path}")
        
        with open(self.json_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        return items

    def create_index_text(self, item: Dict) -> str:
        """
        항목 1건의 메타데이터 속성들을 하나로 묶어 색인용 텍스트를 구성합니다.
        (예시: 영화 도메인 기준)
        """
        title = item.get("title", "")
        original_title = item.get("original_title", "")
        year = item.get("year", "")
        genre = item.get("genre", "")
        director = item.get("director", "")
        cast = item.get("cast", "")
        plot = item.get("plot", "")

        # 검색에 잘 걸리도록 자연스러운 문장 형태로 결합
        index_text = (
            f"제목: {title} (원제: {original_title})\n"
            f"개봉연도: {year}\n"
            f"장르: {genre}\n"
            f"감독: {director}\n"
            f"출연: {cast}\n"
            f"줄거리: {plot}"
        )
        return index_text