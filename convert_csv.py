import os
import pandas as pd
import json

# 파일 경로 (다운로드 받은 위치에 맞게 확인)
csv_path = "tmdb_5000_movies.csv"

# CSV 읽기 (상위 100건 추출)
df = pd.read_csv(csv_path).head(100)

items = []
for index, row in df.iterrows():
    # genres 컬럼은 JSON 문자열 형태이므로 장르 이름만 예쁘게 파싱
    genres_str = "드라마"
    try:
        genres_list = json.loads(row.get("genres", "[]"))
        genres_str = ", ".join([g["name"] for g in genres_list])
    except:
        pass

    items.append({
        "id": str(row.get("id", index + 1)),
        "title": str(row.get("title", f"영화 {index + 1}")),
        "original_title": str(row.get("original_title", "")),
        "year": str(row.get("release_date", "2020"))[:4],
        "genre": genres_str,
        "director": "정보 참조", # movies 파일에는 감독 이름이 직접 없으므로 기본값 처리
        "cast": "정보 참조",
        "plot": str(row.get("overview", "줄거리 정보가 없습니다."))
    })

# data/items.json으로 저장
os.makedirs("data", exist_ok=True)
with open("data/items.json", "w", encoding="utf-8") as f:
    json.dump(items, f, ensure_ascii=False, indent=2)

print("✅ tmdb_5000_movies.csv 기반으로 100건의 data/items.json 변환이 완료되었습니다!")