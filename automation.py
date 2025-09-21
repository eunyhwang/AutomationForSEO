from google.cloud import bigquery
import pandas as pd
import requests
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

keywords_df = pd.read_csv("data/expanded_keywords.csv")  # 확장 키워드 데이터 예시

def select_top_keywords(
    min_search_volume=100,
    max_difficulty=50,
    min_cpc=0.5,
    sim_threshold=0.2,
    min_word_count=3,       # long-tail filter
    top_n=500,
    bq_project="kaggle-big-query-2025",
    input_table="seo_demo.expanded_keywords",
    output_table="seo_demo.top_keywords"
):
    """
    Seed keyword 기반 확장 키워드 선별 및 BigQuery 적재 함수 (Long-tail 포함)

    Parameters:
    - min_search_volume: 최소 검색량 필터
    - max_difficulty: 최대 난이도 필터
    - min_cpc: 최소 CPC 필터
    - sim_threshold: seed keyword와 유사성 임계치
    - min_word_count: 최소 단어 수 (long-tail 키워드)
    - top_n: 최종 선택 키워드 수
    - bq_project: BigQuery 프로젝트
    - input_table: 확장 키워드 테이블
    - output_table: 결과 테이블
    """

    client = bigquery.Client(project=bq_project)

    # 1️⃣ 확장 키워드 데이터 불러오기
    query = f"""
    SELECT
        keyword,
        search_volume,
        difficulty,
        cpc,
        seed_keyword
    FROM
        {input_table}
    WHERE
        search_volume IS NOT NULL
        AND difficulty IS NOT NULL
    """
    df = client.query(query).to_dataframe()

    # 2️⃣ 기본 필터링 (Search Volume, Difficulty, CPC)
    df_filtered = df[
        (df['search_volume'] >= min_search_volume) &
        (df['difficulty'] <= max_difficulty) &
        (df['cpc'] >= min_cpc)
    ]

    # 3️⃣ Long-tail 필터링
    df_filtered['word_count'] = df_filtered['keyword'].apply(lambda x: len(x.split()))
    df_filtered = df_filtered[df_filtered['word_count'] >= min_word_count]

    # 4️⃣ Seed Keyword와 관련성 점수 계산 (TF-IDF + Cosine Similarity)
    seed_texts = df_filtered['seed_keyword'].unique().tolist()
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_filtered['keyword'].tolist() + seed_texts)

    cos_sim = cosine_similarity(tfidf_matrix[:len(df_filtered)], tfidf_matrix[len(df_filtered):])
    df_filtered['similarity_to_seed'] = cos_sim.max(axis=1)

    # 5️⃣ 유사성 기준 필터링
    df_filtered = df_filtered[df_filtered['similarity_to_seed'] >= sim_threshold]

    # 6️⃣ Top N 선택 (검색량 우선)
    df_top = df_filtered.sort_values(by='search_volume', ascending=False).head(top_n)

    # 7️⃣ BigQuery에 적재
    client.load_table_from_dataframe(
        df_top,
        output_table,
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    )

    print(f"Top {top_n} long-tail 확장 키워드 선별 완료! 결과 테이블: {output_table}")
    return df_top


def serp_analyser(
    keywords_df,              # select_top_keywords 결과 DataFrame
    serp_api_key,             # SERP API Key (예: SerpAPI, Google Custom Search)
    bq_project="your-project-id",
    output_table="seo_demo.serp_results",
    batch_size=50,
    pause_seconds=2           # API 호출 간 대기 시간 (rate limit 방지)
):
    """
    Top keywords 기반 SERP 분석 + Intent/Content 정보 추출 + BigQuery 적재

    Parameters:
    - keywords_df: select_top_keywords 결과 DataFrame
    - serp_api_key: SERP API Key
    - bq_project: BigQuery 프로젝트
    - output_table: BigQuery에 적재할 테이블
    - batch_size: 한 번에 처리할 키워드 개수
    - pause_seconds: batch 간 pause 시간
    """
    
    client = bigquery.Client(project=bq_project)
    serp_results = []

    # -------------------------------
    # 1️⃣ Batch 단위로 SERP 크롤링
    # -------------------------------
    for i in range(0, len(keywords_df), batch_size):
        batch_keywords = keywords_df.iloc[i:i+batch_size]['keyword'].tolist()

        for kw in batch_keywords:
            # 여기서는 SerpAPI 또는 Google Custom Search API 예시
            # 실제 구현 시 API 문서 참고
            response = requests.get(
                "https://serpapi.com/search",
                params={
                    "q": kw,
                    "api_key": serp_api_key,
                    "engine": "google",
                    "num": 10
                }
            ).json()

            for idx, item in enumerate(response.get('organic_results', []), start=1):
                serp_results.append({
                    "keyword": kw,
                    "rank": idx,
                    "title": item.get('title', ''),
                    "link": item.get('link', ''),
                    "snippet": item.get('snippet', ''),
                    "serp_type": "organic",
                    "timestamp": datetime.utcnow()
                })
            
            time.sleep(0.1)  # API rate-limit 방지

        time.sleep(pause_seconds)  # batch 단위 pause

    # -------------------------------
    # 2️⃣ DataFrame 변환 및 NLP/Intent 예측용 컬럼 준비
    # -------------------------------
    df_serp = pd.DataFrame(serp_results)
    
    # main_entities 추출 (간단 TF-IDF 키워드 예시)
    tfidf = TfidfVectorizer(stop_words='english', max_features=5)
    tfidf_matrix = tfidf.fit_transform(df_serp['title'].fillna('') + ' ' + df_serp['snippet'].fillna(''))
    feature_names = tfidf.get_feature_names_out()
    df_serp['main_entities'] = [', '.join(feature_names)] * len(df_serp)  # 간단 예시

    # content_type 단순 예시: title snippet 기반 keyword 포함 여부
    def detect_content_type(row):
        t = row['title'].lower() + ' ' + row['snippet'].lower()
        if 'how to' in t or 'guide' in t or 'tutorial' in t:
            return 'informational'
        elif 'buy' in t or 'price' in t or 'review' in t:
            return 'transactional'
        else:
            return 'navigational'

    df_serp['content_type'] = df_serp.apply(detect_content_type, axis=1)

    # -------------------------------
    # 3️⃣ BigQuery 적재
    # -------------------------------
    client.load_table_from_dataframe(
        df_serp,
        output_table,
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    )

    print(f"SERP 분석 + Intent/Content 정보 BigQuery 적재 완료: {output_table}")
    return df_serp
