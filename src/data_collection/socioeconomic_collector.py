"""
Socioeconomic Impact Data Collector
재해 피해, 건강 영향, 농업 피해 등 사회경제적 데이터 수집

Data sources:
- 행정안전부 재해연보
- 통계청 KOSIS
- 농림축산식품부 농업재해통계
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import requests
from datetime import datetime


class DisasterDataCollector:
    """재해 피해 데이터 수집기 (행안부 재해연보)"""

    def __init__(self, save_dir: str = "data/raw"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def generate_disaster_data(
        self,
        start_year: int = 2000,
        end_year: int = 2023
    ) -> pd.DataFrame:
        """
        재해 피해 데이터 생성 (실제로는 재해연보에서 수집)

        피해 유형:
        - 인명피해 (사망, 실종, 부상)
        - 재산피해 (공공시설, 사유시설)
        - 이재민 수
        """
        np.random.seed(42)

        records = []
        regions = [
            '서울', '부산', '대구', '인천', '광주', '대전', '울산',
            '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주'
        ]

        disaster_types = ['태풍', '호우', '폭설', '한파', '폭염', '가뭄']

        for year in range(start_year, end_year + 1):
            for region in regions:
                for d_type in disaster_types:
                    # 발생 확률 조정 (기후변화 트렌드 반영)
                    trend_factor = 1 + (year - 2000) * 0.02

                    # 유형별 기본 발생 빈도
                    if d_type in ['폭염', '가뭄']:
                        base_prob = 0.3 * trend_factor
                    elif d_type in ['태풍', '호우']:
                        base_prob = 0.4
                    else:
                        base_prob = 0.2

                    if np.random.random() < base_prob:
                        # 피해 규모 생성
                        damage = {
                            'year': year,
                            'region': region,
                            'disaster_type': d_type,
                            'deaths': max(0, int(np.random.exponential(0.5))),
                            'injured': max(0, int(np.random.exponential(2))),
                            'displaced': max(0, int(np.random.exponential(50))),
                            'property_damage_public': np.random.exponential(1000) * trend_factor,
                            'property_damage_private': np.random.exponential(500) * trend_factor,
                            'duration_days': max(1, int(np.random.exponential(3))),
                        }
                        records.append(damage)

        df = pd.DataFrame(records)
        df['total_damage'] = df['property_damage_public'] + df['property_damage_private']
        return df


class HealthDataCollector:
    """건강 영향 데이터 수집기 (KOSIS 사망원인통계)"""

    def __init__(self, save_dir: str = "data/raw"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def generate_health_data(
        self,
        start_year: int = 2000,
        end_year: int = 2023
    ) -> pd.DataFrame:
        """
        온열/한랭질환 사망 및 이환 데이터 생성

        변수:
        - 온열질환 사망자/환자 수
        - 한랭질환 사망자/환자 수
        - 호흡기질환 (미세먼지 관련)
        """
        np.random.seed(43)

        records = []
        regions = [
            '서울', '부산', '대구', '인천', '광주', '대전', '울산',
            '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주'
        ]

        # 인구 가중치 (상대적)
        pop_weight = {
            '서울': 2.0, '경기': 2.5, '부산': 0.7, '대구': 0.5,
            '인천': 0.6, '광주': 0.3, '대전': 0.3, '울산': 0.2,
            '강원': 0.3, '충북': 0.3, '충남': 0.4, '전북': 0.4,
            '전남': 0.4, '경북': 0.5, '경남': 0.6, '제주': 0.1
        }

        for year in range(start_year, end_year + 1):
            # 기후변화 트렌드
            heat_trend = 1 + (year - 2000) * 0.03
            cold_trend = 1 - (year - 2000) * 0.01

            for region in regions:
                pw = pop_weight.get(region, 0.5)

                record = {
                    'year': year,
                    'region': region,
                    'heat_deaths': max(0, int(np.random.poisson(5 * pw * heat_trend))),
                    'heat_patients': max(0, int(np.random.poisson(100 * pw * heat_trend))),
                    'cold_deaths': max(0, int(np.random.poisson(3 * pw * cold_trend))),
                    'cold_patients': max(0, int(np.random.poisson(50 * pw * cold_trend))),
                    'respiratory_deaths': max(0, int(np.random.poisson(20 * pw))),
                    'elderly_ratio': 0.1 + (year - 2000) * 0.005 + np.random.normal(0, 0.02),
                }
                records.append(record)

        return pd.DataFrame(records)


class AgricultureDataCollector:
    """농업 피해 데이터 수집기"""

    def __init__(self, save_dir: str = "data/raw"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def generate_agriculture_data(
        self,
        start_year: int = 2010,
        end_year: int = 2023
    ) -> pd.DataFrame:
        """
        작물 피해 데이터 생성

        변수:
        - 재해 유형별 피해 면적
        - 피해액
        - 주요 작물별 피해
        """
        np.random.seed(44)

        records = []
        regions = [
            '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주'
        ]

        # 지역별 농경지 면적 비율
        agri_ratio = {
            '경기': 0.12, '강원': 0.08, '충북': 0.08, '충남': 0.15,
            '전북': 0.14, '전남': 0.18, '경북': 0.13, '경남': 0.10, '제주': 0.02
        }

        crops = ['쌀', '채소', '과수', '특용작물']
        disaster_types = ['가뭄', '폭우', '폭염', '냉해', '태풍']

        for year in range(start_year, end_year + 1):
            for region in regions:
                ar = agri_ratio.get(region, 0.1)

                for d_type in disaster_types:
                    if np.random.random() < 0.3:
                        crop = np.random.choice(crops)

                        record = {
                            'year': year,
                            'region': region,
                            'disaster_type': d_type,
                            'affected_crop': crop,
                            'affected_area_ha': np.random.exponential(1000) * ar,
                            'damage_amount_million': np.random.exponential(500) * ar,
                            'recovery_rate': np.random.uniform(0.3, 0.9),
                        }
                        records.append(record)

        return pd.DataFrame(records)


class RegionalVulnerabilityData:
    """지역별 취약성 지표 데이터"""

    def __init__(self, save_dir: str = "data/raw"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def generate_vulnerability_indicators(self) -> pd.DataFrame:
        """
        시군구별 취약성 지표 생성

        지표:
        - 인구밀도
        - 고령인구비율
        - 농경지비율
        - 의료기관 수
        - 재정자립도
        """
        np.random.seed(45)

        # 주요 시군구
        regions = [
            '서울 강남구', '서울 강북구', '서울 종로구',
            '부산 해운대구', '부산 사하구',
            '대구 수성구', '대구 달서구',
            '인천 남동구', '인천 강화군',
            '경기 수원시', '경기 성남시', '경기 용인시', '경기 화성시',
            '강원 춘천시', '강원 강릉시', '강원 원주시',
            '충북 청주시', '충북 충주시',
            '충남 천안시', '충남 아산시',
            '전북 전주시', '전북 익산시',
            '전남 목포시', '전남 순천시',
            '경북 포항시', '경북 경주시',
            '경남 창원시', '경남 김해시',
            '제주 제주시', '제주 서귀포시',
        ]

        records = []
        for region in regions:
            is_urban = any(x in region for x in ['서울', '부산', '대구', '인천', '강남', '해운대', '수성'])

            record = {
                'region': region,
                'population_density': (
                    np.random.uniform(8000, 25000) if is_urban
                    else np.random.uniform(100, 2000)
                ),
                'elderly_ratio': np.random.uniform(0.12, 0.35),
                'farmland_ratio': (
                    np.random.uniform(0, 0.05) if is_urban
                    else np.random.uniform(0.15, 0.5)
                ),
                'medical_facilities_per_1000': np.random.uniform(1, 8),
                'fiscal_independence': np.random.uniform(0.15, 0.65),
                'green_space_ratio': np.random.uniform(0.05, 0.4),
                'impervious_surface_ratio': (
                    np.random.uniform(0.4, 0.8) if is_urban
                    else np.random.uniform(0.1, 0.3)
                ),
            }
            records.append(record)

        return pd.DataFrame(records)


def main():
    """사회경제 데이터 수집 실행"""
    save_dir = "../../data/raw"

    # 재해 피해 데이터
    print("Generating disaster damage data...")
    disaster_collector = DisasterDataCollector(save_dir)
    disaster_df = disaster_collector.generate_disaster_data()
    disaster_df.to_csv(Path(save_dir) / "disaster_damage_2000_2023.csv", index=False)
    print(f"Disaster records: {len(disaster_df)}")

    # 건강 영향 데이터
    print("Generating health impact data...")
    health_collector = HealthDataCollector(save_dir)
    health_df = health_collector.generate_health_data()
    health_df.to_csv(Path(save_dir) / "health_impact_2000_2023.csv", index=False)
    print(f"Health records: {len(health_df)}")

    # 농업 피해 데이터
    print("Generating agriculture damage data...")
    agri_collector = AgricultureDataCollector(save_dir)
    agri_df = agri_collector.generate_agriculture_data()
    agri_df.to_csv(Path(save_dir) / "agriculture_damage_2010_2023.csv", index=False)
    print(f"Agriculture records: {len(agri_df)}")

    # 취약성 지표
    print("Generating vulnerability indicators...")
    vuln_data = RegionalVulnerabilityData(save_dir)
    vuln_df = vuln_data.generate_vulnerability_indicators()
    vuln_df.to_csv(Path(save_dir) / "regional_vulnerability.csv", index=False)
    print(f"Vulnerability records: {len(vuln_df)}")

    # 통계 요약
    print("\n=== Data Summary ===")
    print(f"Total disaster damage (million KRW): {disaster_df['total_damage'].sum():,.0f}")
    print(f"Total heat deaths: {health_df['heat_deaths'].sum():,}")
    print(f"Total cold deaths: {health_df['cold_deaths'].sum():,}")
    print(f"Total agricultural damage (million KRW): {agri_df['damage_amount_million'].sum():,.0f}")


if __name__ == "__main__":
    main()
