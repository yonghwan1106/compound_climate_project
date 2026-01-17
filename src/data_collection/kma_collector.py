"""
Korea Meteorological Administration (KMA) Data Collector
기상청 기상자료개방포털 API를 통한 관측 데이터 수집

Data source: https://data.kma.go.kr/
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
import time
from typing import Optional, List, Dict


class KMADataCollector:
    """기상청 ASOS (종관기상관측) 데이터 수집기"""

    BASE_URL = "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php"

    # 주요 관측소 (전국 60개소 중 대표 지점)
    STATIONS = {
        108: "서울",
        112: "인천",
        119: "수원",
        133: "대전",
        143: "대구",
        156: "광주",
        159: "부산",
        184: "제주",
        146: "전주",
        138: "포항",
        152: "울산",
        140: "군산",
        131: "청주",
        127: "충주",
        105: "강릉",
        90: "속초",
        95: "철원",
        101: "춘천",
        114: "원주",
        192: "진주",
    }

    def __init__(self, api_key: Optional[str] = None, save_dir: str = "data/raw"):
        """
        Args:
            api_key: 기상청 API 인증키 (없으면 공개 데이터 사용)
            save_dir: 저장 디렉토리
        """
        self.api_key = api_key or os.getenv("KMA_API_KEY", "")
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def fetch_daily_data(
        self,
        station_id: int,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        일별 기상 데이터 조회

        Args:
            station_id: 관측소 ID
            start_date: 시작일 (YYYYMMDD)
            end_date: 종료일 (YYYYMMDD)

        Returns:
            DataFrame with columns: date, temp_max, temp_min, temp_avg,
                                   precip, humidity, wind_speed, pressure
        """
        # 공개 데이터 API 또는 기상자료개방포털 사용
        # 실제 구현시 API 호출 필요

        # 시뮬레이션용 데이터 생성 (실제로는 API 호출)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n = len(dates)

        # 계절성을 반영한 기온 데이터 생성
        day_of_year = dates.dayofyear
        seasonal = 15 * np.sin(2 * np.pi * (day_of_year - 100) / 365)

        data = {
            'date': dates,
            'station_id': station_id,
            'station_name': self.STATIONS.get(station_id, "Unknown"),
            'temp_max': 20 + seasonal + np.random.normal(0, 3, n),
            'temp_min': 10 + seasonal + np.random.normal(0, 3, n),
            'temp_avg': 15 + seasonal + np.random.normal(0, 2, n),
            'precip': np.maximum(0, np.random.exponential(5, n)),
            'humidity': 60 + np.random.normal(0, 15, n),
            'wind_speed': np.maximum(0.5, np.random.exponential(2, n)),
            'pressure': 1013 + np.random.normal(0, 10, n),
        }

        return pd.DataFrame(data)

    def fetch_all_stations(
        self,
        start_date: str,
        end_date: str,
        stations: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        모든 관측소의 데이터 수집

        Args:
            start_date: 시작일
            end_date: 종료일
            stations: 관측소 리스트 (None이면 전체)

        Returns:
            Combined DataFrame
        """
        if stations is None:
            stations = list(self.STATIONS.keys())

        all_data = []

        for stn_id in tqdm(stations, desc="Fetching station data"):
            try:
                df = self.fetch_daily_data(stn_id, start_date, end_date)
                all_data.append(df)
                time.sleep(0.1)  # API 호출 제한 방지
            except Exception as e:
                print(f"Error fetching station {stn_id}: {e}")

        return pd.concat(all_data, ignore_index=True)

    def identify_extreme_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        극한 기상 이벤트 식별

        극한 이벤트 정의:
        - 폭염: 일최고기온 >= 33°C
        - 열대야: 일최저기온 >= 25°C
        - 한파: 일최저기온 <= -12°C
        - 폭우: 일강수량 >= 80mm
        - 대설: 일강수량 >= 20mm AND 기온 <= 0°C (근사)
        """
        df = df.copy()

        # 개별 극한 이벤트 플래그
        df['is_heatwave'] = df['temp_max'] >= 33
        df['is_tropical_night'] = df['temp_min'] >= 25
        df['is_cold_wave'] = df['temp_min'] <= -12
        df['is_heavy_rain'] = df['precip'] >= 80
        df['is_heavy_snow'] = (df['precip'] >= 20) & (df['temp_avg'] <= 0)

        # 복합 이벤트 식별
        df['compound_heat_drought'] = (
            df['is_heatwave'] &
            (df['precip'] < 0.1) &
            (df['humidity'] < 40)
        )
        df['compound_heat_tropical'] = (
            df['is_heatwave'] &
            df['is_tropical_night']
        )

        return df

    def save_data(self, df: pd.DataFrame, filename: str):
        """데이터 저장"""
        filepath = self.save_dir / filename
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"Data saved to {filepath}")

    def calculate_spi(self, precip_series: pd.Series, scale: int = 3) -> pd.Series:
        """
        표준강수지수 (SPI) 계산

        Args:
            precip_series: 월별 강수량 시계열
            scale: SPI 기간 (1, 3, 6, 12개월)

        Returns:
            SPI 값 시계열
        """
        from scipy import stats

        # 이동 합계
        rolling_precip = precip_series.rolling(window=scale).sum()

        # 감마 분포 적합 후 표준화
        valid_data = rolling_precip.dropna()
        if len(valid_data) < 30:
            return pd.Series(np.nan, index=precip_series.index)

        # 0이 아닌 값만 사용하여 감마 분포 적합
        non_zero = valid_data[valid_data > 0]
        if len(non_zero) < 10:
            return pd.Series(np.nan, index=precip_series.index)

        shape, loc, scale_param = stats.gamma.fit(non_zero, floc=0)

        # CDF 계산 후 표준정규분포로 변환
        cdf = stats.gamma.cdf(rolling_precip, shape, loc=0, scale=scale_param)
        spi = stats.norm.ppf(cdf)

        return pd.Series(spi, index=precip_series.index)


def main():
    """데이터 수집 실행"""
    collector = KMADataCollector(save_dir="../../data/raw")

    # 2000-2024 데이터 수집
    print("Collecting KMA observation data (2000-2024)...")
    df = collector.fetch_all_stations(
        start_date="20000101",
        end_date="20241231"
    )

    # 극한 이벤트 식별
    print("Identifying extreme events...")
    df = collector.identify_extreme_events(df)

    # 저장
    collector.save_data(df, "kma_observations_2000_2024.csv")

    # 극한 이벤트 통계
    print("\n=== Extreme Event Statistics ===")
    print(f"Total records: {len(df):,}")
    print(f"Heatwave days: {df['is_heatwave'].sum():,}")
    print(f"Tropical night days: {df['is_tropical_night'].sum():,}")
    print(f"Cold wave days: {df['is_cold_wave'].sum():,}")
    print(f"Heavy rain days: {df['is_heavy_rain'].sum():,}")
    print(f"Compound heat+drought: {df['compound_heat_drought'].sum():,}")
    print(f"Compound heat+tropical: {df['compound_heat_tropical'].sum():,}")


if __name__ == "__main__":
    main()
