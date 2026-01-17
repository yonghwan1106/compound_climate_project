"""
Compound Extreme Event Detection Algorithm
복합 극한기후 현상 탐지 알고리즘

Types of Compound Events:
- Type A: Concurrent (동시 발생) - 폭염+가뭄, 폭염+열대야
- Type B: Sequential (순차 발생) - 폭우→폭염, 가뭄→폭우
- Type C: Spatially Compound (공간 복합) - 지역 연쇄 폭염
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import genextreme, norm
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ExtremeType(Enum):
    """극한 기상 유형"""
    HEATWAVE = "heatwave"           # 폭염
    TROPICAL_NIGHT = "tropical_night"  # 열대야
    COLD_WAVE = "cold_wave"         # 한파
    HEAVY_RAIN = "heavy_rain"       # 폭우
    DROUGHT = "drought"             # 가뭄
    HEAVY_SNOW = "heavy_snow"       # 대설


@dataclass
class ExtremeThreshold:
    """극한 이벤트 임계값 정의"""
    heatwave_temp: float = 33.0        # 일최고기온 >= 33°C
    tropical_night_temp: float = 25.0   # 일최저기온 >= 25°C
    cold_wave_temp: float = -12.0       # 일최저기온 <= -12°C
    heavy_rain_mm: float = 80.0         # 일강수량 >= 80mm
    drought_spi: float = -1.5           # SPI <= -1.5
    heavy_snow_mm: float = 20.0         # 일강설량 >= 20cm


@dataclass
class CompoundEvent:
    """복합 극한기후 이벤트"""
    event_id: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    event_type: str
    components: List[ExtremeType]
    severity: float
    affected_stations: List[str]
    peak_values: Dict[str, float]
    duration_days: int
    is_concurrent: bool


class CompoundEventDetector:
    """복합 극한기후 탐지기"""

    def __init__(self, thresholds: Optional[ExtremeThreshold] = None):
        self.thresholds = thresholds or ExtremeThreshold()
        self.events: List[CompoundEvent] = []

    def detect_individual_extremes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        개별 극한 이벤트 탐지

        Required columns:
        - date, station_id, temp_max, temp_min, temp_avg, precip, humidity
        """
        df = df.copy()

        # 개별 극한 이벤트 플래그
        df['extreme_heatwave'] = df['temp_max'] >= self.thresholds.heatwave_temp
        df['extreme_tropical'] = df['temp_min'] >= self.thresholds.tropical_night_temp
        df['extreme_coldwave'] = df['temp_min'] <= self.thresholds.cold_wave_temp
        df['extreme_heavyrain'] = df['precip'] >= self.thresholds.heavy_rain_mm

        # 가뭄 판단 (30일 누적 강수량 기반 간이 지표)
        df['precip_30d'] = df.groupby('station_id')['precip'].transform(
            lambda x: x.rolling(window=30, min_periods=1).sum()
        )
        # 평균 대비 50% 미만이면 가뭄으로 간주
        df['extreme_drought'] = df['precip_30d'] < df.groupby('station_id')['precip_30d'].transform('mean') * 0.5

        return df

    def detect_concurrent_compounds(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        동시 발생 복합 이벤트 탐지

        Types:
        - compound_heat_drought: 폭염 + 가뭄 (동시)
        - compound_heat_tropical: 폭염 + 열대야 (동시)
        - compound_cold_snow: 한파 + 대설 (동시)
        """
        df = df.copy()

        # Type A: 폭염 + 가뭄
        df['compound_heat_drought'] = (
            df['extreme_heatwave'] &
            df['extreme_drought'] &
            (df['humidity'] < 50)  # 저습도 조건
        )

        # Type B: 폭염 + 열대야 (주야간 지속)
        df['compound_heat_tropical'] = (
            df['extreme_heatwave'] &
            df['extreme_tropical']
        )

        # Type C: 한파 + 대설 (추정)
        df['compound_cold_snow'] = (
            df['extreme_coldwave'] &
            (df['precip'] >= self.thresholds.heavy_snow_mm) &
            (df['temp_avg'] <= 0)
        )

        # 복합 이벤트 강도 계산
        df['compound_severity'] = self._calculate_severity(df)

        return df

    def detect_sequential_compounds(
        self,
        df: pd.DataFrame,
        lag_days: int = 7
    ) -> pd.DataFrame:
        """
        순차 발생 복합 이벤트 탐지

        Types:
        - sequential_rain_heat: 폭우 후 폭염 (7일 이내)
        - sequential_drought_rain: 가뭄 후 폭우 (돌발홍수 위험)
        """
        df = df.copy()
        df = df.sort_values(['station_id', 'date'])

        # 이전 N일 내 이벤트 발생 여부
        for col in ['extreme_heavyrain', 'extreme_heatwave', 'extreme_drought']:
            df[f'{col}_prev_{lag_days}d'] = df.groupby('station_id')[col].transform(
                lambda x: x.rolling(window=lag_days, min_periods=1).max().shift(1)
            )

        # Sequential: 폭우 → 폭염
        df['sequential_rain_heat'] = (
            df['extreme_heatwave'] &
            (df[f'extreme_heavyrain_prev_{lag_days}d'] == 1)
        )

        # Sequential: 가뭄 → 폭우 (돌발홍수)
        df['sequential_drought_rain'] = (
            df['extreme_heavyrain'] &
            (df[f'extreme_drought_prev_{lag_days}d'] == 1)
        )

        return df

    def _calculate_severity(self, df: pd.DataFrame) -> pd.Series:
        """
        복합 이벤트 강도 계산

        Severity = weighted sum of normalized exceedances
        """
        severity = pd.Series(0.0, index=df.index)

        # 기온 초과 정도
        if 'temp_max' in df.columns:
            temp_excess = np.maximum(0, df['temp_max'] - self.thresholds.heatwave_temp)
            severity += temp_excess / 5  # 5°C 당 1점

        # 강수량 초과 정도
        if 'precip' in df.columns:
            rain_excess = np.maximum(0, df['precip'] - self.thresholds.heavy_rain_mm)
            severity += rain_excess / 50  # 50mm 당 1점

        # 열대야 지속
        if 'extreme_tropical' in df.columns:
            severity += df['extreme_tropical'].astype(float) * 0.5

        return severity

    def identify_events(self, df: pd.DataFrame) -> List[CompoundEvent]:
        """
        연속된 복합 이벤트 그룹화

        Returns:
            List of CompoundEvent objects
        """
        df = df.copy()
        df = df.sort_values(['station_id', 'date'])

        # 복합 이벤트 열
        compound_cols = [
            'compound_heat_drought',
            'compound_heat_tropical',
            'compound_cold_snow',
            'sequential_rain_heat',
            'sequential_drought_rain'
        ]

        events = []
        event_id = 0

        for station_id in df['station_id'].unique():
            station_df = df[df['station_id'] == station_id].copy()

            for col in compound_cols:
                if col not in station_df.columns:
                    continue

                # 연속 이벤트 그룹화
                station_df['event_group'] = (
                    station_df[col].astype(int).diff().ne(0).cumsum()
                )

                event_groups = station_df[station_df[col]].groupby('event_group')

                for group_id, group_df in event_groups:
                    event_id += 1

                    event = CompoundEvent(
                        event_id=event_id,
                        start_date=group_df['date'].min(),
                        end_date=group_df['date'].max(),
                        event_type=col,
                        components=self._get_components(col),
                        severity=group_df['compound_severity'].mean() if 'compound_severity' in group_df else 0,
                        affected_stations=[str(station_id)],
                        peak_values={
                            'temp_max': group_df['temp_max'].max(),
                            'temp_min': group_df['temp_min'].min(),
                            'precip_max': group_df['precip'].max(),
                        },
                        duration_days=len(group_df),
                        is_concurrent='sequential' not in col
                    )
                    events.append(event)

        self.events = events
        return events

    def _get_components(self, compound_type: str) -> List[ExtremeType]:
        """복합 이벤트 구성요소 반환"""
        mapping = {
            'compound_heat_drought': [ExtremeType.HEATWAVE, ExtremeType.DROUGHT],
            'compound_heat_tropical': [ExtremeType.HEATWAVE, ExtremeType.TROPICAL_NIGHT],
            'compound_cold_snow': [ExtremeType.COLD_WAVE, ExtremeType.HEAVY_SNOW],
            'sequential_rain_heat': [ExtremeType.HEAVY_RAIN, ExtremeType.HEATWAVE],
            'sequential_drought_rain': [ExtremeType.DROUGHT, ExtremeType.HEAVY_RAIN],
        }
        return mapping.get(compound_type, [])

    def calculate_return_period(
        self,
        df: pd.DataFrame,
        variable: str,
        threshold: float
    ) -> float:
        """
        재현기간 계산 (GEV 분포 기반)

        Args:
            df: 연간 최대값 시계열
            variable: 변수명
            threshold: 임계값

        Returns:
            재현기간 (년)
        """
        annual_max = df.groupby(df['date'].dt.year)[variable].max()

        if len(annual_max) < 10:
            return np.nan

        # GEV 분포 적합
        params = genextreme.fit(annual_max)
        shape, loc, scale = params

        # 초과 확률
        exceedance_prob = 1 - genextreme.cdf(threshold, shape, loc, scale)

        if exceedance_prob > 0:
            return 1 / exceedance_prob
        return np.inf

    def get_event_statistics(self) -> pd.DataFrame:
        """이벤트 통계 요약"""
        if not self.events:
            return pd.DataFrame()

        records = []
        for event in self.events:
            records.append({
                'event_id': event.event_id,
                'event_type': event.event_type,
                'start_date': event.start_date,
                'end_date': event.end_date,
                'duration_days': event.duration_days,
                'severity': event.severity,
                'is_concurrent': event.is_concurrent,
                'n_components': len(event.components),
                'peak_temp': event.peak_values.get('temp_max', np.nan),
                'peak_precip': event.peak_values.get('precip_max', np.nan),
            })

        return pd.DataFrame(records)


class CopulaAnalyzer:
    """Copula 기반 동시발생 확률 분석"""

    def __init__(self):
        self.marginals = {}

    def fit_marginals(self, df: pd.DataFrame, variables: List[str]):
        """주변 분포 적합"""
        for var in variables:
            data = df[var].dropna()
            # 정규분포로 근사 (실제로는 적절한 분포 선택 필요)
            mu, sigma = norm.fit(data)
            self.marginals[var] = {'type': 'normal', 'mu': mu, 'sigma': sigma}

    def compute_joint_probability(
        self,
        df: pd.DataFrame,
        var1: str,
        var2: str,
        threshold1: float,
        threshold2: float
    ) -> float:
        """
        두 변수의 동시 초과 확률 계산

        P(X1 > threshold1 AND X2 > threshold2)
        """
        # 경험적 동시 초과 확률
        joint_exceed = (
            (df[var1] > threshold1) & (df[var2] > threshold2)
        ).mean()

        return joint_exceed


def main():
    """복합 이벤트 탐지 테스트"""
    from data_collection.kma_collector import KMADataCollector

    # 샘플 데이터 생성
    collector = KMADataCollector()
    df = collector.fetch_all_stations("20200101", "20231231")
    df = collector.identify_extreme_events(df)

    # 복합 이벤트 탐지
    detector = CompoundEventDetector()

    print("Detecting individual extremes...")
    df = detector.detect_individual_extremes(df)

    print("Detecting concurrent compounds...")
    df = detector.detect_concurrent_compounds(df)

    print("Detecting sequential compounds...")
    df = detector.detect_sequential_compounds(df)

    print("Identifying events...")
    events = detector.identify_events(df)

    # 통계
    stats_df = detector.get_event_statistics()
    print(f"\n=== Compound Event Statistics ===")
    print(f"Total events detected: {len(events)}")

    if len(stats_df) > 0:
        print("\nEvents by type:")
        print(stats_df.groupby('event_type').agg({
            'event_id': 'count',
            'duration_days': 'mean',
            'severity': 'mean'
        }).round(2))


if __name__ == "__main__":
    main()
