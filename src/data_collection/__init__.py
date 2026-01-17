"""
Data Collection Module
기상 및 사회경제 데이터 수집 모듈
"""

from .kma_collector import KMADataCollector
from .socioeconomic_collector import (
    DisasterDataCollector,
    HealthDataCollector,
    AgricultureDataCollector,
    RegionalVulnerabilityData
)

# ERA5Collector는 선택적 import
try:
    from .era5_collector import ERA5Collector
except ImportError:
    ERA5Collector = None

__all__ = [
    'KMADataCollector',
    'ERA5Collector',
    'DisasterDataCollector',
    'HealthDataCollector',
    'AgricultureDataCollector',
    'RegionalVulnerabilityData',
]
