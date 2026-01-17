"""
ERA5 Reanalysis Data Collector
Copernicus Climate Data Store에서 ERA5 재분석 데이터 수집

Data source: https://cds.climate.copernicus.eu/
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any

# Optional imports
try:
    import cdsapi
    HAS_CDSAPI = True
except ImportError:
    HAS_CDSAPI = False
    cdsapi = None

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False
    xr = None


class ERA5Collector:
    """ERA5 재분석 데이터 수집기"""

    # 한반도 영역 (33°N-43°N, 124°E-132°E)
    KOREA_BBOX = {
        'north': 43,
        'south': 33,
        'west': 124,
        'east': 132
    }

    # 수집 변수
    SURFACE_VARIABLES = [
        '2m_temperature',
        '2m_dewpoint_temperature',
        'total_precipitation',
        'surface_pressure',
        '10m_u_component_of_wind',
        '10m_v_component_of_wind',
        'mean_sea_level_pressure',
        'soil_temperature_level_1',
        'volumetric_soil_water_layer_1',
    ]

    PRESSURE_VARIABLES = [
        'geopotential',
        'temperature',
        'u_component_of_wind',
        'v_component_of_wind',
        'relative_humidity',
    ]

    PRESSURE_LEVELS = ['500', '700', '850', '925']

    def __init__(self, save_dir: str = "data/external"):
        """
        Args:
            save_dir: 저장 디렉토리
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # CDS API 클라이언트 초기화
        self.client = None
        if HAS_CDSAPI:
            try:
                self.client = cdsapi.Client()
            except Exception as e:
                print(f"Warning: CDS API client initialization failed: {e}")

    def generate_sample_data(
        self,
        start_date: str,
        end_date: str,
        resolution: float = 0.25
    ) -> Any:
        """
        샘플 ERA5 데이터 생성 (API 없이 테스트용)

        Args:
            start_date: 시작일 (YYYY-MM-DD)
            end_date: 종료일 (YYYY-MM-DD)
            resolution: 공간 해상도 (도)

        Returns:
            샘플 xarray Dataset 또는 dict
        """
        times = pd.date_range(start=start_date, end=end_date, freq='6H')
        lats = np.arange(
            self.KOREA_BBOX['south'],
            self.KOREA_BBOX['north'] + resolution,
            resolution
        )
        lons = np.arange(
            self.KOREA_BBOX['west'],
            self.KOREA_BBOX['east'] + resolution,
            resolution
        )

        # 계절성 시뮬레이션
        n_times = len(times)
        n_lats = len(lats)
        n_lons = len(lons)

        # 위도에 따른 기온 변화
        lat_effect = (lats - 38)[:, np.newaxis] * -0.5

        # 시간에 따른 계절 변화
        day_of_year = times.dayofyear
        seasonal = 15 * np.sin(2 * np.pi * (day_of_year - 100) / 365)

        # 3D 기온 필드 생성
        t2m = np.zeros((n_times, n_lats, n_lons))
        for i, s in enumerate(seasonal):
            t2m[i] = 288 + s + lat_effect + np.random.normal(0, 2, (n_lats, n_lons))

        # 강수량 (지수 분포)
        precip = np.maximum(0, np.random.exponential(0.0001, (n_times, n_lats, n_lons)))

        if HAS_XARRAY:
            ds = xr.Dataset(
                {
                    't2m': (['time', 'latitude', 'longitude'], t2m),
                    'tp': (['time', 'latitude', 'longitude'], precip),
                },
                coords={
                    'time': times,
                    'latitude': lats,
                    'longitude': lons,
                }
            )
            ds.attrs['title'] = 'ERA5 Sample Data for Korea'
            ds.attrs['source'] = 'Synthetic data for testing'
            return ds
        else:
            # Return as dict if xarray not available
            return {
                't2m': t2m,
                'tp': precip,
                'time': times,
                'latitude': lats,
                'longitude': lons
            }


def main():
    """ERA5 데이터 수집 테스트"""
    collector = ERA5Collector(save_dir="../../data/external")

    # 샘플 데이터 생성
    print("Generating sample ERA5 data...")
    data = collector.generate_sample_data(
        start_date="2020-01-01",
        end_date="2020-12-31"
    )

    if HAS_XARRAY and hasattr(data, 'to_netcdf'):
        output_file = collector.save_dir / "era5_sample_2020.nc"
        data.to_netcdf(output_file)
        print(f"Saved sample data to {output_file}")
        print(f"\nTemperature range: {float(data['t2m'].min()):.1f}K - {float(data['t2m'].max()):.1f}K")
    else:
        print("Sample data generated as dictionary (xarray not available)")
        print(f"Temperature range: {data['t2m'].min():.1f}K - {data['t2m'].max():.1f}K")


if __name__ == "__main__":
    main()
