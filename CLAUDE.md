# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

한반도 복합 극한기후 현상(Compound Extreme Events)의 사회경제적 취약성을 분석하는 AI 기반 연구 프로젝트. AI Co-Scientist Challenge Korea 2026 Track 1 출품작.

## Commands

```bash
# 환경 설정
conda create -n climate python=3.10
conda activate climate
pip install -r requirements.txt

# 전체 분석 파이프라인 실행
python main_analysis.py

# PDF 보고서 생성
python generate_detailed_pdf.py

# 취약성 지도 생성
python generate_korea_vulnerability_map.py
python generate_professional_map.py
```

## Architecture

### 분석 파이프라인 (main_analysis.py)
`CompoundClimateAnalysisPipeline` 클래스가 6단계 워크플로우를 순차 실행:
1. **데이터 수집** - KMA 기상, 재해, 건강, 농업 데이터
2. **복합 이벤트 탐지** - 개별/동시/순차 극한기후 탐지
3. **AI 모델 학습** - Transformer, 영향 예측 모델
4. **취약성 분석** - 지역별 취약성 점수 산출
5. **시각화** - 취약성 지도, 성능 차트
6. **보고서 생성**

### AI 모델 (src/models/)
- **CompoundEventTransformer** - 시계열 복합 이벤트 탐지 (Multi-head Attention)
- **SpatioTemporalGNN** - 공간적 이벤트 전파 분석 (torch_geometric 필요, 선택적)
- **HybridImpactPredictor** - XGBoost+NN 앙상블, 다중 태스크 (재산/건강/농업 피해)
- **VulnerabilityScorer** - Exposure/Sensitivity/Adaptive Capacity 기반 취약성 지수

### 데이터 수집 (src/data_collection/)
- `kma_collector.py` - 기상청 기상자료 수집, 극한기후 임계값 적용
- `socioeconomic_collector.py` - 재해/건강/농업/지역취약성 데이터
- `era5_collector.py` - Copernicus ERA5 재분석 데이터 (cdsapi)

### 복합 이벤트 유형 (src/preprocessing/compound_event_detector.py)
| Type | 구성 | 패턴 |
|------|------|------|
| A | 폭염 + 가뭄 | 동시 |
| B | 폭염 + 열대야 | 동시 |
| C | 한파 + 대설 | 동시 |
| D | 폭우 → 폭염 | 순차 (7일) |
| E | 가뭄 → 폭우 | 순차 (돌발홍수) |

## Key Dependencies

- **PyTorch + torch-geometric** - 딥러닝/GNN
- **xarray + netCDF4** - 기후 데이터
- **geopandas + folium** - 지리 시각화
- **XGBoost + LightGBM** - 앙상블 학습

## Data

- `data/korea_provinces.geojson` - 한국 시도 경계 (시각화용)
- `data/raw/` - 원본 데이터 (기상, 재해, 건강, 농업)
- `data/processed/` - 전처리된 복합 이벤트 데이터
- `results/` - figures/, tables/, models/ 출력

## Submission Files

- `paper/research_report.tex` - NeurIPS 스타일 영문 연구보고서
- `submission/AI_활용보고서.md` - AI 도구 활용 내역
- `submission/활용데이터목록.md` - 데이터 소스 목록
