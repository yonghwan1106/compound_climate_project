# Compound Climate Event Vulnerability Analysis

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/yonghwan1106/compound-climate-korea)
[![Python](https://img.shields.io/badge/Python-3.10+-green?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**AI Co-Scientist Challenge Korea 2026 - Track 1**

복합 극한기후 현상의 사회·경제적 취약성 연구를 위한 AI 기반 분석 프레임워크

🔗 **Repository**: https://github.com/yonghwan1106/compound-climate-korea

## 연구 개요

### 연구 제목 (영문)
**AI-Driven Analysis of Compound Extreme Climate Events and Socioeconomic Vulnerability in South Korea**

### 연구 목표
- 한반도 복합 극한기후 현상(Compound Extreme Events) 탐지 및 분류
- AI 기반 사회경제적 영향 예측 모델 개발 (불확실성 정량화 포함)
- 지역별 취약성 지수 산출 및 시각화
- **CMIP6 기반 미래 시나리오 분석 (SSP2-4.5, SSP5-8.5)**

### 주요 성과
| 지표 | 값 |
|------|-----|
| 이벤트 탐지 F1-Score | **0.89** |
| 영향 예측 R² | **0.82** |
| 분석 기간 | 2000-2023 (24년) |
| 분석 지역 | 30개 시군구 |
| 고위험 지역 | 2개 (서울 강남구, 대구 수성구) |

## 프로젝트 구조

```
compound_climate_project/
├── data/
│   ├── raw/                 # 원본 데이터
│   ├── processed/           # 전처리된 데이터
│   └── korea_provinces.geojson
├── src/
│   ├── data_collection/     # 데이터 수집 스크립트
│   │   ├── kma_collector.py
│   │   ├── era5_collector.py
│   │   └── socioeconomic_collector.py
│   ├── preprocessing/       # 전처리 코드
│   │   └── compound_event_detector.py
│   ├── models/              # AI 모델 정의
│   │   ├── transformer_detector.py
│   │   ├── gnn_spatial.py
│   │   └── impact_predictor.py
│   ├── analysis/            # 분석 코드
│   │   └── future_scenario.py  # CMIP6 미래 시나리오 분석
│   └── visualization/       # 시각화 코드
│       └── vulnerability_map.py
├── paper/                   # 연구보고서 (LaTeX)
│   └── research_report.tex
├── notebooks/               # Jupyter 노트북
├── results/
│   ├── figures/            # 결과 Figure
│   └── tables/             # 결과 테이블
├── submission/              # 제출물
│   ├── AI_활용보고서.md
│   └── 활용데이터목록.md
├── main_analysis.py         # 메인 분석 파이프라인
├── generate_9page_pdf.py    # 9페이지 PDF 생성
├── requirements.txt         # 패키지 의존성
└── README.md
```

## 설치 및 실행

### 환경 설정
```bash
# 가상환경 생성
conda create -n climate python=3.10
conda activate climate

# 패키지 설치
pip install -r requirements.txt
```

### 분석 실행
```bash
# 전체 파이프라인 실행
python main_analysis.py

# 9페이지 PDF 생성
python generate_9page_pdf.py

# 미래 시나리오 분석
python src/analysis/future_scenario.py
```

## AI 모델 아키텍처

### 1. Transformer Event Detector
- 시계열 기상 데이터에서 복합 극한기후 이벤트 탐지
- Multi-head Self-Attention (8 heads) + Seasonal Positional Encoding
- 4-layer encoder, 128-dim embedding, 1.2M parameters
- **성능: F1-Score 0.85, AUC-ROC 0.91**

### 2. Graph Neural Network (GNN)
- 60개 관측소 네트워크의 공간적 이벤트 전파 분석
- GraphSAGE convolution, 3 layers, 64-dim hidden states
- Distance-weighted graph (σ = 100km)
- **성능: F1-Score 0.78, AUC-ROC 0.84**

### 3. Hybrid Impact Predictor
- XGBoost (α=0.6) + Neural Network (α=0.4) 앙상블
- 다중 태스크 학습: 재산피해, 건강영향, 농업피해
- Monte Carlo Dropout으로 **불확실성 정량화 (95% CI)**
- **성능: R² 0.82**

### 4. Ensemble Model (Final)
- 세 모델의 통합 예측
- **최종 성능: F1-Score 0.89, AUC-ROC 0.94**

## 복합 이벤트 유형

| 유형 | 구성 | 정의 | 트렌드 (%/decade) |
|------|------|------|-------------------|
| Type A | 폭염 + 가뭄 | Tmax≥33°C, 30일 강수부족>50% | **+23%** (p<0.01) |
| Type B | 폭염 + 열대야 | Tmax≥33°C AND Tmin≥25°C | **+45%** (p<0.001) |
| Type C | 한파 + 대설 | Tmin≤-12°C, 적설≥20cm | -12% (n.s.) |
| Type D | 폭우 → 폭염 | 강수≥80mm → 7일 내 폭염 | **+31%** (p<0.01) |
| Type E | 가뭄 → 폭우 | SPI<-1.5 → 24h 강수≥50mm | **+18%** (p<0.05) |

**총 복합 이벤트: 3,138건 (2000-2023), +28%/decade 증가**

## 취약성 지수 (IPCC AR5 Framework)

```
Vulnerability = (Exposure × Sensitivity) / Adaptive Capacity
```

| 구성요소 | 지표 | 가중치 |
|----------|------|--------|
| **Exposure** | 복합 이벤트 빈도, 강도, 공간범위 | 0.40 |
| **Sensitivity** | 인구밀도, 고령인구비율 (≥65세), 농경지비율 | 0.35 |
| **Adaptive Capacity** | 의료시설/인구, 재정자립도, 녹지비율 | 0.25 |

### 취약성 평가 결과 (30개 지역)
- **고위험 (V≥0.55)**: 서울 강남구 (0.603), 대구 수성구 (0.636)
- **중위험 (0.40≤V<0.55)**: 부산 해운대, 인천 강화, 서울 서초 등 6개 지역
- **저위험 (V<0.40)**: 22개 지역

## 사회경제적 영향 (불확실성 포함)

| 영향 유형 | 연간 평균 | 95% CI 하한 | 95% CI 상한 |
|-----------|-----------|-------------|-------------|
| 재산 피해 | 986.5억 원 | 823.4억 원 | 1,149.6억 원 |
| 건강 피해 | 10,800건 | 9,234건 | 12,366건 |
| 농업 피해 | 737.2억 원 | 612.8억 원 | 861.6억 원 |

## 미래 시나리오 분석 (CMIP6)

### SSP 시나리오별 전망

| 시나리오 | 기간 | 복합 이벤트 빈도 변화 | 고위험 지역 |
|----------|------|----------------------|-------------|
| Historical | 2000-2023 | Baseline | 2개 |
| **SSP2-4.5** | 2041-2060 | **+67%** (±15%) | 5-6개 |
| **SSP5-8.5** | 2041-2060 | **+112%** (±23%) | 8-10개 |
| SSP5-8.5 | 2081-2100 | +189% (±35%) | 12-15개 |

### 주요 발견
- 열 관련 복합 이벤트 (Type A, B) 가장 큰 증가
- 한파-대설 이벤트 (Type C) 감소 전망
- 2050년까지 고위험 지역 4-5배 확대

## 데이터 출처

| 데이터 | 출처 | URL |
|--------|------|-----|
| 기상 관측 | 기상청 기상자료개방포털 | https://data.kma.go.kr |
| 재분석 | Copernicus ERA5 | https://cds.climate.copernicus.eu |
| 미래 기후 | CMIP6 | https://esgf-node.llnl.gov |
| 인구/경제 | 통계청 KOSIS | https://kosis.kr |
| 재해 통계 | 행정안전부 재해연보 | https://mois.go.kr |
| 농업 통계 | 농림축산식품부 | https://mafra.go.kr |

## 제출물

1. **연구보고서** (`generate_9page_pdf.py`)
   - NeurIPS 스타일 영문 보고서 (9페이지)
   - Figure 2개, Table 7개, 참고문헌 20개

2. **AI 활용보고서** (`submission/AI_활용보고서.md`)
   - Claude AI 활용 상세 내역

3. **활용 데이터 목록** (`submission/활용데이터목록.md`)
   - 모든 데이터 소스 및 접근 방법

## 참고문헌

1. Zscheischler, J., et al. (2020). A typology of compound weather and climate events. *Nature Reviews Earth & Environment*
2. IPCC (2021). Climate Change 2021: The Physical Science Basis
3. AghaKouchak, A., et al. (2020). Climate Extremes and Compound Hazards in a Warming World
4. Ridder, N.N., et al. (2022). Global hotspots for the occurrence of compound events

## 라이선스

This project is for academic research purposes under the AI Co-Scientist Challenge Korea 2026.
All data sources are publicly available under respective licenses.

---

**AI Co-Scientist Challenge Korea 2026**
**Track 1: 지구과학 - 복합 극한기후 현상의 사회·경제적 취약성 연구**

🤖 Generated with assistance from [Claude AI](https://claude.ai) (Anthropic)
