"""
Multi-Task Impact Prediction Model
다중 태스크 학습 기반 사회경제적 영향 예측 모델

Tasks:
1. 경제적 피해액 예측 (재산피해)
2. 건강 영향 예측 (온열/한랭 질환)
3. 농업 피해 예측 (작물 피해)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class ImpactPredictorConfig:
    """영향 예측 모델 설정"""
    climate_features: int = 10      # 기후 변수 수
    socioeconomic_features: int = 8 # 사회경제 변수 수
    hidden_dims: List[int] = None   # 은닉층 차원
    dropout: float = 0.3
    n_tasks: int = 3                # 예측 태스크 수

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64, 32]


class FeatureExtractor(nn.Module):
    """공유 특성 추출기"""

    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TaskHead(nn.Module):
    """개별 태스크 예측 헤드"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        output_dim: int = 1,
        task_type: str = 'regression'
    ):
        super().__init__()

        self.task_type = task_type

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # 출력 활성화 함수
        if task_type == 'regression':
            self.activation = nn.ReLU()  # 피해액은 비음수
        elif task_type == 'classification':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.network(x))


class MultiTaskImpactModel(nn.Module):
    """
    다중 태스크 영향 예측 모델

    Input Features:
    - Climate: temp_max, temp_min, precip, humidity, duration, severity, ...
    - Socioeconomic: pop_density, elderly_ratio, farmland_ratio, ...

    Output Tasks:
    1. property_damage: 재산 피해액 (백만원)
    2. health_impact: 건강 영향 (환자 수)
    3. agriculture_damage: 농업 피해 (피해 면적 ha)
    """

    def __init__(self, config: ImpactPredictorConfig):
        super().__init__()
        self.config = config

        total_input = config.climate_features + config.socioeconomic_features

        # 공유 특성 추출기
        self.shared_extractor = FeatureExtractor(
            total_input, config.hidden_dims, config.dropout
        )

        shared_output_dim = config.hidden_dims[-1]

        # 태스크별 헤드
        self.task_heads = nn.ModuleDict({
            'property_damage': TaskHead(
                shared_output_dim, 32, 1, 'regression'
            ),
            'health_impact': TaskHead(
                shared_output_dim, 32, 1, 'regression'
            ),
            'agriculture_damage': TaskHead(
                shared_output_dim, 32, 1, 'regression'
            )
        })

        # 태스크 간 상호작용 (cross-stitch 유닛)
        self.task_interaction = nn.Parameter(
            torch.eye(config.n_tasks) * 0.9 + torch.ones(config.n_tasks, config.n_tasks) * 0.1
        )

    def forward(
        self,
        climate_features: torch.Tensor,
        socioeconomic_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            climate_features: (batch_size, climate_dim)
            socioeconomic_features: (batch_size, socioeconomic_dim)

        Returns:
            Dict of task predictions
        """
        # 특성 결합
        x = torch.cat([climate_features, socioeconomic_features], dim=-1)

        # 공유 특성 추출
        shared_features = self.shared_extractor(x)

        # 각 태스크 예측
        outputs = {}
        for task_name, task_head in self.task_heads.items():
            outputs[task_name] = task_head(shared_features)

        # 불확실성 추정 (MC Dropout)
        if self.training:
            outputs['shared_features'] = shared_features

        return outputs

    def predict_with_uncertainty(
        self,
        climate_features: torch.Tensor,
        socioeconomic_features: torch.Tensor,
        n_samples: int = 100
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        MC Dropout을 이용한 불확실성 추정

        Returns:
            Dict of (mean, std) for each task
        """
        self.train()  # Enable dropout

        predictions = {task: [] for task in self.task_heads.keys()}

        with torch.no_grad():
            for _ in range(n_samples):
                outputs = self.forward(climate_features, socioeconomic_features)
                for task in self.task_heads.keys():
                    predictions[task].append(outputs[task])

        results = {}
        for task, preds in predictions.items():
            preds = torch.stack(preds, dim=0)
            results[task] = (preds.mean(dim=0), preds.std(dim=0))

        self.eval()
        return results


class HybridImpactPredictor:
    """
    하이브리드 영향 예측 모델 (XGBoost + Neural Network)

    XGBoost: 테이블 데이터에 강건한 예측
    Neural Network: 복잡한 비선형 관계 학습
    Ensemble: 두 모델 결합으로 성능 향상
    """

    def __init__(
        self,
        n_climate_features: int = 10,
        n_socioeconomic_features: int = 8,
        xgb_params: Optional[Dict] = None,
        nn_config: Optional[ImpactPredictorConfig] = None
    ):
        self.n_features = n_climate_features + n_socioeconomic_features
        self.scaler = StandardScaler()

        # XGBoost 모델
        default_xgb_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'reg:squarederror',
            'tree_method': 'hist'
        }
        if xgb_params:
            default_xgb_params.update(xgb_params)

        self.xgb_models = {
            'property_damage': xgb.XGBRegressor(**default_xgb_params),
            'health_impact': xgb.XGBRegressor(**default_xgb_params),
            'agriculture_damage': xgb.XGBRegressor(**default_xgb_params)
        }

        # Neural Network 모델
        nn_config = nn_config or ImpactPredictorConfig(
            climate_features=n_climate_features,
            socioeconomic_features=n_socioeconomic_features
        )
        self.nn_model = MultiTaskImpactModel(nn_config)

        # 앙상블 가중치
        self.ensemble_weights = {'xgb': 0.6, 'nn': 0.4}

    def fit(
        self,
        X_climate: np.ndarray,
        X_socioeconomic: np.ndarray,
        y: Dict[str, np.ndarray],
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 0.001
    ):
        """모델 학습"""
        X = np.concatenate([X_climate, X_socioeconomic], axis=1)
        X_scaled = self.scaler.fit_transform(X)

        # XGBoost 학습
        print("Training XGBoost models...")
        for task, model in self.xgb_models.items():
            if task in y:
                model.fit(X_scaled, y[task])
                print(f"  {task}: Done")

        # Neural Network 학습
        print("\nTraining Neural Network...")
        X_climate_tensor = torch.tensor(self.scaler.transform(X)[:, :X_climate.shape[1]], dtype=torch.float32)
        X_socio_tensor = torch.tensor(self.scaler.transform(X)[:, X_climate.shape[1]:], dtype=torch.float32)

        optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            self.nn_model.train()
            optimizer.zero_grad()

            outputs = self.nn_model(X_climate_tensor, X_socio_tensor)

            total_loss = 0
            for task in outputs.keys():
                if task in y and task != 'shared_features':
                    target = torch.tensor(y[task], dtype=torch.float32).unsqueeze(-1)
                    total_loss += loss_fn(outputs[task], target)

            total_loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss.item():.4f}")

    def predict(
        self,
        X_climate: np.ndarray,
        X_socioeconomic: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """앙상블 예측"""
        X = np.concatenate([X_climate, X_socioeconomic], axis=1)
        X_scaled = self.scaler.transform(X)

        # XGBoost 예측
        xgb_preds = {}
        for task, model in self.xgb_models.items():
            xgb_preds[task] = model.predict(X_scaled)

        # Neural Network 예측
        self.nn_model.eval()
        with torch.no_grad():
            X_climate_tensor = torch.tensor(X_scaled[:, :X_climate.shape[1]], dtype=torch.float32)
            X_socio_tensor = torch.tensor(X_scaled[:, X_climate.shape[1]:], dtype=torch.float32)
            nn_outputs = self.nn_model(X_climate_tensor, X_socio_tensor)

        nn_preds = {}
        for task in self.xgb_models.keys():
            if task in nn_outputs:
                nn_preds[task] = nn_outputs[task].numpy().flatten()

        # 앙상블
        ensemble_preds = {}
        for task in self.xgb_models.keys():
            ensemble_preds[task] = (
                self.ensemble_weights['xgb'] * xgb_preds[task] +
                self.ensemble_weights['nn'] * nn_preds.get(task, xgb_preds[task])
            )

        return ensemble_preds

    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """XGBoost 특성 중요도 반환"""
        importance = {}
        for task, model in self.xgb_models.items():
            importance[task] = model.feature_importances_
        return importance


class VulnerabilityScorer:
    """
    취약성 점수 계산기

    Vulnerability = f(Exposure, Sensitivity, Adaptive Capacity)
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            'exposure': 0.35,
            'sensitivity': 0.35,
            'adaptive_capacity': 0.30
        }

    def calculate_exposure(
        self,
        compound_event_freq: float,
        severity_mean: float,
        duration_mean: float
    ) -> float:
        """노출도 계산"""
        # 정규화된 복합 이벤트 빈도
        freq_score = min(compound_event_freq / 10, 1.0)  # 10회를 최대로

        # 정규화된 강도
        severity_score = min(severity_mean / 5, 1.0)

        # 정규화된 지속기간
        duration_score = min(duration_mean / 7, 1.0)  # 7일을 최대로

        return 0.4 * freq_score + 0.4 * severity_score + 0.2 * duration_score

    def calculate_sensitivity(
        self,
        pop_density: float,
        elderly_ratio: float,
        farmland_ratio: float,
        impervious_ratio: float
    ) -> float:
        """민감도 계산"""
        # 인구밀도 (높을수록 민감)
        pop_score = min(pop_density / 10000, 1.0)

        # 고령인구 비율
        elderly_score = elderly_ratio

        # 농경지 비율
        farm_score = farmland_ratio

        # 불투수면 비율
        imperv_score = impervious_ratio

        return 0.3 * pop_score + 0.3 * elderly_score + 0.2 * farm_score + 0.2 * imperv_score

    def calculate_adaptive_capacity(
        self,
        medical_facilities: float,
        fiscal_independence: float,
        green_space_ratio: float
    ) -> float:
        """적응 역량 계산 (높을수록 좋음)"""
        # 의료시설 (1000명당)
        medical_score = min(medical_facilities / 5, 1.0)

        # 재정자립도
        fiscal_score = fiscal_independence

        # 녹지비율
        green_score = green_space_ratio

        return 0.4 * medical_score + 0.4 * fiscal_score + 0.2 * green_score

    def calculate_vulnerability(
        self,
        exposure: float,
        sensitivity: float,
        adaptive_capacity: float
    ) -> float:
        """
        종합 취약성 점수 계산

        Vulnerability = (Exposure * Sensitivity) / Adaptive Capacity
        -> 0-1 범위로 정규화
        """
        # 적응 역량이 0이면 최대 취약성
        if adaptive_capacity < 0.01:
            adaptive_capacity = 0.01

        raw_vulnerability = (exposure * sensitivity) / adaptive_capacity

        # 0-1 정규화 (sigmoid 변환)
        normalized = 1 / (1 + np.exp(-3 * (raw_vulnerability - 0.5)))

        return normalized

    def score_region(self, region_data: Dict) -> Dict[str, float]:
        """지역 취약성 종합 평가"""
        exposure = self.calculate_exposure(
            region_data.get('compound_event_freq', 0),
            region_data.get('severity_mean', 0),
            region_data.get('duration_mean', 0)
        )

        sensitivity = self.calculate_sensitivity(
            region_data.get('pop_density', 0),
            region_data.get('elderly_ratio', 0),
            region_data.get('farmland_ratio', 0),
            region_data.get('impervious_ratio', 0)
        )

        adaptive = self.calculate_adaptive_capacity(
            region_data.get('medical_facilities', 0),
            region_data.get('fiscal_independence', 0),
            region_data.get('green_space_ratio', 0)
        )

        vulnerability = self.calculate_vulnerability(exposure, sensitivity, adaptive)

        return {
            'exposure': exposure,
            'sensitivity': sensitivity,
            'adaptive_capacity': adaptive,
            'vulnerability': vulnerability,
            'risk_level': self._get_risk_level(vulnerability)
        }

    @staticmethod
    def _get_risk_level(vulnerability: float) -> str:
        """취약성 수준 분류"""
        if vulnerability < 0.2:
            return 'Very Low'
        elif vulnerability < 0.4:
            return 'Low'
        elif vulnerability < 0.6:
            return 'Medium'
        elif vulnerability < 0.8:
            return 'High'
        else:
            return 'Very High'


def main():
    """모델 테스트"""
    # 샘플 데이터 생성
    n_samples = 200
    n_climate = 10
    n_socio = 8

    np.random.seed(42)
    X_climate = np.random.randn(n_samples, n_climate)
    X_socio = np.random.randn(n_samples, n_socio)

    # 타겟 생성 (약간의 관계성 부여)
    y = {
        'property_damage': np.abs(X_climate[:, 0] * 100 + X_socio[:, 0] * 50 + np.random.randn(n_samples) * 10),
        'health_impact': np.abs(X_climate[:, 1] * 20 + X_socio[:, 1] * 10 + np.random.randn(n_samples) * 5),
        'agriculture_damage': np.abs(X_climate[:, 2] * 50 + X_socio[:, 2] * 30 + np.random.randn(n_samples) * 8)
    }

    # 하이브리드 모델 학습
    print("=" * 50)
    print("Training Hybrid Impact Predictor")
    print("=" * 50)

    model = HybridImpactPredictor(n_climate, n_socio)
    model.fit(X_climate, X_socio, y, epochs=50)

    # 예측
    preds = model.predict(X_climate[:10], X_socio[:10])
    print("\nPredictions (first 10 samples):")
    for task, pred in preds.items():
        print(f"  {task}: {pred[:5].round(2)}")

    # 취약성 점수 테스트
    print("\n" + "=" * 50)
    print("Vulnerability Scoring")
    print("=" * 50)

    scorer = VulnerabilityScorer()

    # 샘플 지역 데이터
    regions = {
        '서울 강남구': {
            'compound_event_freq': 5, 'severity_mean': 2.5, 'duration_mean': 3,
            'pop_density': 15000, 'elderly_ratio': 0.15, 'farmland_ratio': 0.01, 'impervious_ratio': 0.7,
            'medical_facilities': 6, 'fiscal_independence': 0.6, 'green_space_ratio': 0.15
        },
        '전남 순천시': {
            'compound_event_freq': 3, 'severity_mean': 2.0, 'duration_mean': 4,
            'pop_density': 500, 'elderly_ratio': 0.28, 'farmland_ratio': 0.35, 'impervious_ratio': 0.2,
            'medical_facilities': 2, 'fiscal_independence': 0.25, 'green_space_ratio': 0.4
        }
    }

    for region, data in regions.items():
        scores = scorer.score_region(data)
        print(f"\n{region}:")
        for k, v in scores.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.3f}")
            else:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
