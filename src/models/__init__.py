"""
AI Models Module
복합 극한기후 분석을 위한 AI 모델
"""

from .transformer_detector import (
    CompoundEventTransformer,
    TransformerConfig,
    CompoundEventLoss
)

# GNN은 torch_geometric이 필요하므로 선택적 import
try:
    from .gnn_spatial import (
        SpatioTemporalGNN,
        GNNConfig,
        SpatialGraphBuilder
    )
except ImportError:
    SpatioTemporalGNN = None
    GNNConfig = None
    SpatialGraphBuilder = None

from .impact_predictor import (
    MultiTaskImpactModel,
    HybridImpactPredictor,
    VulnerabilityScorer,
    ImpactPredictorConfig
)

__all__ = [
    'CompoundEventTransformer',
    'TransformerConfig',
    'CompoundEventLoss',
    'SpatioTemporalGNN',
    'GNNConfig',
    'SpatialGraphBuilder',
    'MultiTaskImpactModel',
    'HybridImpactPredictor',
    'VulnerabilityScorer',
    'ImpactPredictorConfig',
]
