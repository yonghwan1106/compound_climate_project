"""
Graph Neural Network for Spatial-Temporal Climate Analysis
공간-시간 기상 그래프 분석을 위한 GNN 모델

Architecture:
- Spatial Graph: 관측소 네트워크 (거리 기반 연결)
- Temporal Module: GRU/Transformer for time series
- Message Passing: GraphSAGE / GAT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class GNNConfig:
    """GNN 모델 설정"""
    node_features: int = 8          # 노드 특성 수
    hidden_dim: int = 64            # 은닉 차원
    n_gnn_layers: int = 3           # GNN 레이어 수
    n_temporal_layers: int = 2      # 시계열 레이어 수
    output_dim: int = 1             # 출력 차원 (severity)
    dropout: float = 0.2
    gnn_type: str = 'sage'          # 'sage' or 'gat'
    n_attention_heads: int = 4      # GAT 어텐션 헤드


class SpatialGraphBuilder:
    """관측소 네트워크 그래프 구축"""

    def __init__(self, distance_threshold: float = 100.0):
        """
        Args:
            distance_threshold: 연결 임계 거리 (km)
        """
        self.distance_threshold = distance_threshold

    @staticmethod
    def haversine_distance(
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """두 지점 간 거리 계산 (km)"""
        R = 6371  # 지구 반지름 (km)

        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        return R * c

    def build_graph(
        self,
        station_coords: Dict[str, Tuple[float, float]],
        node_features: np.ndarray
    ) -> Data:
        """
        관측소 그래프 구축

        Args:
            station_coords: {station_id: (lat, lon)}
            node_features: (n_nodes, n_features)

        Returns:
            PyG Data object
        """
        stations = list(station_coords.keys())
        n_nodes = len(stations)

        # Edge 구축 (거리 기반)
        edge_index = []
        edge_attr = []

        for i, s1 in enumerate(stations):
            for j, s2 in enumerate(stations):
                if i != j:
                    lat1, lon1 = station_coords[s1]
                    lat2, lon2 = station_coords[s2]
                    dist = self.haversine_distance(lat1, lon1, lat2, lon2)

                    if dist <= self.distance_threshold:
                        edge_index.append([i, j])
                        # 거리 기반 가중치 (가까울수록 높음)
                        edge_attr.append(1 / (1 + dist / 50))

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(-1)
        x = torch.tensor(node_features, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class TemporalModule(nn.Module):
    """시계열 처리 모듈 (GRU 기반)"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_dim, hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=True
        )

        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, seq_len, input_dim)
            hidden: initial hidden state

        Returns:
            output: (batch_size, seq_len, hidden_dim)
            hidden: final hidden state
        """
        output, hidden = self.gru(x, hidden)
        output = self.output_proj(output)
        return output, hidden


class SpatialGNNLayer(nn.Module):
    """공간 GNN 레이어"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        gnn_type: str = 'sage',
        n_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        if gnn_type == 'sage':
            self.conv = SAGEConv(in_channels, out_channels)
        elif gnn_type == 'gat':
            self.conv = GATConv(
                in_channels, out_channels // n_heads,
                heads=n_heads, dropout=dropout
            )
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")

        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (n_nodes, in_channels)
            edge_index: (2, n_edges)
            edge_attr: (n_edges, edge_dim)
        """
        h = self.conv(x, edge_index)
        h = self.norm(h)
        h = F.relu(h)
        h = self.dropout(h)
        return h


class SpatioTemporalGNN(nn.Module):
    """
    시공간 복합 GNN 모델

    Pipeline:
    1. Temporal Encoding: 각 노드의 시계열 처리
    2. Spatial Message Passing: 노드 간 정보 교환
    3. Global Aggregation: 전체 네트워크 상태 집계
    4. Prediction: 이벤트 강도/영향 예측
    """

    def __init__(self, config: GNNConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(config.node_features, config.hidden_dim)

        # Temporal module
        self.temporal = TemporalModule(
            config.hidden_dim,
            config.hidden_dim,
            config.n_temporal_layers,
            config.dropout
        )

        # Spatial GNN layers
        self.gnn_layers = nn.ModuleList([
            SpatialGNNLayer(
                config.hidden_dim if i == 0 else config.hidden_dim,
                config.hidden_dim,
                config.gnn_type,
                config.n_attention_heads,
                config.dropout
            )
            for i in range(config.n_gnn_layers)
        ])

        # Global aggregation
        self.global_pool = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim)
        )

        # Output heads
        self.severity_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.ReLU()  # 강도는 비음수
        )

        self.impact_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 3)  # 인명, 재산, 농업 피해
        )

        self.spatial_extent_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()  # 0-1 범위의 공간 영향 비율
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (n_nodes * seq_len, node_features) or (n_nodes, seq_len, node_features)
            edge_index: (2, n_edges)
            batch: (n_nodes,) - batch assignment
            edge_attr: (n_edges, edge_dim)

        Returns:
            Dict with predictions
        """
        # Handle different input shapes
        if x.dim() == 2:
            # Assume flattened (n_nodes, features)
            h = self.input_proj(x)
        else:
            # (n_nodes, seq_len, features) -> process temporal first
            batch_size, seq_len, _ = x.shape
            x_flat = x.view(-1, x.size(-1))
            h = self.input_proj(x_flat)
            h = h.view(batch_size, seq_len, -1)
            h, _ = self.temporal(h)
            h = h[:, -1, :]  # Take last timestep

        # Spatial GNN
        for gnn_layer in self.gnn_layers:
            h_new = gnn_layer(h, edge_index, edge_attr)
            h = h + h_new  # Residual connection

        # Global pooling
        if batch is not None:
            global_h = global_mean_pool(h, batch)
        else:
            global_h = h.mean(dim=0, keepdim=True)

        global_h = self.global_pool(global_h)

        # Node-level predictions
        severity = self.severity_head(h)

        # Global-level predictions
        impact = self.impact_head(global_h)

        # Spatial extent (combine node and global features)
        if batch is not None:
            global_expanded = global_h[batch]
        else:
            global_expanded = global_h.expand(h.size(0), -1)

        combined = torch.cat([h, global_expanded], dim=-1)
        spatial_extent = self.spatial_extent_head(combined)

        return {
            'node_severity': severity,
            'global_impact': impact,
            'spatial_extent': spatial_extent,
            'node_embeddings': h,
            'global_embedding': global_h
        }


class SpatioTemporalLoss(nn.Module):
    """시공간 모델 손실 함수"""

    def __init__(self, impact_weights: Optional[List[float]] = None):
        super().__init__()
        self.impact_weights = impact_weights or [1.0, 1.0, 1.0]

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        losses = {}

        # Severity loss (node-level)
        if 'severity' in targets:
            losses['severity_loss'] = F.mse_loss(
                outputs['node_severity'].squeeze(-1),
                targets['severity']
            )

        # Impact loss (global-level)
        if 'impact' in targets:
            impact_loss = 0
            for i, w in enumerate(self.impact_weights):
                impact_loss += w * F.mse_loss(
                    outputs['global_impact'][:, i],
                    targets['impact'][:, i]
                )
            losses['impact_loss'] = impact_loss

        # Spatial extent loss
        if 'spatial_extent' in targets:
            losses['extent_loss'] = F.binary_cross_entropy(
                outputs['spatial_extent'].squeeze(-1),
                targets['spatial_extent']
            )

        # Total loss
        losses['total_loss'] = sum(losses.values())

        return losses


def create_sample_graph(n_nodes: int = 20, n_features: int = 8) -> Data:
    """테스트용 샘플 그래프 생성"""
    # 랜덤 노드 특성
    x = torch.randn(n_nodes, n_features)

    # 랜덤 에지 (약 30% 연결)
    edge_list = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j and np.random.random() < 0.3:
                edge_list.append([i, j])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.rand(edge_index.size(1), 1)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def main():
    """모델 테스트"""
    config = GNNConfig(
        node_features=8,
        hidden_dim=64,
        n_gnn_layers=3,
        output_dim=1
    )

    model = SpatioTemporalGNN(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 샘플 그래프 생성
    graph = create_sample_graph()
    print(f"\nGraph: {graph.x.shape[0]} nodes, {graph.edge_index.shape[1]} edges")

    # Forward pass
    outputs = model(graph.x, graph.edge_index, edge_attr=graph.edge_attr)

    print(f"\nNode severity shape: {outputs['node_severity'].shape}")
    print(f"Global impact shape: {outputs['global_impact'].shape}")
    print(f"Spatial extent shape: {outputs['spatial_extent'].shape}")
    print(f"Node embeddings shape: {outputs['node_embeddings'].shape}")

    # Loss calculation
    loss_fn = SpatioTemporalLoss()
    targets = {
        'severity': torch.rand(graph.x.size(0)),
        'impact': torch.rand(1, 3),
        'spatial_extent': torch.rand(graph.x.size(0))
    }
    losses = loss_fn(outputs, targets)

    print(f"\nTotal loss: {losses['total_loss']:.4f}")
    for k, v in losses.items():
        if k != 'total_loss':
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
