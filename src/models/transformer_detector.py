"""
Transformer-based Compound Event Detector
시계열 Transformer 모델을 활용한 복합 극한기후 탐지

Architecture:
- Multi-head Self-Attention for temporal dependencies
- Positional Encoding for sequence order
- Classification head for event type prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """Transformer 모델 설정"""
    input_dim: int = 7          # 입력 특성 수 (temp_max, temp_min, precip, ...)
    d_model: int = 128          # 모델 차원
    n_heads: int = 8            # 어텐션 헤드 수
    n_layers: int = 4           # 인코더 레이어 수
    d_ff: int = 512             # FFN 차원
    dropout: float = 0.1
    max_seq_len: int = 365      # 최대 시퀀스 길이 (1년)
    n_classes: int = 6          # 이벤트 유형 수


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SeasonalEncoding(nn.Module):
    """계절성 인코딩 (기후 데이터용)"""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.linear = nn.Linear(4, d_model)  # sin/cos for day of year and week

    def forward(self, day_of_year: torch.Tensor) -> torch.Tensor:
        """
        Args:
            day_of_year: (batch_size, seq_len) - 1-365
        """
        # 연간 주기
        year_sin = torch.sin(2 * math.pi * day_of_year / 365)
        year_cos = torch.cos(2 * math.pi * day_of_year / 365)

        # 주간 주기
        week_sin = torch.sin(2 * math.pi * day_of_year / 7)
        week_cos = torch.cos(2 * math.pi * day_of_year / 7)

        seasonal = torch.stack([year_sin, year_cos, week_sin, week_cos], dim=-1)
        return self.linear(seasonal)


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query, key, value: (batch_size, seq_len, d_model)
            mask: (batch_size, seq_len, seq_len) or None

        Returns:
            output: (batch_size, seq_len, d_model)
            attention_weights: (batch_size, n_heads, seq_len, seq_len)
        """
        batch_size = query.size(0)

        # Linear projections
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.W_o(context)
        return output, attention_weights


class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention
        attn_out, attn_weights = self.self_attn(x, x, x, mask)
        x = self.norm1(x + attn_out)

        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x, attn_weights


class CompoundEventTransformer(nn.Module):
    """
    복합 극한기후 탐지를 위한 Transformer 모델

    Input: 다변량 기상 시계열 (temp_max, temp_min, precip, humidity, ...)
    Output: 각 시점별 복합 이벤트 유형 분류
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(config.input_dim, config.d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            config.d_model, config.max_seq_len, config.dropout
        )
        self.seasonal_encoding = SeasonalEncoding(config.d_model)

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                config.d_model, config.n_heads, config.d_ff, config.dropout
            )
            for _ in range(config.n_layers)
        ])

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.n_classes)
        )

        # Severity regression head
        self.severity_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 4),
            nn.GELU(),
            nn.Linear(config.d_model // 4, 1),
            nn.ReLU()  # 강도는 비음수
        )

        self._init_weights()

    def _init_weights(self):
        """가중치 초기화"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        day_of_year: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch_size, seq_len, input_dim) - 기상 변수 시계열
            day_of_year: (batch_size, seq_len) - 일자 정보 (1-365)
            mask: (batch_size, seq_len) - 패딩 마스크

        Returns:
            Dict with:
            - logits: (batch_size, seq_len, n_classes)
            - severity: (batch_size, seq_len, 1)
            - attention_weights: List of attention weight tensors
        """
        batch_size, seq_len, _ = x.shape

        # Input projection
        h = self.input_proj(x)

        # Add positional encoding
        h = self.pos_encoding(h)

        # Add seasonal encoding if provided
        if day_of_year is not None:
            h = h + self.seasonal_encoding(day_of_year)

        # Create attention mask
        if mask is not None:
            attn_mask = mask.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.expand(-1, self.config.n_heads, seq_len, -1)
        else:
            attn_mask = None

        # Transformer encoding
        attention_weights = []
        for layer in self.encoder_layers:
            h, attn_w = layer(h, attn_mask)
            attention_weights.append(attn_w)

        # Classification
        logits = self.classifier(h)

        # Severity prediction
        severity = self.severity_head(h)

        return {
            'logits': logits,
            'severity': severity,
            'attention_weights': attention_weights,
            'hidden_states': h
        }

    def predict(
        self,
        x: torch.Tensor,
        day_of_year: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        추론 모드

        Returns:
            predictions: (batch_size, seq_len) - 예측 클래스
            probabilities: (batch_size, seq_len, n_classes) - 클래스 확률
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x, day_of_year)
            probs = F.softmax(outputs['logits'], dim=-1)
            preds = probs.argmax(dim=-1)
        return preds, probs


class CompoundEventLoss(nn.Module):
    """복합 이벤트 학습을 위한 손실 함수"""

    def __init__(
        self,
        n_classes: int,
        class_weights: Optional[torch.Tensor] = None,
        severity_weight: float = 0.2
    ):
        super().__init__()
        self.n_classes = n_classes
        self.severity_weight = severity_weight

        # 클래스 불균형 처리
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: model outputs with 'logits' and 'severity'
            targets: Dict with 'labels' and 'severity'

        Returns:
            Dict with loss components
        """
        # Classification loss
        logits = outputs['logits'].view(-1, self.n_classes)
        labels = targets['labels'].view(-1)
        cls_loss = self.ce_loss(logits, labels)

        # Severity loss (only for positive samples)
        if 'severity' in targets:
            pred_severity = outputs['severity'].squeeze(-1)
            true_severity = targets['severity']

            # 이벤트가 있는 샘플만
            event_mask = targets['labels'] > 0
            if event_mask.any():
                sev_loss = self.mse_loss(
                    pred_severity[event_mask],
                    true_severity[event_mask]
                )
            else:
                sev_loss = torch.tensor(0.0, device=logits.device)
        else:
            sev_loss = torch.tensor(0.0, device=logits.device)

        total_loss = cls_loss + self.severity_weight * sev_loss

        return {
            'total_loss': total_loss,
            'classification_loss': cls_loss,
            'severity_loss': sev_loss
        }


def create_sample_data(
    batch_size: int = 8,
    seq_len: int = 100,
    input_dim: int = 7
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """테스트용 샘플 데이터 생성"""
    x = torch.randn(batch_size, seq_len, input_dim)
    day_of_year = torch.randint(1, 366, (batch_size, seq_len)).float()
    labels = torch.randint(0, 6, (batch_size, seq_len))
    return x, day_of_year, labels


def main():
    """모델 테스트"""
    # 설정
    config = TransformerConfig(
        input_dim=7,
        d_model=128,
        n_heads=8,
        n_layers=4,
        d_ff=512,
        n_classes=6
    )

    # 모델 생성
    model = CompoundEventTransformer(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 샘플 데이터
    x, day_of_year, labels = create_sample_data()

    # Forward pass
    outputs = model(x, day_of_year)

    print(f"\nInput shape: {x.shape}")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Severity shape: {outputs['severity'].shape}")
    print(f"Attention weights: {len(outputs['attention_weights'])} layers")

    # Loss calculation
    loss_fn = CompoundEventLoss(n_classes=config.n_classes)
    targets = {'labels': labels, 'severity': torch.rand_like(labels.float())}
    losses = loss_fn(outputs, targets)

    print(f"\nTotal loss: {losses['total_loss']:.4f}")
    print(f"Classification loss: {losses['classification_loss']:.4f}")
    print(f"Severity loss: {losses['severity_loss']:.4f}")

    # Prediction
    preds, probs = model.predict(x, day_of_year)
    print(f"\nPredictions shape: {preds.shape}")
    print(f"Probabilities shape: {probs.shape}")


if __name__ == "__main__":
    main()
