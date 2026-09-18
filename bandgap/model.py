"""
Архитектура: энкодер PET-MAD + внимание по атомам + регрессионная голова.

ОБЩАЯ СХЕМА
-----------
    кристаллическая структура
        
    энкодер PET-MAD          признаки на каждый атом, 1280 чисел
        
    внимание по атомам       свёртка в один вектор на структуру
        
    MLP-голова               1280 → 1280 → 512 → 256 → 128 → 1
        
    Softplus                 ширина запрещённой зоны, эВ, неотрицательная

"""

import torch
import torch.nn as nn


class GlobalAttentionPooling(nn.Module):
    """
    Свёртка признаков атомов в одно описание структуры через внимание.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.scoring = nn.Linear(dim, 1)

    def forward(self, feats: torch.Tensor, system_indices: torch.Tensor,
                num_systems: int) -> torch.Tensor:
        """
        feats          (атомов, размерность) — признаки всех атомов батча подряд
        system_indices (атомов,) — номер структуры для каждого атома
        num_systems    число структур в батче

        Аккуратно. перепутанные `system_indices` и `num_systems` не вызовут ошибки —
        просто атомы распределятся по структурам неправильно.
        """
        scores = self.scoring(feats) 
        system_indices = system_indices.to(torch.int64)

        # Максимум оценок ВНУТРИ каждой структуры — для численной устойчивости
        # экспоненты. include_self=False, потому что начальное значение -inf
        # не должно участвовать в сравнении.
        max_per_system = torch.full(
            (num_systems, 1), float("-inf"), device=feats.device, dtype=scores.dtype
        )
        max_per_system.index_reduce_(0, system_indices, scores, "amax", include_self=False)
        exp_scores = torch.exp(scores - max_per_system[system_indices])

        # Сумма экспонент по структуре. Всегда >= 1, поэтому деление безопасно.
        sum_exp = torch.zeros((num_systems, 1), device=feats.device, dtype=exp_scores.dtype)
        sum_exp.index_add_(0, system_indices, exp_scores)
        weights = exp_scores / sum_exp[system_indices]

        pooled = torch.zeros(
            (num_systems, feats.shape[1]), device=feats.device, dtype=feats.dtype
        )
        pooled.index_add_(0, system_indices, feats * weights)
        return pooled

class PETBandgap(nn.Module):
    """
    Энкодер PET-MAD → внимание → MLP → ширина зоны.
    """

    def __init__(self, pet_encoder, input_dim: int = 1280, dropout_rate: float = 0.05):
        super().__init__()
        self.encoder = pet_encoder
        self.pooling = GlobalAttentionPooling(input_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Softplus(),
        )

    def forward(self, feats, system_indices, num_systems):
        """
        Признаки атомов → одно число на структуру.
        Важно: энкодер не вызывается, модель получает фичи на вход без участия энкодера
        На первой стадии я один раз прогоняю структуры через энкодер, сохраняю фичи и потом передаю сюда

        feats          (атомов, 1280) — признаки всех атомов батча подряд
        system_indices (атомов,)      — номер структуры для каждого атома
        num_systems                   — сколько структур в батче
        """
        return self.head(self.pooling(feats, system_indices, num_systems))


def encoder_id(cfg) -> str:
    """
    Строка, однозначно определяющая энкодер.
    Записывается в кэш признаков и сверяется при его загрузке.
    """
    return f"{cfg.model.encoder_name}-{cfg.model.encoder_size}-v{cfg.model.encoder_version}"


def build_model(cfg, device):
    """Собирает модель и переносит на устройство."""
    from upet import get_upet

    encoder = get_upet(
        model=cfg.model.encoder_name,
        size=cfg.model.encoder_size,
        version=cfg.model.encoder_version,
    ).to(device)
    model = PETBandgap(
        encoder,
        input_dim=cfg.model.input_dim,
        dropout_rate=cfg.model.dropout_rate,
    ).to(device)
    return model
