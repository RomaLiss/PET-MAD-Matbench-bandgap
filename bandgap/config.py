"""
Конфигурация: YAML на входе, пространство имён на выходе.

Переопределение через командную строку
-----------------------------------
Значения задаются как `раздел.ключ=значение` и разбираются как YAML

    --set train.batch_size=64
    --set stages.s2.clip_grad=null

"""

import random
from types import SimpleNamespace

import yaml


def _to_ns(obj):
    """Рекурсивно превращает словари в пространства имён — ради доступа через точку."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def to_plain(obj):
    """Обратное превращение: пространство имён в словарь, для сохранения."""
    if isinstance(obj, SimpleNamespace):
        return {k: to_plain(v) for k, v in vars(obj).items()}
    if isinstance(obj, list):
        return [to_plain(v) for v in obj]
    return obj


def _set_nested(d, dotted_key, value):
    """
    Записывает значение по пути вида `раздел.подраздел.ключ`.
    Отсутствующий раздел или ключ — ошибка.
    """
    parts = dotted_key.split(".")
    for p in parts[:-1]:
        if p not in d or not isinstance(d[p], dict):
            raise KeyError(f"нет такого раздела конфигурации: {dotted_key!r}")
        d = d[p]
    if parts[-1] not in d:
        raise KeyError(f"нет такого параметра конфигурации: {dotted_key!r}")
    d[parts[-1]] = value


def load_config(path, overrides=None):
    """Читает YAML, применяет переопределения, возвращает пространство имён."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    for item in overrides or []:
        key, sep, val = item.partition("=")
        if not sep:
            raise ValueError(f"неверный формат переопределения {item!r}, нужно ключ=значение")
        _set_nested(raw, key.strip(), yaml.safe_load(val))

    return _to_ns(raw)


def save_config(cfg, path):
    """
    Сохраняет конфигурацию в папку запуска.
    allow_unicode=True обязательно: без него кириллица в описаниях стала бы нечитаемой.
    """
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_plain(cfg), f, sort_keys=False,
                       default_flow_style=False, allow_unicode=True)


def set_seed(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
