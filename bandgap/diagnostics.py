"""
(Этот раздел сделал и закоментил GPT) Журнал изменения весов по слоям.

Этот код фиксирует изменения по эпохам. Чтобы измерить накопленный уход от
предобученного состояния, нужен инструмент tools/measure_drift.py (в расширенной версии).
"""

import os

import torch


class WeightDiagnostics:
    """
    Считает, насколько сильно изменился каждый слой за эпоху.

    ЧТО ЗАПИСЫВАЕТСЯ
    ----------------
    Строка на каждый параметр на каждой эпохе:

        fold, stage, epoch, part, layer_name, change_norm

    где `part` — «Body» для энкодера и «Head» для всего остального.

    ОГРАНИЧЕНИЕ
    -----------
    Этот журнал показывает изменения за эпоху, 
    Для накопленного дрейфа есть tools/measure_drift.py (сгенереный), 
    который сравнивает веса с предобученными напрямую.
    """

    def __init__(self, path, enabled=True):
        self.enabled = enabled
        self.path = path
        if self.enabled and not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write("fold,stage,epoch,part,layer_name,change_norm\n")

    def snapshot(self, model):
        """Копия всех параметров модели для последующего сравнения."""
        if not self.enabled:
            return None
        return {name: p.data.detach().cpu().clone() for name, p in model.named_parameters()}

    def log_changes(self, *, fold, stage, epoch, model, prev):
        """
        Записывает норму изменения каждого параметра за эпоху.
        Возвращает новый снимок, который станет точкой отсчёта для следующей эпохи.
        """
        if not self.enabled:
            return None
        current = self.snapshot(model)
        with open(self.path, "a", encoding="utf-8") as f:
            for name, tensor in current.items():
                if name not in prev:
                    continue
                norm = torch.norm(tensor - prev[name]).item()
                part = "Body" if "encoder" in name else "Head"
                f.write(f"{fold},{stage},{epoch},{part},{name},{norm:.8e}\n")
        return current
