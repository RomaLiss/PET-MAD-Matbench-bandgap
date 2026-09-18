"""
Циклы обучения, оценки и инференса.

Источник признаков вынесен в вызываемый объект, чтобы цикл обучения не знал,
откуда они пришли:

    AtomsForward   гоняет энкодер по структурам. Нужен на второй стадии, где
                   энкодер дообучается и его выход меняется каждый шаг.
    CachedForward  читает готовые признаки из кэша. Допустим только на первой
                   стадии, где энкодер заморожен.
"""

import math
import time
from typing import NamedTuple

import numpy as np
import torch

# metatomic — прослойка между обученной моделью и ASE; ModelOutput описывает,
# какую величину и в каком виде запросить у модели.
from metatomic.torch import ModelOutput
from torch.amp import autocast
from tqdm import tqdm


# ---------------------------------------------------------------------------
# функция потерь и группы параметров
# ---------------------------------------------------------------------------
def build_criterion(cfg):
    """
    Создаёт функцию потерь, указанную в конфигурации, с её параметрами.
    """
    name = cfg.train.loss
    if name == "HuberLoss":
        return torch.nn.HuberLoss(delta=cfg.train.huber_delta)
    if name == "MSELoss":
        return torch.nn.MSELoss()
    raise ValueError(
        f"неизвестная функция потерь {name!r}; поддерживаются "
        f"HuberLoss и MSELoss"
    )


class AtomsForward:
    """Источник признаков: запуск энкодера. Обязателен, когда энкодер обучается."""

    def __init__(self, model, calc):
        self.model = model
        self.calc = calc

    def __call__(self, payload):
        """payload — список объектов ASE Atoms."""
        # (GPT-answer) run_model — метод MetatomicCalculator (metatomic).
        # Прогоняет модель по списку структур разом и считает только
        # запрошенный выход. Обычный интерфейс ASE (get_potential_energy и
        # подобные) отдаёт лишь заранее оговорённый набор величин — энергию,
        # силы; "features" в него не входит, это внутреннее представление
        # модели, а не физическая величина. Отсюда и обходной путь через
        # run_model. per_atom=True — нужен вектор на КАЖДЫЙ атом: пулингу
        # нужны признаки атомов по отдельности, он сам сворачивает их в
        # описание структуры.
        #
        # Результат — словарь {запрошенное имя: TensorMap}; ["features"]
        # достаёт по ключу (запрошен ровно один выход). .block() — тоже из
        # metatomic: признаки в TensorMap хранятся не как обычный тензор
        # PyTorch, .block() извлекает собственно данные.
        block = self.calc.run_model(
            payload, {"features": ModelOutput(per_atom=True)}
        )["features"].block()

        # {"features": вот здесь происходит конкатенация узлов и связей, когда просим отдать фичи с модели
        
        # block.values (атомов, 1280) — признаки всех атомов всех структур payload подряд.
        # block.samples — таблица метаданных на каждую строку values;
        # столбец "system" metatomic заполняет сам — номер структуры
        # для каждого атома, ровно то, что нужно пулингу как system_indices.
        return self.model(block.values, block.samples.column("system"), len(payload))


class CachedForward:
    """Источник признаков: готовый кэш. НУжен при замороженном энкодере."""

    def __init__(self, model, cache, *, device):
        self.model = model
        self.cache = cache
        self.device = device

    def __call__(self, payload):
        """payload — список номеров структур в кэше."""
        feats, sysidx, n = self.cache.gather(payload, self.device)
        return self.model(feats, sysidx, n)


# ---------------------------------------------------------------------------
# циклы
# ---------------------------------------------------------------------------
def train_epoch(model, loader, forward, optimizer, scaler, criterion, *,
                device, amp=True, clip_grad=None, desc="обучение"):
    """Один проход по обучающей выборке. Возвращает (RMSE, MAE).

    clip_grad=None означает отсутствие ограничения на норму градиента; число
    включает ограничение после снятия масштабирования — это правильный порядок
    при смешанной точности.
    """
    model.train()
    sq_sum, abs_sum, count = 0.0, 0.0, 0
    pbar = tqdm(loader, desc=desc, leave=False) # прогресс бар
    for payload, targets in pbar:
        n = len(payload)
        optimizer.zero_grad()
        with autocast(device_type="cuda", enabled=amp):
            preds = forward(payload)
            loss = criterion(preds, targets.to(device).to(preds.dtype))

        scaler.scale(loss).backward()
        if clip_grad is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            err = (preds.detach().float() - targets.to(device).float()).abs()
            sq_sum += (err ** 2).sum().item()
            abs_sum += err.sum().item()
        count += n
        pbar.set_postfix({"RMSE": f"{math.sqrt(sq_sum / count):.4f}"})

    return math.sqrt(sq_sum / count), abs_sum / count


@torch.no_grad()
def evaluate(model, loader, forward, *, device, amp=True):
    """
    Оценка на выборке. Возвращает (RMSE, MAE, стандартное отклонение ошибки).
     """
    model.eval()
    errors = []
    for payload, targets in loader:
        with autocast(device_type="cuda", enabled=amp):
            preds = forward(payload)
            t = targets.to(device).to(preds.dtype)
            errors.extend(torch.abs(preds - t).float().cpu().numpy().flatten())
    # float64 при усреднении: ошибок здесь под десять тысяч, и их
    # и их сложение приведёт к потери точности.
    errors = np.asarray(errors, dtype=np.float64)
    return float(np.sqrt((errors ** 2).mean())), float(errors.mean()), float(errors.std())


@torch.no_grad()
def predict(model, calc, atoms_list, *, batch_size=64, amp=True, desc="инференс"):
    """
    Инференс по списку структур. Возвращает одномерный массив предсказаний.
    """
    model.eval()
    forward = AtomsForward(model, calc)
    out = []
    for i in tqdm(range(0, len(atoms_list), batch_size), desc=desc, leave=False):
        with autocast(device_type="cuda", enabled=amp):
            preds = forward(atoms_list[i : i + batch_size])
        out.append(preds.float().cpu().numpy().flatten())
    return np.concatenate(out) if out else np.array([])


class StageResult(NamedTuple):
    """
    Итог одной стадии обучения.
    """

    best_rmse: float          #: лучший val RMSE за стадию
    best_mae: float           #: лучший val MAE за стадию
    best_rmse_epoch: int      #: эпоха, на которой достигнут лучший RMSE
    best_mae_epoch: int       #: эпоха, на которой достигнут лучший MAE
    lrs: list                 #: скорости обучения по группам на выходе стадии


def run_stage(*, stage_id, fold, model, train_loader, val_loader, forward,
              optimizer, scheduler, scaler, criterion, epochs, clip_grad,
              device, amp, metrics_log, checkpointer, diagnostics=None,
              save_last=True):
    """Проводит одну стадию обучения. Возвращает `StageResult`.

    Лучшая эпоха по RMSE и лучшая эпоха по MAE — сохраняются они в разные файлы. 
    Что именно загружать дальше, решает не эта функция, 
    а вызывающий код: стадия не знает, какая метрика в данной задаче считается главной.
    Расписание скорости обучения - RMSE
    """
    best = {"rmse": float("inf"), "mae": float("inf")}
    best_epoch = {"rmse": 0, "mae": 0}
    prev_weights = diagnostics.snapshot(model) if diagnostics else None

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_rmse, train_mae = train_epoch(
            model, train_loader, forward, optimizer, scaler, criterion,
            device=device, amp=amp, clip_grad=clip_grad,
            desc=f"Ф{fold} С{stage_id} Э{epoch}",
        )
        val_rmse, val_mae, val_std = evaluate(
            model, val_loader, forward, device=device, amp=amp
        )
        lrs = [g["lr"] for g in optimizer.param_groups]

        metrics_log.write_row(
            fold=fold, stage=stage_id, epoch=epoch,
            train_rmse=train_rmse, train_mae=train_mae,
            val_rmse=val_rmse, val_mae=val_mae,
            val_std=val_std, lrs=lrs, seconds=time.time() - t0,
        )

        # Лучшая эпоха по каждой метрике своя, и чекпоинты у них разные.
        # Какой из них загружать дальше, решает вызывающий код.
        for name, value in (("rmse", val_rmse), ("mae", val_mae)):
            if value < best[name]:
                best[name] = value
                best_epoch[name] = epoch
                checkpointer.save_best(model, fold=fold, stage=stage_id, metric=name)
        if save_last:
            checkpointer.save_last(model, fold=fold)

        if diagnostics:
            prev_weights = diagnostics.log_changes(
                fold=fold, stage=stage_id, epoch=epoch, model=model, prev=prev_weights
            )

        scheduler.step(val_rmse)

    return StageResult(
        best_rmse=best["rmse"], best_mae=best["mae"],
        best_rmse_epoch=best_epoch["rmse"], best_mae_epoch=best_epoch["mae"],
        lrs=[g["lr"] for g in optimizer.param_groups],
    )
