"""
Стратегия обучения: в каком порядке идут стадии и что между ними.
"""
import gc

import torch

from bandgap.engine import run_stage


def run_two_stages(cfg, *, common, s1_train, s1_val, s1_forward,
                   s2_train, s2_val, s2_forward,
                   feature_cache=None, report=None):
    """Полный протокол обучения: сначала голова, потом вся модель.

    Аргументы:
        common        словарь общих аргументов run_stage (модель, устройство,
                      критерий, журналы) — один и тот же для обеих стадий,
                      чтобы они не могли разъехаться
        s1_*          загрузчики и источник признаков первой стадии
        s2_*          то же для второй стадии 
        feature_cache если задан и лежит в видеопамяти — вернётся в
                      оперативную перед второй стадией, ей нужно место
        report        необязательная функция для отчёта о памяти

    Возвращает (res1, res2). res2 равен None, если вторая стадия отключена
    (stages.s2.epochs = 0) — это режим экспериментов над головой.
    """
    model = common["model"]
    device = common["device"]
    fold = common["fold"]
    ckpt = common["checkpointer"]

    # ---- стадия 1: энкодер заморожен, учится только голова ----
    print(f"\n>>> СТАДИЯ 1: только голова ({cfg.stages.s1.epochs} эпох)")
    for param in model.encoder.parameters():
        param.requires_grad = False
    # Энкодер заморожен, поэтому оптимизатору отдаём только то, что ещё
    # обучается: голову и слой внимания.
    trainable = [param for param in model.parameters() if param.requires_grad]
    opt1 = torch.optim.Adam(trainable, lr=cfg.stages.s1.lr_head)
    sched1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt1, mode="min", factor=cfg.stages.s1.lr_factor,
        patience=cfg.stages.s1.lr_patience,
    )
    res1 = run_stage(
        stage_id=1, optimizer=opt1, scheduler=sched1,
        train_loader=s1_train, val_loader=s1_val, forward=s1_forward,
        epochs=cfg.stages.s1.epochs, clip_grad=cfg.stages.s1.clip_grad,
        save_last=cfg.stages.s1.save_last, **common,
    )
    final_s1_lr = res1.lrs[0]
    print(f">>> стадия 1 завершена. "
          f"лучший val RMSE {res1.best_rmse:.4f} (эпоха {res1.best_rmse_epoch}), "
          f"лучший val MAE {res1.best_mae:.4f} (эпоха {res1.best_mae_epoch}), "
          f"итоговый LR головы {final_s1_lr:.2e}")

    if cfg.stages.s2.epochs == 0:
        print("стадия 2 отключена (epochs=0), останавливаемся")
        return res1, None

    # Второй стадии нужна видеопамять, которую занимает кэш.
    if feature_cache is not None and feature_cache.device.type == "cuda":
        feature_cache.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        if report:
            report("после возврата кэша в оперативную память")

    # Переход между стадиями: продолжаем с ЛУЧШИХ весов первой, а не с
    # последних. Метрика отбора из конфигурации - по умолчанию MAE
    select = cfg.train.selection_metric
    ckpt.load_best(model, fold=fold, stage=1, device=device, metric=select)
    print(f"загружен лучший чекпоинт стадии 1 по {select.upper()}")

    # ---- стадия 2: дообучение целиком ----
    print(f"\n>>> СТАДИЯ 2: дообучение целиком ({cfg.stages.s2.epochs} эпох)")
    for param in model.parameters():
        param.requires_grad = True

    s2 = cfg.stages.s2
    head_lr = s2.lr_head
    body_lr = head_lr / s2.lr_body_coef
    print(f"скорости обучения стадии 2: голова {head_lr:.2e}, "
          f"тело {body_lr:.2e} (итоговый LR стадии 1 был {final_s1_lr:.2e})")

    wd_head = s2.weight_decay
    wd_encoder = (s2.weight_decay if s2.weight_decay_encoder is None
                  else s2.weight_decay_encoder)

    groups = []
    for module, lr, weight_decay in ((model.encoder, body_lr, wd_encoder),
                                     (model.head, head_lr, wd_head),
                                     (model.pooling, head_lr, wd_head)):
        groups.append({"params": list(module.parameters()),
                       "lr": lr, "weight_decay": weight_decay})
    opt2 = torch.optim.Adam(groups)

    sched2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt2, mode="min", factor=s2.lr_factor, patience=s2.lr_patience,
    )
    res2 = run_stage(
        stage_id=2, optimizer=opt2, scheduler=sched2,
        train_loader=s2_train, val_loader=s2_val, forward=s2_forward,
        epochs=s2.epochs, clip_grad=s2.clip_grad,
        save_last=s2.save_last, **common,
    )
    print(f">>> стадия 2 завершена. "
          f"лучший val RMSE {res2.best_rmse:.4f} (эпоха {res2.best_rmse_epoch}), "
          f"лучший val MAE {res2.best_mae:.4f} (эпоха {res2.best_mae_epoch})")
    return res1, res2
