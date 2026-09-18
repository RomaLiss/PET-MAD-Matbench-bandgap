"""
Обучение на Matbench mp_gap: пять фолдов, две стадии, перенос с PET-MAD.
(гпт тоже помогала сделать здесь все надежнее и симпотичнее)
Примеры обычного запуска:

    python experiments/matbench_mp_gap.py
    python experiments/matbench_mp_gap.py --folds 0 --set stages.s2.epochs=0
    python experiments/matbench_mp_gap.py --name proba --set train.huber_delta=0.1
"""

import argparse
import gc
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bandgap.config import load_config, save_config, set_seed  # noqa: E402 — до импорта torch


def cache_dtype(name):
    """Тип чисел для кэша признаков по названию из конфигурации.

    Явное перечисление вместо getattr(torch, строка): читателю сразу видно,
    из чего выбирать, а опечатка в конфигурации даёт понятное сообщение, а не
    ошибку об отсутствующем атрибуте.

    torch импортируется внутри функции: в этом файле его нельзя трогать на
    уровне модуля, пока не выставлены переменные окружения.
    """
    import torch

    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(
        f"неизвестный тип чисел кэша {name!r}; поддерживаются "
        f"float16 и float32"
    )


def parse_args():
    """Аргументы командной строки.

    Гиперпараметры сюда не добавляются. Их место — в файле конфигурации, а
    разовое изменение делается через `--set раздел.параметр=значение`. Причина:
    конфигурация целиком копируется в папку прогона, и по ней потом видно, чем
    получен результат. Аргумент командной строки в такой снимок не попал бы.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=os.path.join(here, "configs", "mp_gap_base.yaml"),
                   help="файл конфигурации")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   help="переопределить параметр, например --set train.batch_size=64")
    p.add_argument("--name", default=None, help="метка для имени папки запуска")
    p.add_argument("--folds", default=None,
                   help="какие фолды считать через запятую, например '0' для пробы")
    return p.parse_args()


def main():
    """Полный прогон: подготовка, затем по каждому фолду две стадии и инференс.

    ПОРЯДОК ДЕЙСТВИЙ
    ----------------
        1. читаем конфигурацию и выставляем переменные окружения
           (обязательно ДО импорта torch — библиотеки читают их при загрузке)
        2. создаём папку прогона, копируем туда код и конфигурацию
        3. загружаем данные Matbench и переводим структуры в формат ASE
        4. собираем кэш признаков — один раз на весь прогон, а не на фолд
        5. по каждому фолду:
              стадия 1 (энкодер заморожен, признаки из кэша)
              загрузка лучшего чекпоинта стадии 1
              стадия 2 (дообучение целиком, кэш выгружен из видеопамяти)
              загрузка лучшего чекпоинта стадии 2, предсказания на тесте
              полная очистка памяти перед следующим фолдом
        6. записываем файл результатов

    Функция длинная намеренно. Порядок шагов здесь и есть содержание
    эксперимента, и разнесение его по мелким функциям сделало бы этот порядок
    неочевидным — а именно порядок (когда берётся снимок предобученных весов,
    когда выгружается кэш, когда чистится память) уже дважды был источником
    ошибок.
    """
    args = parse_args()
    cfg = load_config(args.config, args.overrides)

    # Обязательно до любого импорта torch или upet: переменные окружения
    # считываются этими библиотеками в момент загрузки.
    os.environ["VESIN_CUDA_MAX_PAIRS_PER_POINT"] = str(cfg.hardware.vesin_max_pairs)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.hardware.cuda_visible_devices)

    # Расширяемые сегменты аллокатора CUDA — против фрагментации.
    #
    # Замерено на прогоне из трёх фолдов: во второй стадии при пике занятой
    # памяти 14.1 ГБ аллокатор удерживал у драйвера 22.3 ГБ из 24 доступных.
    # Разница в восемь гигабайт — фрагментация: структуры содержат разное число
    # атомов, тензоры получаются разной формы, и вместо переиспользования
    # блоков аллокатор запрашивает у драйвера новые сегменты.
    #
    # Само обучение от этого не страдает, а вот библиотека поиска соседей vesin
    # выделяет память МИМО аллокатора PyTorch, и ей остаётся меньше двух
    # гигабайт. На первом фолде хватило, на втором прогон упал с нехваткой
    # памяти. То есть успех первого фолда был везением.
    #
    # Расширяемые сегменты позволяют аллокатору наращивать уже выделенную
    # область вместо запроса новых блоков — именно для нагрузок с переменной
    # формой тензоров этот режим и предназначен.
    if cfg.hardware.expandable_segments:
        prev = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
            (prev + ",") if prev else "") + "expandable_segments:True"

    import torch
    from metatomic.torch.ase_calculator import MetatomicCalculator
    from sklearn.model_selection import train_test_split

    from bandgap import data as data_mod
    from bandgap import engine
    from bandgap.diagnostics import WeightDiagnostics
    from bandgap.engine import build_criterion
    from bandgap.model import build_model, encoder_id
    from bandgap.run import Checkpointer, MetricsLog, Tee, make_run_dir
    from bandgap.strategy import run_two_stages

    # Значение проверяется здесь, а не в момент использования: загрузка
    # чекпоинта случается впервые только после первой стадии, то есть спустя
    # часы после старта, и опечатка в имени метрики обнаружилась бы там.
    if cfg.train.selection_metric not in Checkpointer.METRICS:
        raise SystemExit(
            f"train.selection_metric={cfg.train.selection_metric!r} — "
            f"допустимы {Checkpointer.METRICS}")

    run_dir = make_run_dir(cfg.paths.runs_root, args.name or "mp_gap")
    save_config(cfg, os.path.join(run_dir, "config.yaml"))

    with Tee(os.path.join(run_dir, "stdout.log")):
        print(f"папка запуска: {run_dir}")
        set_seed(cfg.seed)
        device = cfg.hardware.device

        metrics = MetricsLog(os.path.join(run_dir, "metrics.csv"))
        ckpt = Checkpointer(os.path.join(run_dir, "checkpoints"))
        diag = WeightDiagnostics(
            os.path.join(run_dir, "weight_layer_stats.csv"), enabled=cfg.diagnostics.enabled
        )

        mb, task, index_to_ase = data_mod.load_matbench_task(cfg.data.task)

        # Заявляем предобученный энкодер в файле результатов: исходный прогон
        # оставлял это поле пустым, а использование чужой предобученной модели
        # надо декларировать самим, а не ждать вопроса от рецензента.
        mb.add_metadata({
            "algorithm": cfg.submission.algorithm,
            "notes": cfg.submission.notes,
            "encoder": encoder_id(cfg),
            "loss": cfg.train.loss,
        })

        criterion = build_criterion(cfg)

        # --- кэш признаков ------------------------------------------------
        # На первой стадии энкодер заморожен, и он одинаков во всех фолдах,
        # поэтому его выход считается один раз на весь прогон. Именно на этот
        # пересчёт уходило всё время первой стадии.
        feature_cache = None
        cache_pos = None
        if cfg.cache.enabled:
            from bandgap.features import FeatureCache

            all_index = list(task.df.index)
            cache_pos = {idx: i for i, idx in enumerate(all_index)}
            probe = build_model(cfg, device)
            probe_calc = MetatomicCalculator(probe.encoder, device=device)
            feature_cache = FeatureCache.load_or_build(
                os.path.join(cfg.paths.cache_root, cfg.cache.filename),
                probe_calc,
                [index_to_ase[i] for i in all_index],
                encoder_id=encoder_id(cfg),
                batch_size=cfg.cache.build_batch_size,
                amp=cfg.cache.amp,
                dtype=cache_dtype(cfg.cache.dtype),
            )
            del probe, probe_calc
            torch.cuda.empty_cache()

        folds = ([int(x) for x in args.folds.split(",")] if args.folds
                 else list(range(cfg.data.n_folds)))

        def vram(tag, reset=False):
            """Отчёт о видеопамяти.

            Печатаются две величины, и различать их важно:
                занято         — под тензорами прямо сейчас
                зарезервировано — сколько PyTorch держит у драйвера, включая
                                  свободные блоки в своём кэше

            Расхождение между ними и есть фрагментация. Она объясняет ситуации,
            когда «занято» мало, а сторонняя библиотека (например, поиск
            соседей vesin, который выделяет память мимо аллокатора PyTorch)
            всё равно не может получить непрерывный блок и падает с нехваткой
            памяти.

            reset=True сбрасывает счётчик пика: без этого он показывает
            максимум за всё время работы, а не за текущий фолд.
            """
            if not torch.cuda.is_available():
                return
            print(f"    [память] {tag}: занято "
                  f"{torch.cuda.memory_allocated() / 2**30:.2f} ГБ, "
                  f"зарезервировано {torch.cuda.memory_reserved() / 2**30:.2f} ГБ, "
                  f"пик {torch.cuda.max_memory_allocated() / 2**30:.2f} ГБ")
            if reset:
                torch.cuda.reset_peak_memory_stats()

        for fold in folds:
            print(f"\n{'=' * 60}\nФОЛД {fold}\n{'=' * 60}")
            vram("в начале фолда", reset=True)

            train_in, train_out = task.get_train_and_val_data(fold)
            tr_in, val_in, tr_out, val_out = train_test_split(
                train_in, train_out,
                test_size=cfg.data.val_fraction,
                random_state=cfg.data.split_seed,
            )
            y_train, y_val = tr_out.tolist(), val_out.tolist()

            model = build_model(cfg, device)
            calc = MetatomicCalculator(model.encoder, device=device)
            scaler = torch.amp.GradScaler()
            atoms_forward = engine.AtomsForward(model, calc)

            # Загрузчик для второй стадии: там энкодер обучается, признаки
            # приходится считать заново каждый шаг.
            s2_train_loader = data_mod.get_loader(
                [index_to_ase[i] for i in tr_in.index], y_train,
                cfg.train.batch_size, cfg.hardware.num_workers,
                shuffle=cfg.train.shuffle_train,
            )
            s2_val_loader = data_mod.get_loader(
                [index_to_ase[i] for i in val_in.index], y_val,
                cfg.train.batch_size, cfg.hardware.num_workers,
                shuffle=False,
            )

            # Загрузчик для первой стадии: энкодер заморожен, признаки из кэша.
            if feature_cache is not None:
                s1_train_loader = data_mod.get_index_loader(
                    [cache_pos[i] for i in tr_in.index], y_train,
                    cfg.train.batch_size, shuffle=cfg.train.shuffle_train,
                )
                s1_val_loader = data_mod.get_index_loader(
                    [cache_pos[i] for i in val_in.index], y_val,
                    cfg.train.batch_size, shuffle=False,
                )
                # Кэш переезжает в видеопамять на время первой стадии: при
                # хранении в оперативной выборка батчей сама становится узким
                # местом и ускорение падает с сорокакратного до двукратного.
                feature_cache.to(cfg.cache.device)
                s1_forward = engine.CachedForward(model, feature_cache, device=device)
            else:
                s1_train_loader, s1_val_loader = s2_train_loader, s2_val_loader
                s1_forward = atoms_forward

            common = dict(
                fold=fold, model=model, scaler=scaler, criterion=criterion,
                device=device, amp=cfg.train.amp,
                metrics_log=metrics, checkpointer=ckpt,
                diagnostics=diag if cfg.diagnostics.enabled else None,
            )

            res1, res2 = run_two_stages(
                cfg, common=common,
                s1_train=s1_train_loader, s1_val=s1_val_loader,
                s1_forward=s1_forward,
                s2_train=s2_train_loader, s2_val=s2_val_loader,
                s2_forward=atoms_forward,
                feature_cache=feature_cache, report=vram,
            )
            vram("после обучения")

            # При stages.s2.epochs = 0 второй стадии не было, предсказывать
            # нечем — сразу к следующему фолду.
            if res2 is not None:
                select = cfg.train.selection_metric

                # ---- предсказания на тесте для официальной записи ----
                # Той же метрикой, что и на переходе между стадиями: иначе отбор
                # внутри обучения и отбор итоговой модели мерили бы разным.
                ckpt.load_best(model, fold=fold, stage=2, device=device, metric=select)
                print(f"загружен лучший чекпоинт стадии 2 по {select.upper()}")
                test_in = task.get_test_data(fold, include_target=False)
                frames = [index_to_ase[i] for i in test_in.index]
                preds = engine.predict(
                    model, calc, frames,
                    batch_size=cfg.test.batch_size, amp=cfg.test.amp,
                    desc=f"инференс, фолд {fold}",
                )
                task.record(fold, preds.tolist())
                print(f"фолд {fold} записан: {len(preds)} предсказаний, "
                      f"среднее {preds.mean():.4f} эВ")

                # Файл результатов переписывается после каждого фолда: если прогон
                # оборвётся, уже посчитанные фолды не пропадут.
                mb.to_file(os.path.join(run_dir, cfg.paths.results_file))

            # Освобождаем всё, что держит ссылки на модель: загрузчики,
            # функции доступа к признакам, сама модель. Оптимизаторы и
            # планировщики перечислять больше не нужно — они живут внутри
            # run_two_stages и умирают при возврате из неё.
            del (model, calc, atoms_forward, scaler, s1_forward,
                 s1_train_loader, s1_val_loader,
                 s2_train_loader, s2_val_loader)
            gc.collect()
            torch.cuda.empty_cache()
            vram("после очистки")

        print(f"\nВсе запрошенные фолды посчитаны. Результаты в {run_dir}")


if __name__ == "__main__":
    main()
