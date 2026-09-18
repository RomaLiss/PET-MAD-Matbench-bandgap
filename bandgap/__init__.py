"""
Предсказание band_gap поверх предобученного энкодера PET-MAD.

Состав
-------------
bandgap/          библиотека
  model.py          архитектура: энкодер + внимание + голова
  engine.py         проход по данным, эпоха, стадия, инференс
  strategy.py       порядок двух стадий обучения
  inference.py      BandgapPredictor — предсказание для новых структур
  data.py           загрузчики батчей
  features.py       кэш признаков 
  config.py         чтение YAML и переопределения
  run.py            папка прогона, журнал метрик, чекпоинты
  diagnostics.py    журнал дрейфа весов энкодера

experiments/
  matbench_mp_gap.py       обучение на Matbench mp_gap
  configs/mp_gap_base.yaml все гиперпараметры

tools/predict.py    предсказание из командной строки

model/
  fold_0_best_S2_mae.pth   обученные веса, 115 МБ
  train_config.yaml        конфигурация прогона, которым обучены веса

Быстрое предсказание
--------------------
    from bandgap.inference import BandgapPredictor
    from ase.io import read

    predictor = BandgapPredictor()             
    print(predictor(read("структура.cif")))    

"""

__version__ = "0.1.0"
