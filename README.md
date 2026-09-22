# bandgap — предсказание ширины запрещённой зоны

Модель принимает кристаллическую структуру и возвращает band_gap в электронвольтах. Поверх предобученного энкодера PET-MAD.

**Точность:** MAE 0.1267 эВ на 21 223 материалах тесте части фолда 0
Matbench `mp_gap`.

> **Только Linux.**

## Установка

Нужен Python 3.10 или новее. Проверено на 3.10 и 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .                 # только предсказание
pip install -e ".[train]"        # если нужно ещё и обучать
pip install matbench --no-deps   # обязательно, следом за предыдущей строкой
```

Третья строка не лишняя: сам по себе `matbench` не встанет. Он требует
версии четырёхлетней давности (`scikit-learn==1.0.1`, `scipy==1.7.3`),
под нынешний Python их нет, а сборка из исходников падает.
.

**!ТОЛЬКО Версия `upet` в зависимостях закреплена жёстко (`==0.1.2`)**

## Предсказание

```bash
python3 tools/predict.py структура.cif
```

Или из кода — две строки:

```python
from bandgap.inference import BandgapPredictor
from ase.build import bulk

predictor = BandgapPredictor()             
gap = predictor(bulk("Si", "diamond", a=5.43))    
```


Читается любой формат, понятный ASE: CIF, POSCAR, xyz и другие. При первом
запуске энкодер PET-MAD (26 млн параметров) скачивается с HuggingFace — нужен
интернет; дальше берётся из кэша `~/.cache/huggingface`.

По умолчанию считает на процессоре. С картой: `--device cuda`.
 

## Обучение

```bash
python3 experiments/matbench_mp_gap.py --folds 0 --name proba
```

Все пять фолдов — без `--folds`. Нужно **не меньше 16 ГБ
видеопамяти**

Поменять любой параметр, не трогая файлы:

```bash
python3 experiments/matbench_mp_gap.py --set stages.s2.epochs=100
```

Опечатка в имени параметра вызовет ошибку.

Результат каждого запуска — в `runs/<дата_время_имя>/`: метрики по эпохам,
чекпоинты.

## Состав

```
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
  train_config.yaml        конфигурация ТОГО прогона, которым обучены веса
  VERSIONS.txt             чем обучены и что обязательно закреплять
```

## Это первая релизная версия 

