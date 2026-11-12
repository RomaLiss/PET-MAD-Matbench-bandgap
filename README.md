# bandgap — предсказание ширины запрещённой зоны

Модель принимает кристаллическую структуру и возвращает band_gap в электронвольтах. Поверх предобученного энкодера PET-MAD.

**Точность:** MAE 0.1267 эВ на 21 223 материалах тесте части фолда 0
Matbench `mp_gap` (matbench_pbe_v1).

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
 

## Модели

Кроме модели по умолчанию есть ещё шесть: щель PBE и щель HSE для разных наборов данных.
Все веса лежат в [релизе v2.0.0](https://github.com/RomaLiss/PET-MAD-Matbench-bandgap/releases/tag/v2.0.0).
Скачанный файл положите в `model/` или укажите путь явно:

```python
predictor = BandgapPredictor(checkpoint="model/matbench_Kim_hse_v2.pth")
```

```bash
python3 tools/predict.py структура.cif --checkpoint model/matbench_Kim_hse_v2.pth
```

Список моделей, их точность и выбор данных — в разделе [«Модели и данные»](#модели-и-данные) в конце.


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
  matbench_pbe_v1.pth      обученные веса, 115 МБ
  train_config.yaml        конфигурация ТОГО прогона, которым обучены веса
  VERSIONS.txt             чем обучены и что обязательно закреплять
```

## Модели и данные

| файл | предсказывает | обучена на | обучение / проверка / тест |
|---|---|---|---|
| `matbench_pbe_v1.pth` | щель PBE | Matbench `mp_gap`, фолд 0; модель по умолчанию | 76 401 / 8 489 / 21 223 |
| `matbench_pbe_v2.pth` | щель PBE | Matbench `mp_gap`, честное разбиение | 76 401 / 8 489 / 21 223 |
| `matbench_noemd_hse_v1.pth` | щель HSE | noemd, случайное разбиение; энкодер из `matbench_pbe_v1` | 4 491 / 792 / 1 321 |
| `matbench_noemd_hse_v2.pth` | щель HSE | noemd; энкодер из `matbench_pbe_v2` | 4 535 / 1 018 / 1 051 |
| `matbench_MEGNet_hse_v2.pth` | щель HSE | MEGNet; энкодер из `matbench_pbe_v2` | 4 311 / 815 / 904 |
| `matbench_Kim_hse_v2.pth` | щель HSE | Kim et al., немагнитные; энкодер из `matbench_pbe_v2` | 6 931 / 506 / 508 |
| `matbench_noemd_MEGNet_Kim_hse_v2.pth` | щель HSE | noemd + MEGNet + Kim; энкодер из `matbench_pbe_v2` | 15 777 / 2 339 / 2 463 |

### Точность моделей v2

MAE, эВ, на тестовых частях честного разбиения, в скобках — число материалов.

| модель | Matbench PBE (21 223) | noemd HSE (1 051) | MEGNet HSE (904) | Kim HSE (508) |
|---|---|---|---|---|
| `matbench_pbe_v2` | **0.155** | — | — | — |
| `matbench_noemd_hse_v2` | — | **0.207** | 0.712 | 0.460 |
| `matbench_MEGNet_hse_v2` | — | 0.275 | **0.331** | 0.475 |
| `matbench_Kim_hse_v2` | — | 0.319 | 0.515 | **0.368** |
| `matbench_noemd_MEGNet_Kim_hse_v2` | — | 0.215 | 0.337 | 0.378 |

В тесте MEGNet 410 металлов (щель 0). В noemd и Kim металлов нет, поэтому модели,
обученные только на них, металлам дают ненулевую щель.
Модели v1 проверены на случайном разбиении, их числа с таблицей не сравнимы:
`matbench_pbe_v1` — 0.127 (тест фолда 0 Matbench), `matbench_noemd_hse_v1` — 0.148
(случайный тест noemd, 1 321 материал).

### Как выбраны данные и что значит честность разбиения v2

Материалы считаются совпадающими, если совпадает приведённая формула, и формула
целиком лежит в одной части выборки. В проверку и тест попали только формулы,
которых модель не видела ни при предобучении энкодера (MAD), ни при обучении PBE,
ни при обучении HSE; формулы, общие для разных наборов, лежат в одной и той же
части. Для набора Kim условие про MAD ослаблено, и из него взяты только немагнитные
материалы без спин-орбиты. Все модели HSE v2 начинают с энкодера `matbench_pbe_v2`.

Смешанная модель `matbench_noemd_MEGNet_Kim_hse_v2` обучена на всех обучающих
материалах трёх наборов (4 535 + 4 311 + 6 931) без удаления дубликатов: материал,
который есть в нескольких наборах, входит в обучение с меткой из каждого из них.
Проверка вариантов с удалением противоречивых пар и с усреднением меток дубликатов
выигрыша не дала. Тест каждого набора взят целиком.

Наборы: [Matbench mp_gap](https://matbench.materialsproject.org/Leaderboards%20Per-Task/matbench_v0.1_matbench_mp_gap/); noemd (6 613
кристаллов, PBE и HSE); MEGNet HSE ([Chen et al., 2021](https://doi.org/10.1038/s43588-020-00002-x));
Kim et al. ([Sci. Data, 2020](https://doi.org/10.1038/s41597-020-00723-8)).

## Это вторая релизная версия 

