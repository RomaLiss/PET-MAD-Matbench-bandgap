"""
Папка запуска: всё, что производит один прогон, лежит в одном месте.
(здесь тоже сильно помогла гпт)
СОСТАВ ПАПКИ
------------
    runs/<дата_время>_<имя>/
        config.yaml             снимок применённой конфигурации
        metrics.csv             метрики по эпохам
        stdout.log              перехваченный вывод в консоль
        checkpoints/            веса модели

"""

import os
import sys
from datetime import datetime

import torch


def make_run_dir(root, name=None):
    """Создаёт каталог runs/<дата_время>_<имя>/ и возвращает путь к нему."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    dirname = f"{stamp}_{name}" if name else stamp
    path = os.path.join(root, dirname)
    os.makedirs(os.path.join(path, "checkpoints"), exist_ok=True)
    return path


class Tee:
    """Дублирует вывод консоли в файл, чтобы он пережил завершение прогона.

    Строки, затёртые возвратом каретки, в файл НЕ попадают. Индикатор прогресса
    перерисовывает себя тысячи раз за эпоху, и без фильтрации он погребал под
    собой те десять строк, ради которых лог и ведётся: за пять минут работы
    файл разрастался до 48 КБ почти целиком из мусора.
    """

    def __init__(self, path):
        self.file = open(path, "a", buffering=1, encoding="utf-8")  # noqa: SIM115 — закрывается в __exit__
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        self._pending = ""

    def __enter__(self):
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, *exc):
        if self._pending.strip():
            self.file.write(self._pending + "\n")
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        self.file.close()

    def write(self, data):
        """Пишет в консоль как есть, а в файл — только уцелевшие строки.

        Индикатор прогресса перерисовывает строку возвратом каретки `\r`: он
        возвращает курсор в начало и пишет поверх. На экране видна одна
        обновляющаяся строка, но в поток уходят все тысячи вариантов.

        Поэтому текст копится в `_pending` до перевода строки, и в файл идёт
        только то, что осталось ПОСЛЕ последнего `\r`, — то есть итоговое
        состояние строки. Без этого лог за пять минут разрастался до 48 КБ
        почти целиком из мусора.
        """
        self._stdout.write(data)

        # Копим по строкам и оставляем только то, что уцелело после последнего
        # возврата каретки — то есть итоговое состояние строки.
        self._pending += data
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            line = line.split("\r")[-1]
            if line.strip():
                self.file.write(line + "\n")
        # Защита от индикатора, который вообще не переводит строку.
        if len(self._pending) > 8192:
            self._pending = self._pending.split("\r")[-1]

    def flush(self):
        """Сбрасывает оба буфера на диск и на экран.

        Метод обязателен: подменяя `sys.stdout`, класс обязан поддерживать тот
        же набор методов, что и настоящий поток, иначе чужой код, вызывающий
        `flush`, упадёт.
        """
        self._stdout.flush()
        self.file.flush()

    def isatty(self):
        """Отвечает, подключён ли вывод к терминалу.

        Об этом спрашивает tqdm. Честный ответ (False, когда вывод
        перенаправлен в файл) заставляет его печатать заметно меньше:
        перерисовок не будет, останутся только итоговые строки.
        """
        # У обычного терминала метод isatty есть, у некоторых подменённых
        # потоков вывода его нет. Во втором случае считаем, что терминала нет.
        if not hasattr(self._stdout, "isatty"):
            return False
        return self._stdout.isatty()


class MetricsLog:
    """Файл метрик по эпохам, дописываемый по мере обучения.

    Скорости обучения всех групп параметров пишутся в одну колонку через «|»,
    чтобы схема файла не зависела от того, сколько групп определила конкретная
    стадия. У первой стадии группа одна, у второй — три.
    """

    COLUMNS = [
        "fold", "stage", "epoch",
        "train_rmse", "train_mae", "val_rmse", "val_mae", "val_std",
        "lrs", "seconds",
    ]

    def __init__(self, path):
        self.path = path
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(",".join(self.COLUMNS) + "\n")

    def write_row(self, *, fold, stage, epoch, train_rmse, train_mae, val_rmse,
                  val_mae, val_std, lrs, seconds):
        """Дописывает строку метрик за одну эпоху.

        Файл открывается и закрывается на каждой строке. Это чуть медленнее
        буферизованной записи, но зато данные оказываются на диске сразу: если
        прогон оборвётся на середине, журнал предыдущих эпох останется целым.
        Для эпохи длиной в шесть минут накладные расходы неразличимы.

        Все аргументы именованные (звёздочка в сигнатуре запрещает передавать
        их по порядку). Причина простая: аргументов десять, четыре из них —
        похожие друг на друга числа, и перепутанные местами `val_rmse` и
        `val_mae` не вызвали бы никакой ошибки.
        """
        row = [
            fold, stage, epoch,
            f"{train_rmse:.4f}", f"{train_mae:.4f}",
            f"{val_rmse:.4f}", f"{val_mae:.4f}", f"{val_std:.4f}",
            "|".join(f"{lr:.2e}" for lr in lrs), f"{seconds:.1f}",
        ]
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(",".join(str(x) for x in row) + "\n")


class Checkpointer:
    """Сохранение и загрузка весов модели.

    ДВА ЛУЧШИХ ЧЕКПОИНТА, А НЕ ОДИН
    -------------------------------
    Стадия сохраняет лучшую эпоху по RMSE И лучшую эпоху по MAE. Это разные
    эпохи: RMSE сильнее наказывает крупные промахи, поэтому его минимум
    смещён к эпохам без выбросов, а не к эпохам, где модель в среднем точнее.
    Замерено на 29 стадиях архива: расхождение доходило до 0.0043 эВ MAE при
    разбросе между повторами одного прогона 0.0009 эВ, то есть было сравнимо
    с улучшениями, ради которых мы ставим опыты.

    Хранить оба стоит только места на диске, поэтому выбор не приходится
    делать заранее: обе точки остаются, а какую загружать — решает
    `train.selection_metric`.

    ИМЕНА ФАЙЛОВ
    ------------
        fold_N_best_SM.pth           лучший по RMSE
        fold_N_best_SM_mae.pth       лучший по MAE
        last_checkpoint_fold_N.pth   последняя эпоха

    Имя для RMSE намеренно оставлено без суффикса: оно совпадает со схемой
    исходного проекта, и его чекпоинты читаются здесь без переименования —
    именно это позволило проверить эквивалентность рефакторинга на реальных
    весах.
    """

    #: Метрики, по которым отбирается лучшая эпоха. Кортеж задаёт и порядок
    #: сохранения, и множество допустимых значений `train.selection_metric`.
    METRICS = ("rmse", "mae")

    def __init__(self, directory):
        self.dir = directory
        os.makedirs(self.dir, exist_ok=True)

    def best_path(self, fold, stage, metric="rmse"):
        """
        Путь к файлу лучшего чекпоинта для заданных фолда, стадии и метрики.
        """
        if metric not in self.METRICS:
            raise ValueError(f"неизвестная метрика отбора {metric!r}, "
                             f"допустимы {self.METRICS}")
        suffix = "" if metric == "rmse" else f"_{metric}"
        return os.path.join(self.dir, f"fold_{fold}_best_S{stage}{suffix}.pth")

    def last_path(self, fold):
        """Путь к весам ПОСЛЕДНЕЙ эпохи (не лучшей).

        Нужен, чтобы после прерванного прогона было видно, на чём он
        остановился. Для получения результата берут лучший чекпоинт, а не этот.
        """
        return os.path.join(self.dir, f"last_checkpoint_fold_{fold}.pth")

    def save_best(self, model, *, fold, stage, metric="rmse"):
        """Сохраняет веса как лучший результат стадии по указанной метрике.

        `state_dict()` — это словарь «имя параметра → тензор». Сохраняются
        только веса, без структуры модели: чтобы их прочитать, нужно сначала
        собрать такую же модель. Поэтому имена атрибутов в `PETBandgap`
        (`encoder`, `pooling`, `head`) менять нельзя — от них зависят ключи в
        файле и совместимость со всеми уже сохранёнными чекпоинтами.
        """
        torch.save(model.state_dict(), self.best_path(fold, stage, metric))

    def save_last(self, model, *, fold):
        """Сохраняет веса текущей эпохи, перезаписывая предыдущие."""
        torch.save(model.state_dict(), self.last_path(fold))

    def load_best(self, model, *, fold, stage, device, metric="rmse"):
        """Загружает лучшие веса В УЖЕ СОБРАННУЮ модель и возвращает её.

        `map_location=device` кладёт веса сразу на нужное устройство. Без этого
        они сначала попали бы туда, откуда сохранялись, и на другой машине
        загрузка упала бы.

        `load_state_dict` по умолчанию требует ТОЧНОГО совпадения набора ключей.
        Это и нужно: расхождение означает, что модель собрана иначе, чем та,
        чьи веса читаем, и лучше узнать об этом сразу.
        """
        path = self.best_path(fold, stage, metric)
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        return model
