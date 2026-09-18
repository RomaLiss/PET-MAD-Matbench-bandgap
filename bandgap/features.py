"""
Кэш признаков замороженного энкодера.

Зачем
---------------
На первой стадии обучения энкодер PET-MAD заморожен. Раз его веса не меняются,
то и признаки, которые он выдаёт для конкретной структуры, каждую эпоху
получаются одни и те же. 

Формат хранения
---------------

    values   (всего_атомов, 1280)   все признаки подряд, одним массивом
    offsets  (структур + 1,)        где начинается каждая структура

Признаки структуры номер i лежат в строках values[offsets[i] : offsets[i+1]] 
(это тоже разработка гпт, я сам не знал, как хранить кэш оптимально, так что этот раздел по большей части ее).
Такой формат позволяет доставать произвольный батч одной операцией выборки,
без цикла по структурам на стороне Python.

3.18 млн атомов * 1280 признаков - примерно 8.2 ГБ 

Выбран float16, потому что вдвое меньший объём позволяет держать кэш прямо в
видеопамяти (8.2 ГБ против 16.3), а это принципиально. При хранении в
оперативной памяти выборка батчей сама становится узким местом и скорость чтения падает

"""

import os

import torch
from metatomic.torch import ModelOutput
from torch.amp import autocast
from tqdm import tqdm

CACHE_DTYPE = torch.float16
CACHE_FORMAT_VERSION = 2


class FeatureCache:
    """Предпосчитанные признаки энкодера в разрядном формате.

    Аннотации типов здесь расставлены намеренно: разрядное хранение — самое
    неочевидное место во всём коде, и по одним лишь именам переменных
    невозможно понять, что `offsets` индексирует строки в `values`, а не
    структуры. Ошибка в этом месте не вызовет исключения, а молча перемешает
    атомы между структурами.
    """

    def __init__(self, values: torch.Tensor, offsets: torch.Tensor, meta: dict | None = None):
        """
        values : (всего_атомов, размерность) — признаки всех структур подряд
        offsets: (число_структур + 1,) int64 — границы структур внутри values
        meta   : чем и как собран кэш; используется для проверки при загрузке
        """
        self.values = values
        self.offsets = offsets
        # Число атомов в каждой структуре. Хранится отдельно, потому что
        # вычисляется один раз, а используется на каждом батче.
        self.counts = offsets[1:] - offsets[:-1]
        self.meta = meta or {}

    def __len__(self) -> int:
        """Число структур в кэше (offsets содержит на одну границу больше)."""
        return self.offsets.numel() - 1

    @property
    def device(self) -> torch.device:
        """Где сейчас лежит кэш: в оперативной памяти или в видеопамяти.

        Проверять это приходится потому, что кэш переезжает туда-обратно: на
        первой стадии он нужен в видеопамяти ради скорости, а перед второй его
        оттуда убирают, чтобы освободить место обучению.
        """
        return self.values.device

    @property
    def dim(self) -> int:
        """Размерность признаков одного атома."""
        return self.values.shape[1]

    def nbytes(self) -> int:
        """Размер кэша в байтах.

        `numel` — сколько всего чисел, `element_size` — сколько байт занимает
        одно. Для нашего набора это 8.2 ГБ при хранении в float16 и вдвое
        """
        return self.values.numel() * self.values.element_size()

    # ------------------------------------------------------------------
    def to(self, device) -> "FeatureCache":
        """Переносит кэш на устройство (в видеопамять на время стадии 1).

        На видюхе выборка превращается в одну операцию index_select и
        обходится практически бесплатно.

        Перед второй стадией кэш нужно вернуть обратно (`.to("cpu")`) —
        видеопамять понадобится под обучение энкодера.
        """
        self.values = self.values.to(device)
        self.offsets = self.offsets.to(device)
        self.counts = self.counts.to(device)
        return self

    # ------------------------------------------------------------------
    @classmethod
    def build(cls, calc, atoms_list, *, encoder_id: str, batch_size: int = 64,
              amp: bool = True, dtype: torch.dtype = CACHE_DTYPE,
              desc: str = "сборка кэша признаков") -> "FeatureCache":
        """Прогоняет энкодер по всем структурам и складывает признаки в кэш.

        encoder_id — строка, однозначно определяющая энкодер (имя, размер,
        версия). Записывается в файл и проверяется при загрузке: признаки от
        другого энкодера внешне неотличимы, но бессмысленны.

        Пиковый расход оперативной памяти примерно вдвое больше итогового
        кэша: в момент склейки одновременно существуют список фрагментов и
        собранный из них массив.
        """
        slabs, counts = [], []
        with torch.no_grad():
            for i in tqdm(range(0, len(atoms_list), batch_size), desc=desc):
                batch = atoms_list[i : i + batch_size]
                with autocast(device_type="cuda", enabled=amp):
                    block = calc.run_model(
                        batch, {"features": ModelOutput(per_atom=True)}
                    )["features"].block()
                    vals = block.values
                    sysidx = block.samples.column("system").to(torch.int64)
                # Строки в блоке упорядочены по номеру структуры, поэтому
                # достаточно запомнить, сколько атомов у каждой.
                slabs.append(vals.to(dtype).cpu())
                counts.append(torch.bincount(sysidx, minlength=len(batch)).cpu())

        values = torch.cat(slabs)
        counts = torch.cat(counts)
        offsets = torch.zeros(len(counts) + 1, dtype=torch.int64)
        torch.cumsum(counts, dim=0, out=offsets[1:])
        meta = {
            "format_version": CACHE_FORMAT_VERSION,
            "encoder_id": encoder_id,
            "dtype": str(dtype),
            "amp": bool(amp),
            "n_structures": int(len(counts)),
            "n_atoms": int(values.shape[0]),
        }
        return cls(values, offsets, meta)

    # ------------------------------------------------------------------
    def gather(self, indices, device):
        """Собирает батч: (признаки, номера структур, число структур).

        Возвращаемые значения:
            feats  (атомов_в_батче, размерность) float32
            seg    (атомов_в_батче,) int64 — номер структуры для каждого атома
            n      число структур в батче

        Реализовано без цикла по структурам. У структур разное число атомов,
        поэтому нужна разрядная выборка: сначала арифметикой строится плоский
        список номеров строк, потом всё достаётся одним index_select.

        Признаки приводятся к float32 независимо от типа хранения — именно
        такой тип выдаёт сам энкодер, и остальной код рассчитывает на него.
        """
        dev = self.values.device
        idx = torch.as_tensor(indices, dtype=torch.int64, device=dev)
        starts = self.offsets[idx]
        counts = self.counts[idx]

        total = int(counts.sum())
        # Начало каждой структуры в выходном массиве: кумулятивная сумма,
        # сдвинутая на одну позицию (эксклюзивная).
        out_start = torch.cumsum(counts, 0) - counts
        # Для каждого атома выходного массива — номер структуры, к которой он
        # относится. Это же значение служит индексом при агрегации в пулинге.
        seg = torch.repeat_interleave(torch.arange(idx.numel(), device=dev), counts)
        # Номер строки в values: начало структуры плюс смещение атома внутри неё.
        atom_rows = starts[seg] + (torch.arange(total, device=dev) - out_start[seg])

        feats = self.values.index_select(0, atom_rows).to(device).float()
        return feats, seg.to(device), idx.numel()

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Сохраняет кэш на диск вместе с метаданными.

        Метаданные (`meta`) сохраняются в том же файле и потом сверяются при
        загрузке. Без них кэш от другой версии энкодера подошёл бы по размеру и
        загрузился молча — см. `load_or_build`.

        Тензоры переносятся на процессор перед записью: файл, сохранённый прямо
        из видеопамяти, при загрузке потребовал бы видеокарту с тем же номером.
        """
        directory = os.path.dirname(path)
        if directory:                      # путь без каталога — сохраняем рядом
            os.makedirs(directory, exist_ok=True)
        torch.save(
            {"values": self.values.cpu(), "offsets": self.offsets.cpu(), "meta": self.meta},
            path,
        )

    @classmethod
    def load(cls, path: str) -> "FeatureCache":
        """Читает кэш с диска БЕЗ проверки пригодности.

        Проверку делает `load_or_build`. Этот метод оставлен отдельно для
        тестов, где нужно прочитать файл как есть и сравнить содержимое.

        `weights_only=True` — защита: запрещает файлу исполнять произвольный код
        при загрузке. Обычный `torch.load` этого не запрещает.
        """
        blob = torch.load(path, map_location="cpu", weights_only=True)
        return cls(blob["values"], blob["offsets"], blob.get("meta", {}))

    # ------------------------------------------------------------------
    @classmethod
    def load_or_build(cls, path, calc, atoms_list, *, encoder_id: str,
                      batch_size: int = 64, amp: bool = True,
                      dtype: torch.dtype = CACHE_DTYPE) -> "FeatureCache":
        """
        Читает подходящий кэш с диска, иначе собирает и сохраняет новый.
        """
        expected = {
            "format_version": CACHE_FORMAT_VERSION,
            "encoder_id": encoder_id,
            "dtype": str(dtype),
            "amp": bool(amp),
            "n_structures": len(atoms_list),
        }

        if path and os.path.exists(path):
            cache = cls.load(path)
            mismatch = [
                f"{k}: в файле {cache.meta.get(k, '<нет>')!r}, ожидалось {v!r}"
                for k, v in expected.items()
                if cache.meta.get(k) != v
            ]
            if not mismatch:
                print(f"кэш признаков: загружено {len(cache)} структур "
                      f"({cache.nbytes() / 1e9:.1f} ГБ) из {path}")
                return cache
            print(f"кэш признаков в {path} не подходит, пересобираю:")
            for line in mismatch:
                print(f"    {line}")

        cache = cls.build(calc, atoms_list, encoder_id=encoder_id,
                          batch_size=batch_size, amp=amp, dtype=dtype)
        print(f"кэш признаков: собрано {len(cache)} структур "
              f"({cache.nbytes() / 1e9:.1f} ГБ)")
        if path:
            cache.save(path)
            print(f"кэш признаков: сохранён в {path}")
        return cache
