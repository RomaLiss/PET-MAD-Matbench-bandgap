"""
Наборы данных и загрузчики батчей.

ДВА ВИДА ЗАГРУЗЧИКОВ
--------------------
Их два, потому что у двух стадий обучения разные источники признаков.

    get_loader        отдаёт структуры (объекты ASE Atoms). Нужен второй
                      стадии, где энкодер обучается и признаки приходится
                      считать заново каждый шаг.
    get_index_loader  отдаёт номера структур в кэше признаков. Нужен первой
                      стадии, где энкодер заморожен и признаки уже посчитаны.
"""

import torch
from pymatgen.io.ase import AseAtomsAdaptor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class StructureDataset(Dataset):
    """Пары «структура (ASE Atoms) → целевое значение»."""

    def __init__(self, structures, targets):
        assert len(structures) == len(targets), (
            f"длины не совпадают: структур {len(structures)}, целей {len(targets)}"
        )
        self.structures = structures
        self.targets = targets

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        return self.structures[idx], self.targets[idx]


def collate_structures(batch):
    """
    Собирает батч: структуры остаются списком, цели — тензором (N, 1).
    Объекты ASE нельзя сложить в тензор, поэтому они передаются списком —
    энкодер принимает именно такой вид.
    """
    atoms = [item[0] for item in batch]
    targets = torch.tensor([item[1] for item in batch], dtype=torch.float32).unsqueeze(1)
    return atoms, targets


def get_loader(structures, targets, batch_size, num_workers, *, shuffle):
    """
    Загрузчик для обучения по структурам (вторая стадия).
    """
    return DataLoader(
        StructureDataset(structures, targets),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_structures,
        num_workers=num_workers,
        pin_memory=True,
    )


class IndexDataset(Dataset):
    """Пары «номер структуры в кэше → целевое значение»."""

    def __init__(self, cache_indices, targets):
        assert len(cache_indices) == len(targets), (
            f"длины не совпадают: индексов {len(cache_indices)}, целей {len(targets)}"
        )
        self.cache_indices = cache_indices
        self.targets = targets

    def __len__(self):
        return len(self.cache_indices)

    def __getitem__(self, idx):
        return self.cache_indices[idx], self.targets[idx]


def collate_indices(batch):
    """Собирает батч из номеров структур и целей."""
    indices = [item[0] for item in batch]
    targets = torch.tensor([item[1] for item in batch], dtype=torch.float32).unsqueeze(1)
    return indices, targets


def get_index_loader(cache_indices, targets, batch_size, *, shuffle):
    """
    Загрузчик для обучения по кэшу признаков (первая стадия).
    """
    return DataLoader(
        IndexDataset(cache_indices, targets),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_indices,
        num_workers=0,
    )


def convert_to_ase_list(pmg_structures, desc="pymatgen → ASE"):
    """Переводит структуры pymatgen в объекты ASE Atoms."""
    adaptor = AseAtomsAdaptor()
    return [adaptor.get_atoms(s) for s in tqdm(pmg_structures, desc=desc)]


def load_matbench_task(task_name="matbench_mp_gap"):
    """Загружает задачу Matbench и один раз переводит все структуры в ASE.

    Возвращает три значения: (mb, task, index_to_ase)
        mb           объект бенчмарка — через него записываются предсказания
                     и сохраняется файл результатов
        task         сама задача: разбиение на фолды, обучающие и тестовые части
        index_to_ase отображение «индекс строки в таблице → структура ASE»

    Перевод в ASE делается заранее и один раз.
    """
    from matbench.bench import MatbenchBenchmark

    mb = MatbenchBenchmark(autoload=False)
    task = mb.tasks_map[task_name]
    task.load()

    ase_list = convert_to_ase_list(
        task.df["structure"].tolist(), desc=f"{task_name} → ASE"
    )
    # (GPT-answer) strict=True: молчаливое расхождение длин здесь означало бы, что структуры
    # и их индексы разъехались, а обнаружилось бы это только по бессмысленным
    # результатам обучения.
    index_to_ase = dict(zip(task.df.index, ase_list, strict=True))
    return mb, task, index_to_ase
