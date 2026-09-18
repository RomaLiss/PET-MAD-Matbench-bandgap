"""
Предсказание band_gap для новых структур.

    from bandgap.inference import BandgapPredictor
    from ase.io import read

    predictor = BandgapPredictor()             
    print(predictor(read("структура.cif")))    

Объект вызывается как функция — тем же приёмом, что `AtomsForward` в engine.py. 
Почему НЕ калькулятор ASE, как у PET-MAD. У ASE фиксированный набор
свойств — энергия, силы. band_gap нет.
Вернуть её из метода `get_potential_energy()` значило бы соврать именем и запутаться. 
По той же причине сам PET-MAD отдаёт свои признаки через `run_model`, а не через
стандартный интерфейс ASE.
"""
import os

import torch
from ase import Atoms
from metatomic.torch.ase_calculator import MetatomicCalculator

from bandgap import engine
from bandgap.config import load_config
from bandgap.model import build_model

_HERE = os.path.dirname(os.path.abspath(__file__))

#: Конфигурация задаёт АРХИТЕКТУРУ — размер энкодера, размерность признаков.
DEFAULT_CONFIG = os.path.join(_HERE, "..", "model", "train_config.yaml")
DEFAULT_CHECKPOINT = os.path.join(_HERE, "..", "model", "fold_0_best_S2_mae.pth")


class BandgapPredictor:
    """Обученная модель, готовая предсказывать. Вызывается как функция.

        predictor = BandgapPredictor()
        gap = predictor(atoms)              
        gaps = predictor([a1, a2, a3])      

    Список считается одним батчем, а не по очереди, поэтому для многих
    структур передавать список заметно быстрее, чем вызывать в цикле.
    """

    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, config=DEFAULT_CONFIG,
                 device="cpu"):
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(
                f"чекпоинт не найден: {checkpoint}\n"
                f"укажите свой: BandgapPredictor(checkpoint='путь/к/весам.pth')"
            )
        cfg = load_config(config)
        model = build_model(cfg, device)

        state = torch.load(checkpoint, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"чекпоинт не подходит к этой архитектуре: не хватает "
                f"{len(missing)} параметров, лишних {len(unexpected)}. "
                f"Проверьте, что config соответствует тому, чем обучен чекпоинт."
            )
        model.eval()

        self.model = model
        self.calc = MetatomicCalculator(model.encoder, device=device)
        self.device = device

    def __call__(self, atoms):
        """Ширина запрещённой зоны, эВ.

        atoms — объект ASE Atoms либо список таких объектов. Вернётсяиличисло или массив.
        """
        single = isinstance(atoms, Atoms)
        atoms_list = [atoms] if single else list(atoms)
        preds = engine.predict(self.model, self.calc, atoms_list,
                               amp=(self.device == "cuda"))
        return float(preds[0]) if single else preds
