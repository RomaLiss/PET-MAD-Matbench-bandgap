#!/usr/bin/env python3
"""Предсказывает ширину запрещённой зоны для одной структуры.

Запуск:
    python3 tools/predict.py структура.cif
    python3 tools/predict.py структура.cif --checkpoint путь/к/весам.pth
    python3 tools/predict.py структура.cif --device cuda

structure — любой файл, который умеет читать ASE: CIF, POSCAR, xyz и другие
(формат определяется по расширению, см. ase.io.read).

"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

os.environ.setdefault("VESIN_CUDA_MAX_PAIRS_PER_POINT", "1024")

from ase.io import read  # noqa: E402 — после установки переменной окружения

from bandgap.inference import DEFAULT_CHECKPOINT, BandgapPredictor  # noqa: E402

ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
ap.add_argument("structure", help="файл структуры")
ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                help="путь к весам (по умолчанию — лучший чекпоинт фолда 0)")
ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
args = ap.parse_args()

atoms = read(args.structure)
print(f"структура: {atoms.get_chemical_formula()}, атомов {len(atoms)}")

predictor = BandgapPredictor(checkpoint=args.checkpoint, device=args.device)
print(f"предсказанная ширина запрещённой зоны: {predictor(atoms):.4f} эВ")
