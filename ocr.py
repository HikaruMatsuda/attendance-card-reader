"""学籍番号の認識エンジン（マス単位の数字CNN / Indra OCR）。"""

from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np

import cards
import digits

ENGINE_CNN = "数字CNN（マス単位・推奨）"
ENGINE_INDRA = "Indra OCR（datalab-to/indra-ocr）"
ENGINES = [ENGINE_CNN, ENGINE_INDRA]

_indra = None


@dataclass(frozen=True)
class Reading:
    text: str  # 例: "A2500000"（読めない桁は "?"）
    confs: tuple[float, ...]  # 数字 7 桁それぞれの確信度（不明なら空）

    @property
    def min_conf(self) -> float:
        return min(self.confs) if self.confs else 0.0


def _get_indra():
    global _indra
    if _indra is None:
        import indra_ocr

        _indra = indra_ocr.load(threads=4)
    return _indra


_CONFUSABLE = str.maketrans({"O": "0", "o": "0", "D": "0", "I": "1", "l": "1", "|": "1",
                             "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2", "g": "9", "q": "9"})


def _read_cnn(card: np.ndarray, prefix: str) -> Reading:
    cells = cards.id_cells(card)
    res = [digits.classify(c) for c in cells[1:]]  # 先頭マスは英字（固定）
    return Reading(prefix + "".join(d for d, _ in res), tuple(p for _, p in res))


def _read_indra(card: np.ndarray, prefix: str) -> Reading:
    """マスを間隔を空けて 1 行に連結して Indra に読ませる（単独の 1 文字は検出されないため）。"""
    cells = cards.id_cells(card)[1:]
    big = [cv2.erode(cv2.resize(c, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC),
                     np.ones((3, 3), np.uint8)) for c in cells]
    gap = np.full((big[0].shape[0], 40), 255, np.uint8)
    strip = np.hstack([p for c in big for p in (c, gap)])
    text = _get_indra().ocr(strip).text.translate(_CONFUSABLE)
    found = re.sub(r"\D", "", text)
    found = (found + "?" * 7)[:7]
    return Reading(prefix + found, ())


def recognize(card: np.ndarray, engine: str, prefix: str = "A") -> Reading:
    if engine == ENGINE_INDRA:
        return _read_indra(card, prefix)
    return _read_cnn(card, prefix)
