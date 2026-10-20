"""学籍番号の認識エンジン（マス単位の数字CNN / Indra OCR）。

認識結果は学籍番号のルール（idrule）で絞り込む:
  先頭の英字は A/H、数字の先頭 2 桁（入学年度）は 20〜26 の中から最も確からしいものを選ぶ。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np

import cards
import digits
import idrule

ENGINE_CNN = "数字CNN（マス単位・推奨）"
ENGINE_INDRA = "Indra OCR（datalab-to/indra-ocr）"
ENGINES = [ENGINE_CNN, ENGINE_INDRA]

# これ未満の確信度の桁があるカードは「要確認」にする。
# 誤りを見逃さないことを優先した値（実写 35 枚の評価で、誤りをすべて要確認に回せる最小値）
REVIEW_THRESHOLD = 0.85

_indra = None


@dataclass(frozen=True)
class Reading:
    text: str  # 例: "A2500000"（読めない桁は "?"）
    confs: tuple[float, ...]  # 数字 7 桁それぞれの確信度（不明なら空）
    letter_conf: float = 1.0  # 先頭英字（A/H）の確信度

    @property
    def min_conf(self) -> float:
        return min((*self.confs, self.letter_conf)) if self.confs else 0.0


def _get_indra():
    global _indra
    if _indra is None:
        import indra_ocr

        _indra = indra_ocr.load(threads=4)
    return _indra


_CONFUSABLE = str.maketrans({"O": "0", "o": "0", "D": "0", "I": "1", "l": "1", "|": "1",
                             "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2", "g": "9", "q": "9"})


def _decode_year(p_tens: np.ndarray | None, p_ones: np.ndarray | None) -> tuple[str, float, float]:
    """入学年度 2 桁を、ルール上あり得る年度（20〜26）の中から同時確率が最大のものに決める。

    確信度はあり得る年度の中での相対確率（ルールで候補が絞られた分だけ高くなる）。
    """
    if p_tens is None or p_ones is None:
        return "??", 0.0, 0.0
    scores = np.array([p_tens[int(y[0])] * p_ones[int(y[1])] for y in idrule.VALID_YEARS])
    if scores.sum() <= 0:
        return "??", 0.0, 0.0
    k = int(scores.argmax())
    conf = float(scores[k] / scores.sum())
    year = idrule.VALID_YEARS[k]
    # 十の位が全年度で同じ（20〜26 なら常に 2）なら、ルールで確定する
    tens_fixed = len({y[0] for y in idrule.VALID_YEARS}) == 1
    return year, (1.0 if tens_fixed else conf), conf


def _read_cnn(card: np.ndarray) -> Reading:
    cells = cards.id_cells(card)
    letter, letter_conf = digits.classify_letter(cells[0])
    ps = [digits.probs(c) for c in cells[1:]]
    year, c_tens, c_ones = _decode_year(ps[0], ps[1])
    rest, rest_conf = "", []
    for p in ps[2:]:
        if p is None:
            rest += "?"
            rest_conf.append(0.0)
        else:
            k = int(p.argmax())
            rest += str(k)
            rest_conf.append(float(p[k]))
    return Reading(letter + year + rest, (c_tens, c_ones, *rest_conf), letter_conf)


def _read_indra(card: np.ndarray) -> Reading:
    """マスを間隔を空けて 1 行に連結して Indra に読ませる（単独の 1 文字は検出されないため）。"""
    cells = cards.id_cells(card)
    letter, letter_conf = digits.classify_letter(cells[0])
    big = [cv2.erode(cv2.resize(c, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC),
                     np.ones((3, 3), np.uint8)) for c in cells[1:]]
    gap = np.full((big[0].shape[0], 40), 255, np.uint8)
    strip = np.hstack([p for c in big for p in (c, gap)])
    text = _get_indra().ocr(strip).text.translate(_CONFUSABLE)
    found = re.sub(r"\D", "", text)
    found = (found + "?" * idrule.N_DIGITS)[: idrule.N_DIGITS]
    return Reading(letter + found, (), letter_conf)


def recognize(card: np.ndarray, engine: str = ENGINE_CNN) -> Reading:
    if engine == ENGINE_INDRA:
        return _read_indra(card)
    return _read_cnn(card)


def _digit_likeness(card: np.ndarray) -> float:
    """数字らしさ（年度以外の 5 桁の平均確信度）。逆さのカードでは数字として読めず低くなる。"""
    confs = _read_cnn(card).confs[2:]  # 年度 2 桁はルールで底上げされるので使わない
    return float(np.mean(confs)) if confs else 0.0


def orient_and_read(raw: np.ndarray, engine: str = ENGINE_CNN) -> tuple[np.ndarray, bool | None, Reading]:
    """向きを決めて読み取る → (向き補正済みカード, 反転したか(None=不確定), 読み取り結果)。

    基本は表罫線の位置で判定し、両方の向きでの「数字らしさ」でも確かめる。
    罫線の判定と中身が強く食い違うときは中身を優先し、罫線で決まらないときは中身で決める。
    """
    card, flipped = cards.orient_card(raw)
    other = cv2.rotate(card, cv2.ROTATE_180)
    margin = _digit_likeness(card) - _digit_likeness(other)
    if margin < -0.25 or (flipped is None and margin < -0.1):
        card, flipped = other, not flipped
    elif flipped is None and margin > 0.1:
        flipped = False
    return card, flipped, recognize(card, engine)
