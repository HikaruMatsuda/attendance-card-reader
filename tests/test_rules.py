import cv2
import numpy as np
import pytest

import digits
import idrule
import ocr


@pytest.mark.parametrize("sid,ok", [
    ("A2500000", True), ("H2012345", True), ("H2699999", True),
    ("A2500000", False),  # 入学年度が範囲外
    ("A2500000", False),
    ("A2500000", False),  # 英字が A/H 以外
    ("A2500000", False),   # 桁不足
    ("A2500000", False),
])
def test_is_valid(sid, ok):
    assert idrule.is_valid(sid) is ok


def test_decode_year_picks_best_valid_year():
    p_tens = np.full(10, 0.01); p_tens[7] = 0.9; p_tens[2] = 0.09  # 7 と読んだが、ルール上 2 しかあり得ない
    p_ones = np.full(10, 0.01); p_ones[5] = 0.8; p_ones[8] = 0.11
    year, c_tens, c_ones = ocr._decode_year(p_tens, p_ones)
    assert year == "25" and c_tens == 1.0 and c_ones > 0.8


def test_decode_year_rejects_out_of_range_ones_digit():
    p_tens = np.full(10, 0.0); p_tens[2] = 1.0
    p_ones = np.full(10, 0.01); p_ones[8] = 0.6; p_ones[6] = 0.3  # 28 は範囲外 → 26
    assert ocr._decode_year(p_tens, p_ones)[0] == "26"


def _hand(ch: str, slant: float = 0.0, seed: int = 0) -> np.ndarray:
    """手書き風の A/H を描いたマス画像（白地黒字）。"""
    rng = np.random.default_rng(seed)
    img = np.full((90, 80), 255, np.uint8)
    j = lambda x, y: (int(x + slant * (70 - y) + rng.normal(0, 1.5)), int(y + rng.normal(0, 1.5)))
    strokes = {
        "A": [((22, 75), (40, 15)), ((40, 15), (58, 75)), ((29, 52), (51, 52))],
        "H": [((24, 15), (24, 75)), ((56, 15), (56, 75)), ((24, 45), (56, 45))],
    }[ch]
    for a, b in strokes:
        cv2.line(img, j(*a), j(*b), 0, 3)
    return img


@pytest.mark.parametrize("slant", [0.0, 0.15, -0.1])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_classify_letter_synthetic(slant, seed):
    assert digits.classify_letter(_hand("A", slant, seed))[0] == "A"
    letter, conf = digits.classify_letter(_hand("H", slant, seed))
    assert letter == "H" and conf >= 0.7
