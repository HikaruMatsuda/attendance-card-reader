"""机に 41 枚並べた実写（tests/fixtures/many.jpg）での回帰テスト。

正解は写真から目視で書き起こしたもの（読み取りに自信のないカードは除外）。
写真と正解（truth.json）は個人情報を含むため git 管理外。無ければスキップする。
"""

import json
from pathlib import Path

import cv2
import pytest

import cards
import idrule
import ocr

PHOTO = Path(__file__).parent / "fixtures" / "many.jpg"

# 検出順（上→下、左→右）の番号 → 学籍番号。写真から書き起こした正解で、個人情報なので git 管理外
TRUTH_FILE = Path(__file__).parent / "fixtures" / "truth.json"
TRUTH = (
    {int(k): v for k, v in json.loads(TRUTH_FILE.read_text())["many.jpg"].items()}
    if TRUTH_FILE.exists() else {}
)
pytestmark = pytest.mark.skipif(not PHOTO.exists() or not TRUTH, reason="many.jpg / truth.json がありません")


@pytest.fixture(scope="module")
def results():
    img = cv2.imread(str(PHOTO))
    boxes = cards.detect_cards(img, cards.DetectParams())
    return len(boxes), {i: ocr.orient_and_read(cards.warp_card(img, b)) for i, b in enumerate(boxes, 1)}


def test_detects_all_cards(results):
    assert results[0] == 41


def test_digit_accuracy(results):
    _, res = results
    ok = sum(a == b for i, t in TRUTH.items() for a, b in zip(res[i][2].text[1:], t[1:]))
    assert ok / (7 * len(TRUTH)) >= 0.87  # 現状 89%


def test_every_wrong_reading_is_flagged_for_review(results):
    _, res = results
    missed = [
        (i, r.text, t) for i, t in TRUTH.items()
        for _, fl, r in [res[i]]
        if r.text != t and not (r.min_conf < ocr.REVIEW_THRESHOLD or fl is None or not idrule.is_valid(r.text))
    ]
    assert missed == []
