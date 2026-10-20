"""机に 41 枚並べた実写（tests/fixtures/many.jpg）での回帰テスト。

正解は写真から目視で書き起こしたもの（読み取りに自信のないカードは除外）。
写真は個人情報を含むため git 管理外。無ければスキップする。
"""

from pathlib import Path

import cv2
import pytest

import cards
import idrule
import ocr

PHOTO = Path(__file__).parent / "fixtures" / "many.jpg"
pytestmark = pytest.mark.skipif(not PHOTO.exists(), reason="many.jpg がありません")

# 検出順（上→下、左→右）の番号 → 学籍番号
TRUTH = {1:"A2500000",2:"A2500000",3:"A2500000",4:"A2500000",5:"A2500000",6:"A2500000",7:"A2500000",
 8:"A2500000",9:"A2500000",10:"A2500000",11:"A2500000",12:"A2500000",14:"A2500000",15:"A2500000",
 16:"A2500000",17:"A2500000",18:"A2500000",19:"A2500000",20:"A2500000",21:"A2500000",22:"A2500000",
 23:"A2500000",24:"A2500000",26:"A2500000",27:"A2500000",28:"A2500000",29:"A2500000",30:"A2500000",
 31:"A2500000",33:"A2500000",34:"A2500000",35:"A2500000",36:"A2500000",39:"A2500000",40:"A2500000"}


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
