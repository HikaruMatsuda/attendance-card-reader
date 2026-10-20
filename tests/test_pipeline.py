import json
from pathlib import Path

import cv2
import pytest

import cards
import ocr

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE = FIXTURES / "sample.jpg"
TRUTH_FILE = FIXTURES / "truth.json"  # 写真から書き起こした正解。個人情報なので git 管理外

pytestmark = pytest.mark.skipif(
    not (SAMPLE.exists() and TRUTH_FILE.exists()), reason="sample.jpg / truth.json がありません"
)
TRUTH = json.loads(TRUTH_FILE.read_text())["sample.jpg"] if TRUTH_FILE.exists() else []


@pytest.fixture(scope="module")
def sample_cards():
    img = cv2.imread(str(SAMPLE))
    boxes = cards.detect_cards(img, cards.DetectParams())
    return [cards.orient_card(cards.warp_card(img, b)) for b in boxes]


def test_detects_all_cards(sample_cards):
    assert len(sample_cards) == 3


def test_orientation_is_upright(sample_cards):
    assert all(flipped is False for _, flipped in sample_cards)


def test_upside_down_is_corrected(sample_cards):
    card, _ = sample_cards[0]
    fixed, flipped = cards.orient_card(cv2.rotate(card, cv2.ROTATE_180))
    assert flipped is True and fixed.shape == card.shape


def test_id_cells_split_into_eight(sample_cards):
    assert all(len(cards.id_cells(c)) == cards.N_CELLS for c, _ in sample_cards)


def test_cnn_digit_accuracy(sample_cards):
    # 検出順は写真の配置依存なので、各カードを最も近い正解と照合して桁一致数を数える
    total = sum(
        max(sum(a == b for a, b in zip(ocr.recognize(c, ocr.ENGINE_CNN).text, t)) for t in TRUTH)
        for c, _ in sample_cards
    )
    assert total >= 17  # 24 桁中 17 桁以上（現状は 21 桁）


def test_id_cells_survive_degenerate_cards():
    # 枠が途中で切れた・カードでない領域でも例外にならず 8 マス返す
    import numpy as np

    blank = np.full((cards.CARD_H, cards.CARD_W, 3), 200, np.uint8)
    assert len(cards.id_cells(blank)) == cards.N_CELLS
    noise = np.random.default_rng(0).integers(0, 255, (cards.CARD_H, cards.CARD_W, 3), dtype=np.uint8)
    assert len(cards.id_cells(noise)) == cards.N_CELLS
    assert ocr.recognize(noise, ocr.ENGINE_CNN).text.startswith("A")
