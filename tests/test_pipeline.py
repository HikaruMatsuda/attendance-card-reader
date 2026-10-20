from pathlib import Path

import cv2
import pytest

import cards
import ocr

SAMPLE = Path(__file__).parent / "fixtures" / "sample.jpg"
# 左上・左下・右の順（読み順）。サンプル写真は個人情報を含むためリポジトリには含めない
TRUTH = ["A2500000", "A2500000", "A2500000"]

pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="sample.jpg がありません")


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
