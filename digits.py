"""手書き数字 1 桁の分類（MNIST ONNX）。マス単位の画像を MNIST 形式に整えて推論する。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = Path(__file__).parent / "models" / "digits_aug.onnx"
DESKEW = True
_session: ort.InferenceSession | None = None


def _get_session() -> ort.InferenceSession:
    global _session
    if _session is None:
        _session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    return _session


def _deskew(img: np.ndarray) -> np.ndarray:
    """二次モーメントから傾き（せん断）を推定して直立に戻す（MNIST 標準の前処理）。"""
    m = cv2.moments(img)
    if abs(m["mu02"]) < 1e-2:
        return img
    skew = m["mu11"] / m["mu02"]
    size = img.shape[0]
    mat = np.float32([[1, skew, -0.5 * size * skew], [0, 1, 0]])
    return cv2.warpAffine(img, mat, img.shape[::-1], flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)


def to_mnist(cell_gray: np.ndarray) -> np.ndarray | None:
    """白地黒字のマス画像 → 28x28 の黒地白字（直立・重心を中央に）。文字が無ければ None。"""
    ink = (cell_gray < 128).astype(np.uint8) * 255
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))  # 細かいゴミ除去
    ys, xs = np.where(ink > 0)
    if len(ys) < 25:
        return None
    # 傾きは切り詰め前の大きな画像で補正する（縮小後だと精度が落ちる）
    pad = cv2.copyMakeBorder(ink, 30, 30, 60, 60, cv2.BORDER_CONSTANT, value=0)
    if DESKEW:
        pad = _deskew(pad)
    ys, xs = np.where(pad > 0)
    if len(ys) == 0:  # デスキューで線が消えた場合
        return None
    glyph = pad[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    h, w = glyph.shape
    s = 20.0 / max(h, w)
    glyph = cv2.resize(glyph, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    glyph = cv2.dilate(glyph, np.ones((2, 2), np.uint8))  # 細い線を MNIST の太さに寄せる
    canvas = np.zeros((28, 28), np.uint8)
    gh, gw = glyph.shape
    canvas[(28 - gh) // 2 : (28 - gh) // 2 + gh, (28 - gw) // 2 : (28 - gw) // 2 + gw] = glyph
    m = cv2.moments(canvas)
    if m["m00"] > 0:
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        canvas = cv2.warpAffine(canvas, np.float32([[1, 0, 14 - cx], [0, 1, 14 - cy]]), (28, 28))
    return canvas


def probs(cell_gray: np.ndarray) -> np.ndarray | None:
    """0〜9 の確率（長さ 10）。文字が無ければ None。"""
    img = to_mnist(cell_gray)
    if img is None:
        return None
    x = (img.astype(np.float32) / 255.0).reshape(1, 1, 28, 28)
    logits = _get_session().run(None, {"input": x})[0][0]
    e = np.exp(logits - logits.max())
    return e / e.sum()


def classify(cell_gray: np.ndarray) -> tuple[str, float]:
    """(数字 or '?', 確信度 0-1)。"""
    p = probs(cell_gray)
    if p is None:
        return "?", 0.0
    k = int(p.argmax())
    return str(k), float(p[k])


def classify_letter(cell_gray: np.ndarray) -> tuple[str, float]:
    """先頭マスの英字が A か H かを形で判定する → (英字, 確信度)。

    A は上部が頂点 1 か所にすぼまり、H は上部でも 2 本の縦棒が離れている。
    手書きの A は頂点が開くこともあるので、3 つの特徴がすべてそろったときだけ H とし、
    きわどいものは確信度を下げて人の確認に回す。
    """
    ink = (cell_gray < 128).astype(np.uint8)
    ys, xs = np.where(ink)
    if len(ys) < 25:
        return "A", 0.0
    g = ink[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    h, w = g.shape
    q = max(2, h // 4)
    top = cv2.dilate(g[:q], np.ones((1, 3), np.uint8))  # 細かな途切れは無視
    bottom = g[-q:]

    runs, gaps = [], []
    for r in top:
        if not r.any():
            continue
        edges = np.diff(np.r_[0, r, 0].astype(np.int16))  # uint8 のままだと 0-1 が 255 に回り込む
        starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
        runs.append(len(starts))
        if len(starts) >= 2:
            gaps.append((starts[-1] - ends[0]) / w)

    def span(rows: np.ndarray) -> float:
        return float(np.median([np.ptp(np.where(r)[0]) if r.any() else 0 for r in rows]))

    two_posts = bool(runs) and float(np.median(runs)) >= 2
    wide_gap = bool(gaps) and float(np.median(gaps)) >= 0.35
    parallel = span(top) / max(1.0, span(bottom)) >= 0.7
    score = two_posts + wide_gap + parallel
    if score == 3:
        return "H", 0.9
    if score == 2:
        return "A", 0.5  # きわどい → 要確認
    return "A", 0.95 if score == 0 else 0.8
