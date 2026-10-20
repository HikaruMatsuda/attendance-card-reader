"""出席カードの検出・切り出し・向き補正・ID行の抽出（OpenCV）。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# 切り出し後のカードの正規サイズ（横長）。実物は約 2:1 の名刺サイズ
CARD_W, CARD_H = 900, 480


@dataclass
class DetectParams:
    min_area_ratio: float = 0.004  # 画像面積に対するカード最小面積
    max_area_ratio: float = 0.25
    color_thresh: float = 10.0  # 背景色からの Lab 距離のしきい値
    lightness_weight: float = 0.15  # 明度差の重み（小さいほど影に強い）
    dark_cut: float = 0.6  # 背景の明度×この値より暗い画素は除外
    aspect_min: float = 1.4
    aspect_max: float = 2.8
    max_side: int = 1600  # 検出時の縮小サイズ（長辺）


def _estimate_background(lab: np.ndarray) -> np.ndarray:
    """画像全体で最頻の色（=机）を背景色として推定する。"""
    small = cv2.resize(lab, (64, 64), interpolation=cv2.INTER_AREA).reshape(-1, 3)
    q = (small // 8).astype(np.int32)
    keys = q[:, 0] * 10000 + q[:, 1] * 100 + q[:, 2]
    vals, counts = np.unique(keys, return_counts=True)
    mode = vals[counts.argmax()]
    mask = keys == mode
    return np.median(small[mask], axis=0)


def detect_cards(img_bgr: np.ndarray, p: DetectParams) -> list[np.ndarray]:
    """画像中のカードの 4 頂点（元画像座標, float32 (4,2)）を返す。"""
    h, w = img_bgr.shape[:2]
    scale = min(1.0, p.max_side / max(h, w))
    small = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (5, 5), 0)

    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg = _estimate_background(lab)
    # 明度は照明ムラの影響が大きいので a,b（色味）を重視し、明度は弱く効かせる
    diff = lab - bg
    dist = np.sqrt((diff[..., 0] * p.lightness_weight) ** 2 + diff[..., 1] ** 2 + diff[..., 2] ** 2)
    mask = (dist > p.color_thresh).astype(np.uint8)
    # 床や下敷きなど、背景より極端に暗い領域はカードではない
    mask &= (lab[..., 0] > bg[0] * p.dark_cut).astype(np.uint8)
    mask *= 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_img = small.shape[0] * small.shape[1]
    boxes: list[np.ndarray] = []
    for c in contours:
        a = cv2.contourArea(c)
        if not (p.min_area_ratio * area_img <= a <= p.max_area_ratio * area_img):
            continue
        rect = cv2.minAreaRect(c)
        (rw, rh) = rect[1]
        if min(rw, rh) < 1:
            continue
        aspect = max(rw, rh) / min(rw, rh)
        fill = a / (rw * rh)  # 矩形らしさ
        if not (p.aspect_min <= aspect <= p.aspect_max) or fill < 0.75:
            continue
        boxes.append(cv2.boxPoints(rect) / scale)

    # 読み順（上→下、左→右）に並べる
    boxes.sort(key=lambda b: (round(b[:, 1].mean() / (h / 12)), b[:, 0].mean()))
    return [b.astype(np.float32) for b in boxes]


def _order_points(pts: np.ndarray) -> np.ndarray:
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]],
        dtype=np.float32,
    )  # TL, TR, BR, BL


def warp_card(img_bgr: np.ndarray, box: np.ndarray) -> np.ndarray:
    """4 頂点から横長に射影変換して切り出す（向きは未確定=0/180° 不定）。"""
    tl, tr, br, bl = _order_points(box)
    width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    if height > width:  # 縦長なら 90° 回して横長に
        tl, tr, br, bl = tr, br, bl, tl
    dst = np.array([[0, 0], [CARD_W, 0], [CARD_W, CARD_H], [0, CARD_H]], np.float32)
    m = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], np.float32), dst)
    return cv2.warpPerspective(img_bgr, m, (CARD_W, CARD_H), flags=cv2.INTER_CUBIC)


def _ink_mask(card: np.ndarray) -> np.ndarray:
    """印刷の黒い罫線・文字を拾う二値画像（255=インク）。"""
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 12
    )


def horizontal_lines(card: np.ndarray) -> list[int]:
    """表の横罫線の y 座標（上から順）。カード幅の 40% 以上続く線を採用。"""
    ink = _ink_mask(card)
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (int(CARD_W * 0.18), 1))
    lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kern)
    prof = lines.sum(axis=1) / 255.0
    ys = np.where(prof > CARD_W * 0.4)[0]
    groups: list[list[int]] = []
    for y in ys:
        if groups and y - groups[-1][-1] <= 6:
            groups[-1].append(int(y))
        else:
            groups.append([int(y)])
    found = [int(np.mean(g)) for g in groups]
    # カードの縁（外枠）を拾った線は除く
    margin = CARD_H * 0.05
    return [y for y in found if margin < y < CARD_H - margin]


def is_upside_down(card: np.ndarray) -> bool | None:
    """表の上余白＞下余白なら正立。罫線が拾えなければ None。

    正立時は上部に「出席票/月日」の見出しがあり表の上余白が広く、
    下は「上智大学」の 1 行だけで余白が狭い。
    """
    ys = horizontal_lines(card)
    if len(ys) < 3:
        return None
    top, bottom = ys[0], CARD_H - ys[-1]
    if abs(top - bottom) < CARD_H * 0.04:
        return None
    return top < bottom


def orient_card(card: np.ndarray) -> tuple[np.ndarray, bool | None]:
    flipped = is_upside_down(card)
    if flipped:
        card = cv2.rotate(card, cv2.ROTATE_180)
    return card, flipped


def id_row_crop(card: np.ndarray) -> np.ndarray:
    """学籍番号の行（左のラベル列を除く）を切り出す。罫線が拾えれば罫線基準、無ければ比率。"""
    ys = horizontal_lines(card)
    if len(ys) >= 4:
        y0, y1 = ys[1], ys[2]  # 表の罫線は上から 科目名行の下線, 学籍番号行の下線 の順
    else:
        y0, y1 = int(CARD_H * 0.50), int(CARD_H * 0.70)
    pad = 3
    x0 = int(CARD_W * 0.27)  # ラベル「学生番号/ID Number」を避ける
    return card[max(0, y0 + pad) : y1 - pad, x0 : CARD_W - 8]


def remove_grid_lines(row_bgr: np.ndarray) -> np.ndarray:
    """マス目の縦横罫線を白で塗りつぶし、手書き文字だけ残した画像（グレースケール, 白地黒字）。"""
    gray = cv2.cvtColor(row_bgr, cv2.COLOR_BGR2GRAY)
    ink = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )
    h, w = ink.shape
    v = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(h * 0.7)))
    )
    hz = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.3), 1))
    )
    grid = cv2.dilate(cv2.bitwise_or(v, hz), np.ones((3, 3), np.uint8))
    clean = cv2.bitwise_and(ink, cv2.bitwise_not(grid))
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    out = 255 - clean
    return cv2.copyMakeBorder(out, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)


N_CELLS = 8  # 学籍番号は「A」＋数字7桁の 8 マス


def _plausible_edges(edges: list[int], width: int) -> bool:
    """マス境界が 9 本そろい、幅がほぼ均等で十分な大きさかを確認する。"""
    if len(edges) != N_CELLS + 1:
        return False
    widths = np.diff(edges)
    return bool(widths.min() >= width * 0.05 and widths.max() <= widths.min() * 1.8)


def id_cells(card: np.ndarray) -> list[np.ndarray]:
    """学籍番号行をマスごとに分割し、罫線を消した白地黒字のグレースケール画像を返す。"""
    ys = horizontal_lines(card)
    if len(ys) >= 4:
        y0, y1 = ys[1] + 3, ys[2] - 3
    else:
        y0, y1 = int(CARD_H * 0.50) + 3, int(CARD_H * 0.70) - 3
    band = card[y0:y1]
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    ink = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )
    h, w = ink.shape
    vert = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(h * 0.75)))
    )
    prof = vert.sum(axis=0) / 255.0
    xs = np.where(prof > h * 0.7)[0]
    groups: list[list[int]] = []
    for x in xs:
        if groups and x - groups[-1][-1] <= 5:
            groups[-1].append(int(x))
        else:
            groups.append([int(x)])
    edges = [int(np.mean(g)) for g in groups]
    # ラベル列との境界（左端）以降の縦線だけを使う
    edges = [x for x in edges if w * 0.15 < x < w * 0.96]
    if len(edges) >= N_CELLS + 1:
        edges = edges[-(N_CELLS + 1) :]
    if not _plausible_edges(edges, w):  # 罫線が拾えない・ずれている場合は等分で代用
        left = edges[0] if edges and edges[0] < w * 0.5 else int(w * 0.27)
        right = edges[-1] if len(edges) > 1 and edges[-1] > w * 0.6 else w - 8
        edges = [int(left + (right - left) * i / N_CELLS) for i in range(N_CELLS + 1)]

    cells: list[np.ndarray] = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = 7  # 罫線の太さ分だけ内側を使う
        cell_ink = ink[:, a + m : b - m].copy()
        if cell_ink.shape[1] < 10 or cell_ink.shape[0] < 20:  # 空の画像は cv2 がエラーにする
            cells.append(np.full((60, 120), 255, np.uint8))
            continue
        cell_ink[:5] = 0  # 上下の横罫線の名残
        cell_ink[-5:] = 0
        cell_ink = cv2.morphologyEx(cell_ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        img = 255 - cell_ink
        cells.append(cv2.copyMakeBorder(img, 20, 20, 40, 40, cv2.BORDER_CONSTANT, value=255))
    return cells
