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


def _rect_if_card(contour: np.ndarray, p: DetectParams, area_img: int):
    """輪郭がカードらしければ minAreaRect を返す。"""
    a = cv2.contourArea(contour)
    if not (p.min_area_ratio * area_img <= a <= p.max_area_ratio * area_img):
        return None
    rect = cv2.minAreaRect(contour)
    rw, rh = rect[1]
    if min(rw, rh) < 1:
        return None
    aspect = max(rw, rh) / min(rw, rh)
    if p.aspect_min <= aspect <= p.aspect_max and a / (rw * rh) >= 0.75:
        return rect
    return None


def _card_rects(mask: np.ndarray, p: DetectParams, area_img: int, depth: int = 0) -> list:
    """二値マスクからカードの矩形を集める。接してつながった塊は、収縮で分離してから元の大きさに戻して再判定する。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found = []
    for c in contours:
        rect = _rect_if_card(c, p, area_img)
        if rect is not None:
            found.append(rect)
            continue
        # カード 1.5 枚分以上の大きさなら、複数枚が接している可能性がある
        if depth >= 3 or cv2.contourArea(c) < p.min_area_ratio * area_img * 1.5:
            continue
        blob = np.zeros_like(mask)
        cv2.drawContours(blob, [c], -1, 255, cv2.FILLED)
        for r in range(2, 30, 2):
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            n, labels = cv2.connectedComponents(cv2.erode(blob, k))
            if n - 1 >= 2:  # 2 つ以上に分離できた
                for i in range(1, n):
                    part = cv2.dilate((labels == i).astype(np.uint8) * 255, k) & blob
                    found += _card_rects(part, p, area_img, depth + 1)
                break
    return found


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

    area_img = small.shape[0] * small.shape[1]
    boxes: list[np.ndarray] = []
    for rect in _card_rects(mask, p, area_img):
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
    # 低解像度の写真では罫線が点線状に途切れるので、先に横方向の隙間を埋める
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 1)))
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (int(CARD_W * 0.25), 1))
    lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kern)
    prof = lines.sum(axis=1) / 255.0
    ys = np.where(prof > CARD_W * 0.55)[0]
    groups: list[list[int]] = []
    for y in ys:
        if groups and y - groups[-1][-1] <= 6:
            groups[-1].append(int(y))
        else:
            groups.append([int(y)])
    # 罫線は細い。手書き文字の行が横に連結してできた太い塊は除く
    found = [int(np.mean(g)) for g in groups if len(g) <= 10]
    # カードの縁（外枠）を拾った線は除く
    margin = CARD_H * 0.05
    return [y for y in found if margin < y < CARD_H - margin]


# 正立したカードの表罫線（上端・科目名の下・学籍番号の下・氏名の下）の y 座標。CARD_H=480 基準
TABLE_TEMPLATE = np.array([130.0, 215.0, 300.0, 387.0])


def fit_table(card: np.ndarray) -> tuple[list[int], bool | None]:
    """検出した罫線を、正立/180°反転の 2 つのテンプレートに当てはめる。

    戻り値: (表の罫線 4 本の y 座標（現在の向きで上から順）, 上下逆なら True / 判定不能なら None)。
    線が欠けても、見つかった線から残りの位置を補う。何も見つからなければ正立テンプレートを返す。
    """
    found = horizontal_lines(card)
    tol = 14
    best: dict[bool, tuple[float, np.ndarray]] = {}
    for flipped in (False, True):
        tpl = np.sort(CARD_H - TABLE_TEMPLATE) if flipped else TABLE_TEMPLATE
        for dy in range(-20, 21, 2):
            pos = tpl + dy
            errs = np.array([min((abs(p - y) for y in found), default=999) for p in pos])
            score = (errs <= tol).sum() * 100 - np.minimum(errs, tol).sum()
            if flipped not in best or score > best[flipped][0]:
                snapped = np.array(
                    [min(found, key=lambda y: abs(y - p)) if e <= tol else p for p, e in zip(pos, errs)]
                )
                best[flipped] = (float(score), snapped)
    (s_up, ys_up), (s_dn, ys_dn) = best[False], best[True]
    if max(s_up, s_dn) < 200:  # 2 本以上そろわない=表を見つけられていない
        return [int(y) for y in TABLE_TEMPLATE], None
    if abs(s_up - s_dn) < 40:
        return [int(y) for y in (ys_up if s_up >= s_dn else ys_dn)], None
    flipped = s_dn > s_up
    return [int(y) for y in (ys_dn if flipped else ys_up)], flipped


def is_upside_down(card: np.ndarray) -> bool | None:
    return fit_table(card)[1]


def orient_card(card: np.ndarray) -> tuple[np.ndarray, bool | None]:
    flipped = is_upside_down(card)
    if flipped:
        card = cv2.rotate(card, cv2.ROTATE_180)
    return card, flipped


def id_row_crop(card: np.ndarray) -> np.ndarray:
    """学籍番号の行（左のラベル列を除く）を切り出す。"""
    ys, _ = fit_table(card)
    pad = 3
    x0 = int(CARD_W * 0.27)  # ラベル「学生番号/ID Number」を避ける
    return card[max(0, ys[1] + pad) : ys[2] - pad, x0 : CARD_W - 8]


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


# 学籍番号マスの左端（ラベル列との境界）と表の右端の、カード幅に対する比率
LABEL_EDGE, TABLE_RIGHT = 0.285, 0.905


def _id_columns(card: np.ndarray, ys: list[int]) -> tuple[float, float]:
    """学籍番号行の縦罫線 9 本（等間隔）の位置 (最左, 最右) を求める。

    縦線の候補を拾い、「最左・最右を動かして 9 本の等間隔の線が最も多く候補に重なる」位置を探す。
    文字の縦画などの偽の候補があっても、等間隔という配置全体の一致で選ぶので外れにくい。
    線が拾えなければ事前位置（テンプレート）を返す。
    """
    row = card[ys[1] + 3 : ys[2] - 3]
    h, w = row.shape[:2]
    prior_l, prior_r = LABEL_EDGE * w, TABLE_RIGHT * w
    if h < 20:
        return prior_l, prior_r
    gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9)))
    vert = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(5, int(h * 0.6)))))
    prof = vert.sum(axis=0) / 255.0
    cand = np.where(prof > h * 0.5)[0].astype(float)
    if len(cand) == 0:
        return prior_l, prior_r
    best = (-1.0, prior_l, prior_r)
    for left in np.arange(prior_l - 0.04 * w, prior_l + 0.04 * w, 1.0):
        for right in np.arange(prior_r - 0.04 * w, prior_r + 0.04 * w, 1.0):
            pos = left + (right - left) * np.arange(N_CELLS + 1) / N_CELLS
            hit = (np.abs(pos[:, None] - cand[None, :]).min(axis=1) <= 4).sum()
            # 同点なら事前位置に近いものを選ぶ
            score = hit - (abs(left - prior_l) + abs(right - prior_r)) / (w * 10)
            if score > best[0]:
                best = (score, left, right)
    return best[1], best[2]


def _drop_grid_remnants(cell_ink: np.ndarray) -> np.ndarray:
    """マス内に残った罫線の断片（縦に長い細線・端に張り付いた細線）と点状のゴミを取り除く。"""
    h, w = cell_ink.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cell_ink, connectivity=8)
    out = cell_ink.copy()
    biggest = max((stats[i, cv2.CC_STAT_AREA] for i in range(1, n)), default=0)
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        is_full_line = ch >= 0.85 * h and cw <= 0.25 * w  # 行の高さ全体に渡る縦線
        hugs_edge = cw <= 6 and (x <= 3 or x + cw >= w - 3)  # 左右の端に張り付いた細線
        speck = area < max(12, 0.05 * biggest)
        if is_full_line or hugs_edge or speck:
            out[labels == i] = 0
    return out


def id_cells(card: np.ndarray) -> list[np.ndarray]:
    """学籍番号行をマスごとに分割し、罫線を消した白地黒字のグレースケール画像を返す。"""
    ys, _ = fit_table(card)
    y0, y1 = ys[1] + 3, ys[2] - 3
    band = card[y0:y1]
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    ink = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )
    h, w = ink.shape
    left, right = _id_columns(card, ys)
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
        cell_ink = _drop_grid_remnants(cell_ink)
        img = 255 - cell_ink
        cells.append(cv2.copyMakeBorder(img, 20, 20, 40, 40, cv2.BORDER_CONSTANT, value=255))
    return cells
