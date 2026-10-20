"""出席カード読み取りアプリ: 写真 → カードを検出・切り出し・向き補正 → 学籍番号を認識 → 一覧化。"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

import cards
import ocr

try:  # iPhone の HEIC 写真に対応
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pass

st.set_page_config(page_title="出席カード読み取り", page_icon="🗂️", layout="wide")

LOW_CONF = 0.7  # これ未満の桁があるカードは「要確認」


@dataclass
class Record:
    uid: str
    src: str
    card: np.ndarray  # 向き補正済みの切り出し画像 (BGR)
    reading: ocr.Reading
    orientation_sure: bool
    boxes: list = field(default_factory=list)

    @property
    def needs_check(self) -> bool:
        return (not self.orientation_sure) or self.reading.min_conf < LOW_CONF or "?" in self.reading.text


def load_bgr(data: bytes) -> np.ndarray:
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def png_bytes(bgr: np.ndarray) -> bytes:
    return cv2.imencode(".png", bgr)[1].tobytes()


def draw_boxes(img: np.ndarray, boxes: list[np.ndarray]) -> np.ndarray:
    out = img.copy()
    t = max(2, img.shape[1] // 400)
    for i, b in enumerate(boxes, 1):
        cv2.polylines(out, [b.astype(np.int32)], True, (0, 0, 255), t)
        x, y = b.mean(axis=0).astype(int)
        cv2.putText(out, str(i), (x - 20, y + 15), cv2.FONT_HERSHEY_SIMPLEX, t * 1.2, (255, 0, 0), t * 2)
    return out


def process(files: list, params: cards.DetectParams, engine: str, prefix: str) -> tuple[list[Record], dict]:
    records: list[Record] = []
    overlays: dict[str, np.ndarray] = {}
    bar = st.progress(0.0, text="解析中…")
    for fi, f in enumerate(files):
        img = load_bgr(f.getvalue())
        boxes = cards.detect_cards(img, params)
        overlays[f.name] = draw_boxes(img, boxes)
        for ci, box in enumerate(boxes, 1):
            try:
                card, flipped = cards.orient_card(cards.warp_card(img, box))
                reading = ocr.recognize(card, engine, prefix)
            except Exception as e:  # noqa: BLE001 - 1 枚の失敗で全体を止めず、要確認として残す
                card = cards.warp_card(img, box)
                reading, flipped = ocr.Reading(prefix + "?" * 7, ()), None
                st.warning(f"{f.name}#{ci} を読み取れませんでした（{type(e).__name__}）。手入力してください。")
            records.append(Record(f"{fi}-{ci}", f"{f.name}#{ci}", card, reading,
                                  orientation_sure=flipped is not None))
        bar.progress((fi + 1) / len(files), text=f"解析中… {fi + 1}/{len(files)} 枚")
    bar.empty()
    return records, overlays


def rotate_card(uid: str, engine: str, prefix: str) -> None:
    """向きが逆だったカードを 180° 回して再認識する。"""
    rec = next(r for r in st.session_state.records if r.uid == uid)
    rec.card = cv2.rotate(rec.card, cv2.ROTATE_180)
    rec.reading = ocr.recognize(rec.card, engine, prefix)
    rec.orientation_sure = True
    st.session_state[f"id_{uid}"] = rec.reading.text


# ---------------------------------------------------------------- サイドバー
with st.sidebar:
    st.header("設定")
    engine = st.selectbox("認識エンジン", ocr.ENGINES, help="Indra は文章向けで、手書きの1桁ずつは取りこぼしやすいです。")
    prefix = st.text_input("学籍番号の先頭文字", "A", max_chars=2)
    with st.expander("カード検出の調整"):
        thresh = st.slider("色の感度（小さいほど拾いやすい）", 4, 30, int(cards.DetectParams.color_thresh))
        min_area = st.slider("最小サイズ（画像に対する%）", 0.2, 3.0, cards.DetectParams.min_area_ratio * 100, 0.1)
    only_check = st.toggle("要確認のみ表示", False)
    sort_by_id = st.toggle("学籍番号順に並べる", True)

params = cards.DetectParams(color_thresh=float(thresh), min_area_ratio=min_area / 100)

# ---------------------------------------------------------------- メイン
st.title("🗂️ 出席カード読み取り")
st.caption("机に並べたカードの写真から、1枚ずつ正しい向きで切り抜き、学籍番号を抽出して一覧にします。")

files = st.file_uploader("カードを撮影した写真（複数可）", type=["jpg", "jpeg", "png", "heic", "webp"],
                         accept_multiple_files=True)

if files and st.button("解析する", type="primary"):
    with st.spinner("モデルを読み込み中…"):
        recs, overlays = process(files, params, engine, prefix)
    st.session_state.records, st.session_state.overlays = recs, overlays
    for r in recs:  # 前回の手修正を引き継がない
        st.session_state[f"id_{r.uid}"] = r.reading.text

recs: list[Record] = st.session_state.get("records", [])
if not recs:
    st.info("写真をアップロードして「解析する」を押してください。")
    st.stop()

with st.expander(f"検出結果（番号付きの枠）— {len(st.session_state.overlays)} 枚の写真"):
    for name, ov in st.session_state.overlays.items():
        st.image(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB), caption=name, use_container_width=True)

ids = {r.uid: st.session_state.get(f"id_{r.uid}", r.reading.text).strip() for r in recs}
counts = pd.Series(list(ids.values())).value_counts()
dups = set(counts[counts > 1].index) - {""}
n_check = sum(r.needs_check for r in recs)
c1, c2, c3 = st.columns(3)
c1.metric("検出したカード", len(recs))
c2.metric("要確認", n_check)
c3.metric("学籍番号の重複", len(dups))
if dups:
    st.warning("重複している学籍番号: " + ", ".join(sorted(dups)))

order = sorted(recs, key=lambda r: ids[r.uid]) if sort_by_id else recs
shown = [r for r in order if r.needs_check] if only_check else order

st.divider()
for r in shown:
    left, right = st.columns([2, 5], vertical_alignment="center")
    with left:
        st.text_input(r.src, key=f"id_{r.uid}", help="読み取りが違う場合は直接書き換えてください。")
        if not r.orientation_sure:
            st.caption("⚠️ 向きを判定できませんでした")
        if r.reading.confs:
            low = [i + 1 for i, p in enumerate(r.reading.confs) if p < LOW_CONF]
            st.caption(("⚠️ 要確認: " + "・".join(f"{i}桁目" for i in low)) if low
                       else f"✅ 確信度 最低 {r.reading.min_conf:.0%}")
        st.button("↻ 180°回転して再認識", key=f"rot_{r.uid}", on_click=rotate_card, args=(r.uid, engine, prefix))
    with right:
        st.image(cv2.cvtColor(r.card, cv2.COLOR_BGR2RGB), width=520)

# ---------------------------------------------------------------- ダウンロード
st.divider()
final = [(ids[r.uid] or "unknown", r) for r in order]
df = pd.DataFrame({"学籍番号": [i for i, _ in final], "元画像": [r.src for _, r in final]})
buf = io.BytesIO()
used: dict[str, int] = {}
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for sid, r in final:
        used[sid] = used.get(sid, 0) + 1
        name = sid if used[sid] == 1 else f"{sid}_{used[sid]}"
        z.writestr(f"{name}.png", png_bytes(r.card))
d1, d2 = st.columns(2)
d1.download_button("学籍番号一覧 (CSV)", df.to_csv(index=False).encode("utf-8-sig"), "attendance_ids.csv", "text/csv")
d2.download_button("切り抜き画像 (ZIP)", buf.getvalue(), "cards.zip", "application/zip")
