"""複数の CSV をまとめて、学籍番号（A/H → 入学年 → 残りの桁）の順に並べ替える。"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import idrule
import sorting

st.set_page_config(page_title="CSV並べ替え", page_icon="🔢", layout="wide")
st.title("🔢 CSV 並べ替え")
st.caption("複数の CSV を結合し、学籍番号を「A / H ごと → 入学年（25, 26…）→ 残りの桁」の順に並べます。")

files = st.file_uploader("CSV ファイル（複数可）", type=["csv"], accept_multiple_files=True)
if not files:
    st.info("読み取りアプリが出力した CSV など、学籍番号の列を含む CSV をアップロードしてください。")
    st.stop()

frames: dict[str, pd.DataFrame] = {}
id_cols: dict[str, str] = {}
for f in files:
    try:
        frames[f.name] = sorting.read_csv_bytes(f.getvalue())
    except Exception as e:  # noqa: BLE001 - 読めないファイルを画面に出して続行する
        st.error(f"{f.name}: {e}")

if not frames:
    st.stop()

with st.expander("学籍番号の列（自動判定。違う場合は変更）", expanded=len(frames) == 1):
    for name, df in frames.items():
        guess = sorting.guess_id_column(df)
        id_cols[name] = st.selectbox(f"{name}（{len(df)} 行）", list(df.columns),
                                     index=list(df.columns).index(guess), key=f"col_{name}")

merged = sorting.merge_and_sort(frames, id_cols)
valid = merged["学籍番号"].map(idrule.is_valid)
dup = merged["学籍番号"].duplicated(keep=False) & valid

drop_dup = st.toggle("重複した学籍番号を 1 件にまとめる（最初の行を残す）", False)
if drop_dup:
    merged = merged[~(merged["学籍番号"].duplicated() & valid)].reset_index(drop=True)
    valid = merged["学籍番号"].map(idrule.is_valid)
    dup = pd.Series(False, index=merged.index)

c1, c2, c3 = st.columns(3)
c1.metric("合計", len(merged))
c2.metric("重複", int(dup.sum()))
c3.metric("形式不正（末尾に配置）", int((~valid).sum()))
if (~valid).any():
    st.warning("学籍番号のルール（A/H＋入学年度20〜26＋5桁）に合わないものがあります（末尾にまとめています）: "
               + ", ".join(sorted(set(merged.loc[~valid, "学籍番号"].replace("", "（空）")))[:20]))

group = st.radio("表示", ["すべて"] + sorted(set(merged.loc[valid, "区分"])), horizontal=True)
view = merged if group == "すべて" else merged[merged["区分"] == group]
st.dataframe(view.style.apply(lambda r: ["background-color: #5c4a00" if dup.get(r.name, False) else "" for _ in r], axis=1),
             use_container_width=True, hide_index=True)

st.download_button("並べ替えた CSV をダウンロード", merged.to_csv(index=False).encode("utf-8-sig"),
                   "sorted_ids.csv", "text/csv", type="primary")
