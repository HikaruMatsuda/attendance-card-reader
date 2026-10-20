"""学籍番号の正規化と並べ替え。

並び順: 先頭の英字（A → H）ごとにまとめる → 入学年度 2 桁（20〜26）→ 残りの桁（昇順）。
ルールは idrule を参照。
"""

from __future__ import annotations

import io
import re
import unicodedata

import pandas as pd

import idrule

RESERVED = ("学籍番号", "元ファイル", "区分", "入学年")
ID_COLUMN_HINTS = ("学籍番号", "学生番号", "学籍", "id", "ID", "student")


def normalize_id(value: object) -> str:
    """全角・空白・小文字を正規化する（例: 'A2500000' → 'A2500000'）。"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return re.sub(r"[\s\-_]", "", unicodedata.normalize("NFKC", str(value))).upper()


def sort_key(student_id: str) -> tuple:
    """(形式の正しさ, 英字, 年, 残り)。形式不正は末尾にまとめて文字列順。"""
    parts = idrule.split(student_id)
    if parts is None:
        return (1, "", "", student_id)
    return (0, *parts)


def read_csv_bytes(data: bytes) -> pd.DataFrame:
    """UTF-8（BOM 可）→ CP932 の順に試して読む。すべて文字列として扱う。"""
    for enc in ("utf-8-sig", "cp932"):
        try:
            return pd.read_csv(io.StringIO(data.decode(enc)), dtype=str, keep_default_na=False)
        except UnicodeDecodeError:
            continue
    raise ValueError("文字コードを判定できません（UTF-8 か Shift_JIS にしてください）")


def guess_id_column(df: pd.DataFrame) -> str:
    """列名のヒント → 学籍番号の形式に合う値が最も多い列、の順で推定する。"""
    for hint in ID_COLUMN_HINTS:
        for col in df.columns:
            if hint.lower() in str(col).lower():
                return col
    scores = {c: df[c].map(lambda v: idrule.is_valid(normalize_id(v))).sum() for c in df.columns}
    return max(scores, key=scores.get)


def merge_and_sort(frames: dict[str, pd.DataFrame], id_columns: dict[str, str]) -> pd.DataFrame:
    """複数 CSV を縦に結合し、学籍番号で並べ替える。

    出力列: 学籍番号（正規化済み）, 区分（先頭英字）, 入学年（2桁）, 元ファイル, 以降は各 CSV の元の列。
    """
    parts = []
    for name, df in frames.items():
        ids = df[id_columns[name]].map(normalize_id)
        # 出力側で付ける列名と衝突する元の列（例: 本アプリが出力した CSV の「学籍番号」）は置き換える
        d = df.drop(columns=[c for c in df.columns if c in RESERVED], errors="ignore")
        d.insert(0, "学籍番号", ids)
        d.insert(1, "元ファイル", name)
        parts.append(d)
    out = pd.concat(parts, ignore_index=True)
    keys = out["学籍番号"].map(sort_key)
    out["区分"] = keys.map(lambda k: k[1])
    out["入学年"] = keys.map(lambda k: k[2])
    out = out.iloc[sorted(range(len(out)), key=lambda i: keys.iloc[i])].reset_index(drop=True)
    first = ["学籍番号", "区分", "入学年", "元ファイル"]
    return out[first + [c for c in out.columns if c not in first]]
