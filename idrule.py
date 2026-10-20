"""学籍番号のルール（認識・検証・並べ替えで共通に使う）。

形式: 英字 1 文字（A または H）＋ 数字 7 桁
  - 数字の先頭 2 桁: 入学年度（20〜26）
  - 残り 5 桁: 指定なし
例: A2512345
"""

from __future__ import annotations

import re

LETTERS = ("A", "H")
YEAR_MIN, YEAR_MAX = 20, 26
N_DIGITS = 7

ID_RE = re.compile(rf"^([{''.join(LETTERS)}])(\d{{2}})(\d{{{N_DIGITS - 2}}})$")
VALID_YEARS = [f"{y:02d}" for y in range(YEAR_MIN, YEAR_MAX + 1)]


def is_valid(student_id: str) -> bool:
    m = ID_RE.match(student_id)
    return bool(m) and m.group(2) in VALID_YEARS


def split(student_id: str) -> tuple[str, str, str] | None:
    """(英字, 入学年度2桁, 残り) に分ける。ルールに合わなければ None。"""
    m = ID_RE.match(student_id)
    if not m or m.group(2) not in VALID_YEARS:
        return None
    return m.group(1), m.group(2), m.group(3)
