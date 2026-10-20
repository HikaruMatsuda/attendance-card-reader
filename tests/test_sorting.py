import pandas as pd

import sorting


def test_normalize_id():
    assert sorting.normalize_id(" ａ26 00001 ") == "A2600001"
    assert sorting.normalize_id(None) == ""


def test_sort_groups_by_letter_then_year_then_rest():
    ids = ["H2511111", "A2600002", "A2500000", "H2012345", "A2600001", "A2000003", "A2500001"]
    assert sorted(ids, key=sorting.sort_key) == [
        "A2000003", "A2500000", "A2500001", "A2600001", "A2600002", "H2012345", "H2511111",
    ]


def test_invalid_ids_go_last():
    # 英字が A/H 以外、入学年度が 20〜26 の範囲外のものもルール違反として末尾へ
    ids = ["", "X1", "H2600000", "A1900000", "B2500000", "A2600000"]
    assert sorted(ids, key=sorting.sort_key) == ["A2600000", "H2600000", "", "A1900000", "B2500000", "X1"]


def test_merge_and_sort_multiple_csvs_with_different_columns():
    a = pd.DataFrame({"学籍番号": ["a2600001", "H2511111"], "氏名": ["x", "y"]})
    b = pd.DataFrame({"ID": ["A2400003"], "備考": ["z"]})
    out = sorting.merge_and_sort({"a.csv": a, "b.csv": b}, {"a.csv": "学籍番号", "b.csv": "ID"})
    assert list(out["学籍番号"]) == ["A2400003", "A2600001", "H2511111"]
    assert list(out["区分"]) == ["A", "A", "H"]
    assert list(out["元ファイル"]) == ["b.csv", "a.csv", "a.csv"]


def test_guess_id_column_by_values_when_header_is_unhelpful():
    df = pd.DataFrame({"col1": ["x", "y"], "col2": ["A2600001", "H2511111"]})
    assert sorting.guess_id_column(df) == "col2"


def test_read_csv_bytes_handles_bom_and_cp932():
    assert sorting.read_csv_bytes("学籍番号\nA2600001\n".encode("utf-8-sig")).iloc[0, 0] == "A2600001"
    assert sorting.read_csv_bytes("学籍番号\nA2600001\n".encode("cp932")).columns[0] == "学籍番号"


def test_merge_with_app_output_csv_having_same_column_name():
    # 本アプリの出力 CSV は「学籍番号」「元画像」列を持つ
    a = pd.DataFrame({"学籍番号": ["A2600001"], "元画像": ["p.jpg#1"]})
    out = sorting.merge_and_sort({"a.csv": a}, {"a.csv": "学籍番号"})
    assert list(out.columns) == ["学籍番号", "区分", "入学年", "元ファイル", "元画像"]
