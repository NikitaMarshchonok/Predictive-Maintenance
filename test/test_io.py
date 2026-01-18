from pathlib import Path
import pandas as pd
import pytest

from src.predict import read_cmapss_txt


def test_read_cmapss_txt_shape(tmp_path: Path):
    # create dummy CMAPSS-like file with 26 columns
    # unit_id, cycle, op1, op2, op3, s1..s21 => 5 + 21 = 26
    row = "1 1 0 0 0 " + " ".join(["1"] * 21) + "\n"
    p = tmp_path / "test_FD001.txt"
    p.write_text(row, encoding="utf-8")

    df = read_cmapss_txt(p)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 26)
    assert "unit_id" in df.columns
    assert "cycle" in df.columns
    assert "s21" in df.columns


def test_read_cmapss_txt_raises_on_wrong_cols(tmp_path: Path):
    # only 10 columns -> should raise
    p = tmp_path / "bad.txt"
    p.write_text("1 1 0 0 0 1 1 1 1 1\n", encoding="utf-8")

    with pytest.raises(ValueError):
        _ = read_cmapss_txt(p)
