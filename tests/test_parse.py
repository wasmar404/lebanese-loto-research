import json
from datetime import date
from pathlib import Path

import pytest

from loto.parse import ParseError, parse_draw_page, parse_index_page, parse_latest_draw_number

FIX = Path(__file__).parent / "fixtures"


def read(name):
    return (FIX / name).read_text(encoding="utf-8")


def test_draw_1_unsorted_display_and_bonus():
    r = parse_draw_page(read("draw_0001.html"), expected_draw=1)
    assert r.draw == 1
    assert r.draw_date == date(2002, 9, 12)          # DD/MM/YYYY on the page
    assert r.numbers_displayed == [7, 25, 11, 8, 20, 32]
    assert r.numbers_sorted == [7, 8, 11, 20, 25, 32]
    assert r.bonus == 30
    assert not r.displayed_sorted
    assert r.dropdown_date == r.draw_date


def test_draw_1000_prizes_and_no_footer_jackpot():
    r = parse_draw_page(read("draw_1000.html"), expected_draw=1000)
    assert r.numbers_sorted == [5, 12, 13, 29, 34, 39] and r.bonus == 22
    assert r.draw_date == date(2012, 6, 28)
    assert [p.rank for p in r.prizes] == [1, 2, 3, 4, 5]
    assert r.prizes[1].winning_grids == 1 and r.prizes[1].total_lbp == 290_752_983
    # the page footer shows the *next* jackpot (LBP 106,000,000,000) - it must not leak in
    assert all(p.total_lbp != 106_000_000_000 for p in r.prizes)


def test_expected_draw_mismatch_raises():
    with pytest.raises(ParseError):
        parse_draw_page(read("draw_1000.html"), expected_draw=999)


def test_garbage_page_raises():
    with pytest.raises(ParseError):
        parse_draw_page("<html><body>maintenance</body></html>")


def test_latest_and_index():
    assert parse_latest_draw_number(read("draw_1000.html")) == 1000
    rows = parse_index_page(json.loads(read("index_offset0.json"))["value"])
    assert len(rows) == 20
    assert rows[0] == (2458, date(2026, 10, 15))
    assert all(a[0] == b[0] + 1 for a, b in zip(rows, rows[1:]))
