"""Parsers for LLDJ Loto result pages.

Page structure (verified October 2026):

    <div id="resultcontainer">
      <div class="drawresultheader">
        <span class="draw">Results of <span class="yellow">Draw 1000 On 28/06/2012</span></span>
        <ul class="list ballslist">
          <li class="ball big ball-5">5</li> ... six main balls ...
          <li class="ball big extraball ball-22">22</li>      <- additional number
        </ul>
      </div>
      <ul class="list resultstable"> prize rows </ul>
    </div>
    <div class="extrasharedfooter"> ... NEXT-draw jackpot / countdown (NOT historical) ... </div>

Only ``#resultcontainer`` is parsed; the footer jackpot is ignored on purpose because it
shows the *current* estimated jackpot on every historical page.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime

from bs4 import BeautifulSoup

HEADER_RE = re.compile(r"Draw\s+(\d+)\s+On\s+(\d{1,2})/(\d{1,2})/(\d{4})", re.I)
DROPDOWN_RE = re.compile(r"Draw\s*#\s*(\d+)\s*-\s*(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})")
INDEX_OPTION_RE = re.compile(r"SelectResultDrawFromDropdown\(event,\s*\$\(this\),\s*'(\d+)'\)\">\s*Draw\s*#\s*\d+\s*-\s*(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})")


class ParseError(ValueError):
    pass


@dataclass
class PrizeRow:
    rank: int
    distribution: str
    total_lbp: int | None
    winning_grids: int | None
    prize_per_grid_lbp: int | None


@dataclass
class DrawRecord:
    draw: int
    draw_date: date
    numbers_displayed: list[int]          # main numbers, in the order shown on the page
    bonus: int
    dropdown_date: date | None = None     # independent date from the page's draw selector
    prizes: list[PrizeRow] = field(default_factory=list)

    @property
    def numbers_sorted(self) -> list[int]:
        return sorted(self.numbers_displayed)

    @property
    def displayed_sorted(self) -> bool:
        return self.numbers_displayed == sorted(self.numbers_displayed)

    def to_row(self) -> dict:
        s = self.numbers_sorted
        row = {
            "draw": self.draw,
            "draw_date": self.draw_date.isoformat(),
            **{f"n{i + 1}": s[i] for i in range(6)},
            "bonus": self.bonus,
            **{f"d{i + 1}": self.numbers_displayed[i] for i in range(6)},
            "displayed_sorted": self.displayed_sorted,
            "dropdown_date": self.dropdown_date.isoformat() if self.dropdown_date else None,
            "jackpot_winners": next((pr.winning_grids for pr in self.prizes if pr.rank == 1), None),
        }
        return row

    def to_dict(self) -> dict:
        d = asdict(self)
        d["draw_date"] = self.draw_date.isoformat()
        d["dropdown_date"] = self.dropdown_date.isoformat() if self.dropdown_date else None
        return d


def _to_int(text: str) -> int | None:
    t = re.sub(r"[^\d]", "", text or "")
    return int(t) if t else None


def _month_date(day: str, month: str, year: str) -> date:
    return datetime.strptime(f"{int(day):02d} {month} {year}", "%d %B %Y").date()


def parse_latest_draw_number(html: str) -> int:
    """The no-parameter results page shows the latest completed draw."""
    soup = BeautifulSoup(html, "lxml")
    inp = soup.find("input", id="resultdateinput")
    if inp and inp.get("value", "").isdigit():
        return int(inp["value"])
    m = HEADER_RE.search(soup.get_text(" "))
    if not m:
        raise ParseError("cannot find latest draw number")
    return int(m.group(1))


def parse_draw_page(html: str, expected_draw: int | None = None) -> DrawRecord:
    soup = BeautifulSoup(html, "lxml")
    container = soup.find(id="resultcontainer")
    if container is None:
        raise ParseError("no #resultcontainer on page")

    header = container.select_one(".drawresultheader .drawinfobox .draw") or container.select_one("span.draw")
    if header is None:
        raise ParseError("no draw header")
    m = HEADER_RE.search(header.get_text(" ", strip=True))
    if not m:
        raise ParseError(f"unrecognised header text: {header.get_text(' ', strip=True)!r}")
    draw = int(m.group(1))
    day, month, year = int(m.group(2)), int(m.group(3)), int(m.group(4))
    draw_date = date(year, month, day)  # page uses DD/MM/YYYY (draw 1 = 12/09/2002 = 12 Sep 2002)
    if expected_draw is not None and draw != expected_draw:
        raise ParseError(f"page shows draw {draw}, expected {expected_draw}")

    balls = container.select(".drawresultheader ul.ballslist li.ball")
    main, extra = [], []
    for li in balls:
        val = _to_int(li.get_text())
        if val is None:
            raise ParseError(f"non-numeric ball: {li!r}")
        (extra if "extraball" in (li.get("class") or []) else main).append(val)
    if len(main) != 6 or len(extra) != 1:
        raise ParseError(f"expected 6 main + 1 extra ball, got {len(main)} + {len(extra)}")

    dropdown_date = None
    dd = soup.find("input", id="chooseresultdrawdate")
    if dd is not None:
        mm = DROPDOWN_RE.search(dd.get("value", ""))
        if mm and int(mm.group(1)) == draw:
            try:
                dropdown_date = _month_date(mm.group(2), mm.group(3), mm.group(4))
            except ValueError:
                dropdown_date = None

    prizes: list[PrizeRow] = []
    for row in container.select("ul.resultstable li.row.data"):
        cells = {c.get("title", ""): c.get_text(" ", strip=True) for c in row.select("span.cell")}
        rank = _to_int(cells.get("Rank", ""))
        if rank is None:
            continue
        prizes.append(
            PrizeRow(
                rank=rank,
                distribution=cells.get("Distribution", ""),
                total_lbp=_to_int(cells.get("TOTAL AMOUNT OF PRIZES (LBP)", "")),
                winning_grids=_to_int(cells.get("NO. OF WINNING GRIDS", "")),
                prize_per_grid_lbp=_to_int(cells.get("PRIZE PER GRID (LBP)", "")),
            )
        )

    return DrawRecord(draw=draw, draw_date=draw_date, numbers_displayed=main, bonus=extra[0],
                      dropdown_date=dropdown_date, prizes=prizes)


def parse_index_page(html_fragment: str) -> list[tuple[int, date]]:
    """Parse one page of the draw-selector endpoint: [(draw, date), ...]."""
    out = []
    for m in INDEX_OPTION_RE.finditer(html_fragment):
        out.append((int(m.group(1)), _month_date(m.group(2), m.group(3), m.group(4))))
    return out
