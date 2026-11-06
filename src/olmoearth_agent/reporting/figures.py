# SPDX-License-Identifier: LicenseRef-OlmoEarth-Artifact-License
# Copyright (c) 2026 OlmoEarth Agent contributors
"""Figures of a determination, drawn from the run's own data.

Each check route (which windows to check first, how wrong a map is, which
part to trust, which of two maps is more accurate) writes one SVG beside the
data files it already writes: the window grid as the tool saw it, with what
the determination used marked on it, and the figure the answer rests on (an
interval, the levels tested, the labels' counts). Windows are drawn by row
and column, row 0 at the top; no coordinate is drawn or written (rule 1).

The grid is a PNG embedded in the SVG, written here with the standard
library, so the figures need no plotting dependency. Marks (outlined or
filled windows) are SVG rectangles up to :data:`MAX_MARKS`, painted into the
PNG past that.
"""

from __future__ import annotations

import base64
import html
import math
import struct
import textwrap
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field

#: An sRGB colour.
RGB = tuple[int, int, int]

#: A window with no value (no-data, outside the map, failed).
NO_DATA: RGB = (255, 255, 255)
#: The confidence ramp: the surest window to the least sure.
SURE: RGB = (233, 238, 243)
UNSURE: RGB = (40, 54, 70)
#: Two maps agree / differ at a window.
AGREE: RGB = (236, 239, 242)
DIFFER: RGB = (244, 170, 66)
#: A certified zone's tint.
ZONE: RGB = (52, 120, 214)

OUTLINE = "#111111"
#: The review set: amber, which stands out on the darkest (least confident) windows.
REVIEW = "#F5A623"
SAMPLE = "#00989A"
RIGHT = "#2E9E5B"
WRONG = "#E8298A"
UNJUDGED = "#B4B4B4"
MAP_A = "#1F6FB4"
MAP_B = "#F07F0E"
NEITHER = "#8C8C8C"
INK = "#1A1A1A"
MUTED = "#5F6368"

#: The longest side of the drawn grid, in px.
MAP_BOX = 420
#: The side panel's width, in px.
SIDE = 300
#: Marks drawn as SVG rectangles; more are painted into the grid's PNG.
MAX_MARKS = 2000
#: Characters per line of the subtitle and caption.
LINE_CHARS = 112
#: The narrowest figure, in px: room for a full line of the caption.
MIN_WIDTH = 740
FONT = "DejaVu Sans, Helvetica, Arial, sans-serif"


def css(colour: RGB) -> str:
    """An sRGB triple as ``#rrggbb``."""
    return "#{:02x}{:02x}{:02x}".format(*colour)


def hex_rgb(colour: str) -> RGB:
    """``#rrggbb`` as an sRGB triple."""
    value = colour.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def pct(value: float | None) -> str:
    """A fraction as a percent, as the facts give it: one decimal from 1%, two
    significant figures below (``statistical_rules.percent``); ``None`` is n/a."""
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    share = 100.0 * float(value)
    if share == 0.0 or abs(share) >= 1.0:
        return f"{share:.1f}%"
    return f"{share:.2g}%"


def level(value: float) -> str:
    """A set level (alpha, an axis tick) as a percent with no rounding: 2.5%, 10%."""
    return f"{100.0 * float(value):g}%"


# --------------------------------------------------------------------------- the grid


def window_grid(n_windows: int, grid: Sequence[int] | None) -> tuple[int, int, bool]:
    """``(rows, cols, given)``: the run's window grid, or a layout by index.

    A grid whose cells number ``n_windows`` is the run's own (``given``);
    otherwise the windows are laid out in index order, row by row, in a
    near-square block, and the figure says so.
    """
    if n_windows < 1:
        raise ValueError("a figure needs at least one window")
    if grid is not None and len(grid) == 2:
        rows, cols = int(grid[0]), int(grid[1])
        if rows > 0 and cols > 0 and rows * cols == n_windows:
            return rows, cols, True
    cols = max(1, math.ceil(math.sqrt(n_windows)))
    rows = max(1, math.ceil(n_windows / cols))
    return rows, cols, False


def mix(a: RGB, b: RGB, t: float) -> RGB:
    """``a`` at ``t = 0`` to ``b`` at ``t = 1``."""
    t = min(1.0, max(0.0, t))
    return (
        round(a[0] + (b[0] - a[0]) * t),
        round(a[1] + (b[1] - a[1]) * t),
        round(a[2] + (b[2] - a[2]) * t),
    )


def fade(colour: RGB, amount: float) -> RGB:
    """``colour`` blended toward white by ``amount`` (0 keeps it, 1 is white)."""
    return mix(colour, (255, 255, 255), amount)


def confidence_base(
    margin: Sequence[float | None], n_cells: int, *, faded: float = 0.0
) -> list[RGB | None]:
    """Each window shaded by its confidence rank: the surest light, the least dark.

    ``margin`` holds one value per grid cell. The shade follows the window's
    rank among the windows with a value, so it shows the order a review set
    takes (least confident first) whatever the margins' spread; tied margins
    share one shade (their mean rank), so no order is drawn that the scores
    do not hold, and a map with one margin everywhere is drawn flat
    (:func:`flat_confidence`). ``None`` or a non-finite margin is no-data.
    ``faded`` blends the shades toward white, for a grid that carries marks.
    """
    cells: list[RGB | None] = [None] * n_cells
    valid = []
    for i, m in enumerate(margin):
        if i >= n_cells:
            break
        if m is not None and math.isfinite(float(m)):
            valid.append((float(m), i))
    valid.sort()
    if flat_confidence(margin):
        for _, i in valid:
            cells[i] = fade(mix(UNSURE, SURE, 0.75), faded)
        return cells
    last = max(1, len(valid) - 1)
    start = 0
    while start < len(valid):
        end = start
        while end + 1 < len(valid) and valid[end + 1][0] == valid[start][0]:
            end += 1
        shade = fade(mix(UNSURE, SURE, (start + end) / 2 / last), faded)
        for _, i in valid[start : end + 1]:
            cells[i] = shade
        start = end + 1
    return cells


def flat_confidence(margin: Sequence[float | None]) -> bool:
    """Whether every window with a margin has the same one (no confidence to draw)."""
    seen = {float(m) for m in margin if m is not None and math.isfinite(float(m))}
    return len(seen) <= 1


def mask_base(
    mask: Sequence[bool | None], n_cells: int, *, on: RGB, off: RGB
) -> list[RGB | None]:
    """Each window ``on`` where ``mask`` holds, else ``off``; ``None`` is no-data."""
    cells: list[RGB | None] = [None] * n_cells
    for i, value in enumerate(mask):
        if i >= n_cells:
            break
        if value is not None:
            cells[i] = on if value else off
    return cells


def tint(
    base: Sequence[RGB | None], windows: Sequence[int], colour: RGB, amount: float
) -> list[RGB | None]:
    """``base`` with ``windows`` mixed toward ``colour`` by ``amount``."""
    cells = list(base)
    for i in windows:
        if 0 <= i < len(cells):
            cells[i] = mix(cells[i] or NO_DATA, colour, amount)
    return cells


def _blocked(
    cells: Sequence[RGB | None], rows: int, cols: int, k: int
) -> tuple[list[RGB | None], int, int]:
    """The grid in k x k blocks, each the mean colour of its windows with one."""
    out_rows, out_cols = math.ceil(rows / k), math.ceil(cols / k)
    sums = [[0, 0, 0, 0] for _ in range(out_rows * out_cols)]
    for i, rgb in enumerate(cells[: rows * cols]):
        if rgb is None:
            continue
        r, c = divmod(i, cols)
        acc = sums[(r // k) * out_cols + c // k]
        acc[0] += rgb[0]
        acc[1] += rgb[1]
        acc[2] += rgb[2]
        acc[3] += 1
    out: list[RGB | None] = [
        (round(s[0] / s[3]), round(s[1] / s[3]), round(s[2] / s[3])) if s[3] else None
        for s in sums
    ]
    return out, out_rows, out_cols


def png(cells: Sequence[RGB | None], rows: int, cols: int, scale: int) -> bytes:
    """An RGB PNG of the grid, each window ``scale`` x ``scale`` px (no-data white)."""
    raw = bytearray()
    for r in range(rows):
        line = bytearray()
        for c in range(cols):
            i = r * cols + c
            rgb = cells[i] if i < len(cells) else None
            line += bytes(rgb or NO_DATA) * scale
        raw += (b"\x00" + bytes(line)) * scale

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    header = struct.pack(">IIBBBBB", cols * scale, rows * scale, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )


# --------------------------------------------------------------------------- the figure


def _text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 12,
    weight: int = 400,
    fill: str = INK,
    anchor: str = "start",
) -> str:
    """One line of SVG text, escaped."""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}" text-anchor="{anchor}">{html.escape(str(text))}</text>'
    )


def _nice_ceiling(value: float) -> float:
    """The smallest of 1, 2, 2.5 or 5 times a power of ten at or above ``value``."""
    if value <= 0:
        return 1.0
    power = 10.0 ** math.floor(math.log10(value))
    for step in (1.0, 2.0, 2.5, 5.0):
        if step * power >= value - 1e-12:
            return step * power
    return 10.0 * power


def _wrap(lines: Sequence[str]) -> list[str]:
    """Each line wrapped at :data:`LINE_CHARS` characters."""
    out: list[str] = []
    for line in lines:
        out.extend(textwrap.wrap(line, LINE_CHARS) or [""])
    return out


@dataclass
class _Mark:
    """Windows outlined or filled in one colour."""

    windows: list[int]
    colour: str
    filled: bool


@dataclass
class GridFigure:
    """One determination's figure: the window grid, its marks and a side panel.

    ``base`` holds one colour per grid cell (``rows * cols``, ``None`` for
    no-data). Marks outline or fill windows; the legend names the base's
    meaning and each mark; charts in the side panel show the figure the
    answer rests on.
    """

    title: str
    subtitle: str
    rows: int
    cols: int
    given_grid: bool
    base: list[RGB | None]
    caption: list[str] = field(default_factory=list)
    marks: list[_Mark] = field(default_factory=list, init=False, repr=False)
    legend_rows: list[tuple[str, str, str]] = field(
        default_factory=list, init=False, repr=False
    )
    charts: list[tuple[list[str], int]] = field(
        default_factory=list, init=False, repr=False
    )

    def mark(
        self,
        windows: Sequence[int],
        colour: str,
        *,
        filled: bool = False,
        label: str | None = None,
    ) -> GridFigure:
        """Outline (or fill) ``windows``; a label adds a legend row."""
        n = self.rows * self.cols
        keep = [int(i) for i in windows if 0 <= int(i) < n]
        self.marks.append(_Mark(keep, colour, filled))
        if label:
            self.legend("fill" if filled else "outline", colour, label)
        return self

    def legend(self, kind: str, colour: str, label: str) -> GridFigure:
        """A legend row: ``kind`` is ``fill``, ``outline`` or ``ramp``."""
        self.legend_rows.append((kind, colour, label))
        return self

    def _next_panel(self) -> str:
        """The next chart's panel letter: the grid is (a), charts (b), (c), ..."""
        return chr(ord("b") + len(self.charts))

    def caption_text(self) -> str:
        """The figure's legend as one paragraph, as the SVG and the result give it."""
        notes = list(self.caption)
        k = self.block()
        if k > 1:
            notes.append(
                f"Each drawn cell is a block of up to {k} x {k} windows, coloured by "
                "their mean."
            )
        if not self.given_grid:
            notes.append(
                "No window grid was given, so the windows are laid out in index "
                "order, row by row."
            )
        notes.append(
            "Windows are drawn by row and column, row 0 at the top; no "
            "coordinates are drawn."
        )
        return " ".join(notes)

    def interval_chart(
        self, title: str, rows: Sequence[tuple[str, float, float, float | None]]
    ) -> GridFigure:
        """Each ``(label, low, high, estimate)`` as a bar on one percent axis from 0.

        The axis ends at a round value above the highest bound, and never
        past 100%: a rate cannot exceed it.
        """
        top = max([float(high) for _, _, high, _ in rows] + [0.01])
        upper = min(1.0, _nice_ceiling(top * 1.15))
        width = SIDE - 20
        head = f"({self._next_panel()}) {title}"
        parts = [_text(0, 14, head, size=12, weight=600)]
        y = 34
        for label, low, high, est in rows:
            parts.append(_text(0, y, label, size=11, fill=MUTED))
            x0 = width * float(low) / upper
            x1 = width * float(high) / upper
            parts.append(
                f'<line x1="{x0:.1f}" y1="{y + 10}" x2="{x1:.1f}" y2="{y + 10}" '
                f'stroke="{INK}" stroke-width="4" stroke-linecap="round"/>'
            )
            shown = f"{pct(low)} to {pct(high)}"
            if est is not None:
                xe = width * float(est) / upper
                parts.append(
                    f'<circle cx="{xe:.1f}" cy="{y + 10}" r="5" fill="#ffffff" '
                    f'stroke="{INK}" stroke-width="2"/>'
                )
                shown = f"{pct(est)} ({shown})"
            parts.append(_text(0, y + 30, shown, size=11))
            y += 48
        parts.append(
            f'<line x1="0" y1="{y}" x2="{width}" y2="{y}" stroke="{MUTED}" '
            'stroke-width="1"/>'
        )
        for frac, anchor in ((0.0, "start"), (0.5, "middle"), (1.0, "end")):
            parts.append(
                _text(
                    width * frac,
                    y + 14,
                    level(upper * frac),
                    size=10,
                    fill=MUTED,
                    anchor=anchor,
                )
            )
        self.charts.append((parts, y + 24))
        return self

    def bar_chart(
        self,
        title: str,
        rows: Sequence[tuple[str, float, str]],
        *,
        limit: float | None = None,
        limit_label: str | None = None,
        as_percent: bool = False,
    ) -> GridFigure:
        """Each ``(label, value, colour)`` as a bar; ``limit`` draws a line.

        With ``as_percent`` the bars are fractions on an axis that never
        passes 100%; otherwise counts, the longest bar the full width.
        """
        values = [float(v) for _, v, _ in rows]
        if limit is not None:
            values.append(float(limit))
        top = max(values + [1e-9])
        upper = min(1.0, _nice_ceiling(top * 1.1)) if as_percent else top
        width = SIDE - 150
        head = f"({self._next_panel()}) {title}"
        parts = [_text(0, 14, head, size=12, weight=600)]
        y = 26
        for label, value, colour in rows:
            w = max(1.0, width * float(value) / upper)
            parts.append(_text(0, y + 11, label, size=11, fill=MUTED))
            parts.append(
                f'<rect x="100" y="{y}" width="{w:.1f}" height="14" fill="{colour}"/>'
            )
            shown = pct(value) if as_percent else f"{value:g}"
            # A fixed column right of the bars, clear of the limit line.
            parts.append(_text(100 + width + 8, y + 11, shown, size=11))
            y += 22
        if limit is not None:
            x = 100 + width * float(limit) / upper
            parts.append(
                f'<line x1="{x:.1f}" y1="22" x2="{x:.1f}" y2="{y}" stroke="{WRONG}" '
                'stroke-width="1.5" stroke-dasharray="4 3"/>'
            )
            if limit_label:
                parts.append(
                    _text(x, y + 13, limit_label, size=10, fill=WRONG, anchor="middle")
                )
                y += 14
        self.charts.append((parts, y + 10))
        return self

    def block(self) -> int:
        """Windows per drawn cell along a side: 1, or more past the box."""
        return max(1, math.ceil(max(self.rows, self.cols) / MAP_BOX))

    def to_svg(self) -> str:
        """The figure as one standalone SVG document."""
        n_cells = self.rows * self.cols
        cells = list(self.base[:n_cells]) + [None] * max(0, n_cells - len(self.base))
        many = sum(len(m.windows) for m in self.marks) > MAX_MARKS
        if many:
            # Too many to draw one by one: paint them into the grid.
            for m in self.marks:
                colour = hex_rgb(m.colour)
                for i in m.windows:
                    cells[i] = (
                        colour if m.filled else mix(cells[i] or NO_DATA, colour, 0.8)
                    )
        rows, cols, k = self.rows, self.cols, self.block()
        if k > 1:
            # A grid larger than the box: each drawn cell is k x k windows,
            # the mean of their colours, so the SVG stays small.
            cells, rows, cols = _blocked(cells, self.rows, self.cols, k)
        scale = max(1, MAP_BOX // max(rows, cols))
        native_w, native_h = cols * scale, rows * scale
        shrink = min(1.0, MAP_BOX / max(native_w, native_h))
        map_w, map_h = native_w * shrink, native_h * shrink

        rects: list[str] = []
        for m in [] if many else self.marks:
            drawn = sorted(
                {(i // self.cols // k) * cols + (i % self.cols) // k for i in m.windows}
            )
            for i in drawn:
                r, c = divmod(i, cols)
                if m.filled:
                    rects.append(
                        f'<rect x="{c * scale}" y="{r * scale}" width="{scale}" '
                        f'height="{scale}" fill="{m.colour}"/>'
                    )
                else:
                    inset = 0.5 if scale > 2 else 0.0
                    rects.append(
                        f'<rect x="{c * scale + inset}" y="{r * scale + inset}" '
                        f'width="{scale - 2 * inset}" height="{scale - 2 * inset}" '
                        f'fill="none" stroke="{m.colour}" stroke-width="1.4" '
                        'vector-effect="non-scaling-stroke"/>'
                    )

        image = base64.b64encode(png(cells, rows, cols, scale)).decode()
        subtitle = _wrap([self.subtitle])
        caption = _wrap([self.caption_text()])
        top = 72 + 16 * len(subtitle)
        legend_parts, legend_h = self._legend_svg()
        side_h = legend_h + sum(h + 14 for _, h in self.charts)
        body_h = max(map_h, side_h)
        side_x = 16 + max(map_w, 160) + 32
        # Wide enough for a full line of the subtitle and caption.
        width = max(int(side_x + SIDE + 16), MIN_WIDTH)
        height = int(top + body_h + 18 + 16 * len(caption) + 12)

        out = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" font-family="{FONT}">',
            f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
            _text(16, 28, self.title, size=17, weight=600),
        ]
        for k, line in enumerate(subtitle):
            out.append(_text(16, 50 + 16 * k, line, size=12, fill=MUTED))
        out += [
            _text(16, top - 8, "(a)", size=12, weight=600),
            f'<g transform="translate(16 {top}) scale({shrink:.6f})">',
            f'<image width="{native_w}" height="{native_h}" '
            f'href="data:image/png;base64,{image}" '
            'style="image-rendering:pixelated"/>',
            *rects,
            f'<rect x="0" y="0" width="{native_w}" height="{native_h}" fill="none" '
            f'stroke="{MUTED}" stroke-width="1" vector-effect="non-scaling-stroke"/>',
            "</g>",
            f'<g transform="translate({side_x:.1f} {top})">',
            *legend_parts,
        ]
        chart_y = legend_h
        for parts, h in self.charts:
            out.append(f'<g transform="translate(0 {chart_y + 14})">')
            out.extend(parts)
            out.append("</g>")
            chart_y += h + 14
        out.append("</g>")
        line_y = top + body_h + 18
        for line in caption:
            out.append(_text(16, line_y + 12, line, size=11, fill=MUTED))
            line_y += 16
        out.append("</svg>")
        return "\n".join(out) + "\n"

    def _legend_svg(self) -> tuple[list[str], int]:
        """The legend rows, in the side panel's coordinates, and their height."""
        parts: list[str] = []
        y = 0
        for kind, colour, label in self.legend_rows:
            if kind == "ramp":
                for k in range(8):
                    shade = css(mix(SURE, UNSURE, k / 7))
                    parts.append(
                        f'<rect x="{k * 6}" y="{y + 2}" width="6" height="12" '
                        f'fill="{shade}"/>'
                    )
                parts.append(_text(56, y + 12, label, size=11))
            elif kind == "fill":
                # A white swatch (no-data, not compared) gets a thin edge.
                edge = (
                    f' stroke="{MUTED}" stroke-width="0.6"'
                    if colour.lower() == css(NO_DATA)
                    else ""
                )
                parts.append(
                    f'<rect x="0" y="{y + 2}" width="14" height="12" '
                    f'fill="{colour}"{edge}/>'
                )
                parts.append(_text(22, y + 12, label, size=11))
            else:
                parts.append(
                    f'<rect x="1" y="{y + 3}" width="12" height="10" fill="none" '
                    f'stroke="{colour}" stroke-width="1.6"/>'
                )
                parts.append(_text(22, y + 12, label, size=11))
            y += 20
        return parts, y


# --------------------------------------------------------------------------- the routes


@dataclass(frozen=True)
class Rendered:
    """A determination's figure (``svg``) and its legend (``caption``).

    The caption is the scientific explanation of the figure, written from the
    run's numbers: what each panel shows, how many windows, by which method,
    the result with its interval, and what it assumes. The SVG carries it at
    its foot; a tool result gives it as ``figure_caption``.
    """

    svg: str
    caption: str


def _layout(
    values: Sequence[object], grid: Sequence[int] | None
) -> tuple[int, int, bool]:
    """The figure's grid for one value per window."""
    return window_grid(len(values), grid)


def _grid_words(rows: int, cols: int, given: bool) -> str:
    """``on its 128 x 128 window grid`` (or nothing without a grid)."""
    return f" on its {rows} x {cols} window grid" if given else ""


def _shade(
    margin: Sequence[float | None], *, no_confidence: bool = False
) -> tuple[str, str]:
    """The legend's label and the caption's words for a faded confidence grid.

    ``no_confidence`` marks a class map without top-k scores: its margin is
    the same wherever a window holds a class, so its grid is drawn flat and
    the words say why (``statistical_rules.CLASS_MAP_NO_CONFIDENCE``).
    """
    if no_confidence:
        return (
            "no confidence (class map)",
            "on a grid drawn flat: a class map has no confidence to rank by",
        )
    if flat_confidence(margin):
        return (
            "one confidence everywhere (flat)",
            "on a flat grid, as the map gives one confidence everywhere",
        )
    return "map confidence (faded)", "on the map's confidence rank (faded)"


def _done(fig: GridFigure) -> Rendered:
    """The figure's SVG and caption."""
    return Rendered(fig.to_svg(), fig.caption_text())


def review_set_figure(
    margin: Sequence[float | None],
    grid: Sequence[int] | None,
    review: Sequence[int],
    *,
    budget: float,
    n_valid: int,
) -> Rendered:
    """Which windows to check first: the grid by confidence, the review set outlined.

    ``margin`` holds one value per grid cell (``None`` where no window is);
    ``review`` the review set's window indices on that grid.
    """
    rows, cols, given = _layout(margin, grid)
    n = len(review)
    flat = flat_confidence(margin)
    shading = (
        "drawn flat, as the map gives one confidence everywhere"
        if flat
        else "shaded by confidence rank from the map's own scores (dark: least "
        "confident; tied scores share a shade)"
    )
    fig = GridFigure(
        "Which windows to check first",
        f"The {n:,} least confident of {n_valid:,} windows ({pct(budget)} "
        "review budget), outlined. No labels were used.",
        rows,
        cols,
        given,
        confidence_base(margin, rows * cols),
        caption=[
            f"Which windows to check first. (a) The map's {n_valid:,} windows"
            f"{_grid_words(rows, cols, given)}, {shading}; the "
            f"{n:,} least confident, a {pct(budget)} review budget, are "
            "outlined. No labels were used: the review set says where to look "
            "first, not how wrong the map is."
        ],
    )
    fig.legend("ramp", "", _shade(margin)[0] if flat else "most sure to least sure")
    fig.mark(review, REVIEW, label="check first (the review set)")
    return _done(fig)


def plan_figure(
    margin: Sequence[float | None],
    grid: Sequence[int] | None,
    sample: Sequence[int],
    *,
    design: str,
    seed: int | str,
    n_valid: int,
    no_confidence: bool = False,
) -> Rendered:
    """A label plan: the windows drawn for labelling, on the map's confidence.

    ``no_confidence`` marks a class map without top-k scores (:func:`_shade`).
    """
    rows, cols, given = _layout(margin, grid)
    n = len(sample)
    legend, shading = _shade(margin, no_confidence=no_confidence)
    fig = GridFigure(
        "Which windows to label",
        f"{n:,} of {n_valid:,} windows drawn by the {design} design (seed {seed}).",
        rows,
        cols,
        given,
        confidence_base(margin, rows * cols, faded=0.55),
        caption=[
            f"Which windows to label. (a) The {n:,} windows drawn by the {design} "
            f"design (seed {seed}) from the map's {n_valid:,} windows"
            f"{_grid_words(rows, cols, given)}, outlined {shading}. "
            "Label every drawn window: the error rate and its interval come from "
            "these labels alone, and the number to label is fixed before any "
            "label is read."
        ],
    )
    fig.legend("ramp", "", legend)
    fig.mark(sample, SAMPLE, label="drawn for labelling")
    return _done(fig)


def estimate_figure(
    margin: Sequence[float | None],
    grid: Sequence[int] | None,
    sample: Sequence[int],
    wrong: Sequence[int | None],
    *,
    intervals: Sequence[tuple[str, float, float, float | None]],
    design: str,
    n_valid: int,
    no_confidence: bool = False,
) -> Rendered:
    """How wrong the map is: each labelled window right or wrong, and the error rate.

    ``wrong`` holds, per drawn window in ``sample``'s order, 1 (the map is
    wrong there), 0 (right) or ``None`` (not judged); ``intervals`` are
    ``(label, low, high, estimate)`` as fractions, the first the headline.
    The caption gives the headline's estimate once, then each interval once.
    ``no_confidence`` marks a class map without top-k scores (:func:`_shade`).
    """
    rows, cols, given = _layout(margin, grid)
    pairs = list(zip(sample, wrong))
    right = [i for i, w in pairs if w == 0]
    bad = [i for i, w in pairs if w == 1]
    unjudged = [i for i, w in pairs if w not in (0, 1)]
    n_labelled = len(right) + len(bad)
    head = intervals[0] if intervals else None
    result = (
        f"{pct(head[3])} ({pct(head[1])} to {pct(head[2])}, {head[0]})"
        if head is not None and head[3] is not None
        else "not available"
    )
    if head is not None and len(intervals) > 1:
        bands = "; ".join(
            f"{label}: {pct(lo)} to {pct(hi)}" for label, lo, hi, _ in intervals
        )
        told = f", {pct(head[3])}; {bands}" if head[3] is not None else f": {bands}"
    else:
        told = f", {result}"
    legend, shading = _shade(margin, no_confidence=no_confidence)
    fig = GridFigure(
        "How wrong the map is",
        f"{len(bad):,} of {n_labelled:,} labelled windows are wrong; the error "
        f"rate over all {n_valid:,} windows is {result}.",
        rows,
        cols,
        given,
        confidence_base(margin, rows * cols, faded=0.6),
        caption=[
            f"How wrong the map is. (a) The {n_labelled:,} labelled windows of a "
            f"{design} sample from the map's {n_valid:,} windows"
            f"{_grid_words(rows, cols, given)}, shown {shading}; green where the "
            f"label agrees with the map ({len(right):,}), magenta where it does "
            f"not ({len(bad):,})"
            + (f", grey not judged ({len(unjudged):,})" if unjudged else "")
            + f". (b) The map's error rate over all its windows{told}. The "
            "intervals assume the labels are right."
        ],
    )
    fig.legend("ramp", "", legend)
    fig.mark(right, RIGHT, filled=True, label=f"label agrees: right ({len(right):,})")
    fig.mark(bad, WRONG, filled=True, label=f"label differs: wrong ({len(bad):,})")
    if unjudged:
        fig.mark(
            unjudged, UNJUDGED, filled=True, label=f"not judged ({len(unjudged):,})"
        )
    fig.interval_chart("Error rate over all windows", intervals)
    return _done(fig)


def zone_figure(
    margin: Sequence[float | None],
    grid: Sequence[int] | None,
    zone: Sequence[int],
    sample: Sequence[int],
    wrong: Sequence[int | None],
    *,
    alpha: float,
    delta: float,
    rule: str,
    levels: Sequence[tuple[str, float, bool]],
    n_valid: int,
    no_confidence: bool = False,
) -> Rendered:
    """Which part to trust: the certified zone, the labels, and the levels tested.

    ``zone`` holds the certified windows (empty when none was certified);
    ``alpha``, ``delta`` and ``rule`` are the test's, stated as the tool's
    result states them (the zone's error rate is at most alpha, a statement
    that fails on at most delta of random draws); ``levels`` are ``(label,
    upper bound, certified)`` per level tested, as the rule decided them
    (under a multi-level rule a bound below ``alpha`` can still fail);
    ``wrong`` is as for :func:`estimate_figure`. ``no_confidence`` marks a
    class map without top-k scores (:func:`_shade`).
    """
    rows, cols, given = _layout(margin, grid)
    base = tint(confidence_base(margin, rows * cols, faded=0.6), zone, ZONE, 0.55)
    pairs = list(zip(sample, wrong))
    right = [i for i, w in pairs if w == 0]
    bad = [i for i, w in pairs if w == 1]
    share = pct(len(zone) / max(n_valid, 1))
    holds = (
        f"with a statement that fails on at most {level(delta)} of random draws "
        f"({rule} rule)"
    )
    if zone:
        subtitle = (
            f"{len(zone):,} of {n_valid:,} windows ({share}) certified at an "
            f"error rate of at most {level(alpha)}, {holds}."
        )
        verdict = (
            f"The certified zone (blue) holds the map's {len(zone):,} most "
            f"confident windows ({share} of {n_valid:,}), where the error rate "
            f"is at most {level(alpha)} {holds}."
        )
    else:
        subtitle = (
            f"No zone certified at an error rate of at most {level(alpha)} "
            f"with these labels ({rule} rule, delta {level(delta)})."
        )
        verdict = (
            f"The {rule} rule certified no level with these labels, so no zone "
            "is given."
        )
    legend, shading = _shade(margin, no_confidence=no_confidence)
    fig = GridFigure(
        "Which part of the map to trust",
        subtitle,
        rows,
        cols,
        given,
        base,
        caption=[
            f"Which part of the map to trust. (a) The map's {n_valid:,} windows"
            f"{_grid_words(rows, cols, given)}, shown {shading}; the random "
            f"sample's labels: green right ({len(right):,}), magenta "
            f"wrong ({len(bad):,}). {verdict}"
            + (
                " (b) The upper bound on the error rate at each confidence level "
                f"tested, with alpha at {level(alpha)}; blue marks the levels the "
                f"{rule} rule certified, grey those it did not (a rule that tests "
                "several levels can refuse a bound below alpha)."
                if levels
                else ""
            )
            + " The guarantee rests on the random sample's labels being right."
        ],
    )
    fig.legend("ramp", "", legend)
    if zone:
        fig.legend("fill", css(ZONE), "certified zone")
    fig.mark(right, RIGHT, label="labelled right")
    fig.mark(bad, WRONG, label="labelled wrong")
    if levels:
        fig.bar_chart(
            "Upper bound on the error, per level",
            [
                (label, bound, css(ZONE) if certified else NEITHER)
                for label, bound, certified in levels
            ],
            limit=alpha,
            limit_label=f"alpha {level(alpha)}",
            as_percent=True,
        )
    return _done(fig)


def two_map_plan_figure(
    differ: Sequence[bool | None],
    grid: Sequence[int] | None,
    sample: Sequence[int],
) -> Rendered:
    """Which of two maps: where they differ, and the differing windows drawn to label.

    ``differ`` holds, per grid cell, whether the two maps' classes differ
    there (``None`` where either map has no class).
    """
    rows, cols, given = _layout(differ, grid)
    n_compared = sum(1 for d in differ if d is not None)
    n_differ = sum(1 for d in differ if d)
    n = len(sample)
    fig = GridFigure(
        "Where the two maps differ",
        f"They differ on {n_differ:,} of the {n_compared:,} windows compared; "
        f"{n:,} of those are drawn to label.",
        rows,
        cols,
        given,
        mask_base(differ, rows * cols, on=DIFFER, off=AGREE),
        caption=[
            f"Where the two maps differ. (a) The {n_compared:,} windows compared"
            f"{_grid_words(rows, cols, given)}: amber where the two maps' classes "
            f"differ ({n_differ:,}), grey where they agree; the {n:,} outlined "
            "windows are drawn at random among the differing ones to label. Where "
            "the maps agree, both are right or both are wrong, so only the "
            "differing windows can say which map is more accurate."
        ],
    )
    fig.legend("fill", css(DIFFER), "the maps differ")
    fig.legend("fill", css(AGREE), "the maps agree")
    if any(d is None for d in differ):
        fig.legend("fill", css(NO_DATA), "not compared")
    fig.mark(sample, OUTLINE, label="drawn to label")
    return _done(fig)


def two_map_grade_figure(
    differ: Sequence[bool | None],
    grid: Sequence[int] | None,
    sample: Sequence[int],
    sides: Sequence[str],
    *,
    verdict: str,
    differ_known: bool = True,
) -> Rendered:
    """Which of two maps: the side each label took, and the verdict it supports.

    ``sides`` holds, per drawn window in ``sample``'s order, ``a`` (the label
    is map a's class), ``b``, ``neither`` or ``?`` (not judged); ``verdict``
    is the package's verdict sentence, quoted as it is. The labelled windows
    count the ``?`` ones, as the verdict's ``n_labelled`` does; the judged
    ones do not. ``differ_known`` is false for a design file written before
    the plan recorded where the maps differ: only its drawn windows are
    known to differ, and the caption says so.
    """
    rows, cols, given = _layout(differ, grid)
    pairs = list(zip(sample, sides))
    groups = {
        s: [i for i, side in pairs if side == s] for s in ("a", "b", "neither", "?")
    }
    n_a, n_b, n_none = len(groups["a"]), len(groups["b"]), len(groups["neither"])
    n_unjudged = len(groups["?"])
    n_judged = n_a + n_b + n_none
    n_labelled = n_judged + n_unjudged
    counted = (
        f"The {n_judged:,} judged windows of the {n_labelled:,} labelled"
        if n_unjudged
        else f"The {n_labelled:,} labelled windows"
    )
    unrecorded = (
        ""
        if differ_known
        else " This design file does not record where the maps differ elsewhere, "
        "so only the labelled windows are shown."
    )
    differ_shade = fade(DIFFER, 0.45)
    fig = GridFigure(
        "Which of the two maps is more accurate",
        verdict,
        rows,
        cols,
        given,
        mask_base(differ, rows * cols, on=differ_shade, off=AGREE),
        caption=[
            f"Which of the two maps is more accurate. (a) {counted} where the maps "
            f"differ{_grid_words(rows, cols, given)}: blue where the label is map "
            f"a's class ({n_a:,}), orange map b's ({n_b:,}), grey neither "
            f"({n_none:,})"
            + (f", outlined not judged ({n_unjudged:,})" if n_unjudged else "")
            + f".{unrecorded} (b) The same counts. {verdict} The comparison covers "
            "only the windows where the maps differ and says nothing of either "
            "map's own accuracy."
        ],
    )
    fig.legend("fill", css(differ_shade), "the maps differ")
    if differ_known:
        fig.legend("fill", css(AGREE), "the maps agree")
    if any(d is None for d in differ):
        fig.legend("fill", css(NO_DATA), "not compared")
    fig.mark(groups["a"], MAP_A, filled=True, label=f"label is map a's ({n_a:,})")
    fig.mark(groups["b"], MAP_B, filled=True, label=f"label is map b's ({n_b:,})")
    fig.mark(groups["neither"], NEITHER, filled=True, label=f"neither's ({n_none:,})")
    if groups["?"]:
        fig.mark(groups["?"], UNJUDGED, label=f"not judged ({len(groups['?']):,})")
    fig.bar_chart(
        "Labels where the maps differ",
        [
            ("map a right", n_a, MAP_A),
            ("map b right", n_b, MAP_B),
            ("neither", n_none, NEITHER),
        ],
    )
    return _done(fig)


# --------------------------------------------------------------------------- writing


def write_figure(stem: str, figure: Rendered) -> str:
    """Write ``figure`` as ``<stem>_figure.svg`` in the write root; return the path."""
    from olmoearth_agent.security.paths import safe_path
    from olmoearth_agent.tools.review_set import write_root

    target = safe_path(f"{stem}_figure.svg", root=write_root())
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(figure.svg, encoding="utf-8")
    return str(target)


def figure_fields(stem: str, figure: Rendered) -> dict[str, str]:
    """A tool result's figure keys: ``figure_path`` and ``figure_caption``."""
    return {"figure_path": write_figure(stem, figure), "figure_caption": figure.caption}
