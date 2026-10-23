"""Tiny bitmap letters for the visible watermark.

OpenCV's Hershey font cannot draw Chinese. These six glyphs are original
pixel art for the default Chinese label. Anything else can use a font file
supplied in the config. No third-party font is bundled.
"""

from __future__ import annotations

import numpy as np

# 12 rows. '#' is ink. The shapes are original and only need to be readable.
_ROWS: dict[str, tuple[str, ...]] = {
    "人": (
        "....#.......",
        "....#.......",
        "...#.#......",
        "...#.#......",
        "..#...#.....",
        "..#...#.....",
        ".#.....#....",
        ".#.....#....",
        "#.......#...",
        "#.......#...",
        "............",
        "............",
    ),
    "工": (
        "###########.",
        "............",
        "....##......",
        "....##......",
        "....##......",
        "....##......",
        "....##......",
        "............",
        "###########.",
        "............",
        "............",
        "............",
    ),
    "智": (
        ".###...###..",
        ".#.#...#.#..",
        ".###...###..",
        "............",
        "##########..",
        "...#....#...",
        "...#....#...",
        "##########..",
        "...#....#...",
        "...#....#...",
        "............",
        "............",
    ),
    "能": (
        "#..#..####..",
        "#..#..#..#..",
        "####..####..",
        "#..#..#..#..",
        "#..#..####..",
        "............",
        "####..#..#..",
        "#..#..####..",
        "####..#..#..",
        "#..#..#..#..",
        "............",
        "............",
    ),
    "生": (
        ".....#......",
        ".....#......",
        "###########.",
        ".....#......",
        ".....#......",
        ".#########..",
        ".....#......",
        ".....#......",
        "###########.",
        "............",
        "............",
        "............",
    ),
    "成": (
        "....#####...",
        "...#....#...",
        "..#.....#...",
        ".#..##..#...",
        "....#...#...",
        "...#....#...",
        "..#.#...#...",
        ".#...#..#...",
        "#.....#.#...",
        ".......#....",
        "............",
        "............",
    ),
}

GLYPHS: dict[str, np.ndarray] = {
    char: np.array([[1 if cell == "#" else 0 for cell in row] for row in rows], dtype=np.uint8)
    for char, rows in _ROWS.items()
}


def covers(text: str) -> bool:
    """True when every character is ASCII or one of the built-in glyphs."""

    return all(ord(char) < 128 or char in GLYPHS for char in text)


def blit(canvas: np.ndarray, glyph: np.ndarray, x: int, y: int, scale: int, color: tuple[int, int, int]) -> None:
    """Paint ``glyph`` onto ``canvas``. Clipped at the edges."""

    height, width = canvas.shape[:2]
    scale = max(1, int(scale))
    for row in range(glyph.shape[0]):
        for col in range(glyph.shape[1]):
            if glyph[row, col] == 0:
                continue
            y0 = y + row * scale
            x0 = x + col * scale
            y1 = min(height, y0 + scale)
            x1 = min(width, x0 + scale)
            if y0 >= height or x0 >= width or y1 <= 0 or x1 <= 0:
                continue
            canvas[max(0, y0) : y1, max(0, x0) : x1] = color
