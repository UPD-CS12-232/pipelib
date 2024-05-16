# pyright: strict

"""Helper library to implement the "Pipe Dream" game.

Aside from the global constants, the most important functions here are `generate_paths` (which
generate disjoint paths, for use in generating solvable puzzles) and `draw_pipe` (which draws a
pipe piece). The `DifficultySettings` dataclass as well as the `DIFFICULTY_SETTINGS` global constant
are also needed.

At initialization time, you should load the Pyxel resource file with the filename
`PIPE_RESOURCE_PATH` (equal to `"pipe.pyxres"`, which must be present in the game directory).

The following global constants are useful:

    DIM: int

        cell dimension size (width and height in pixels)

    DIJS: tuple[tuple[int, int], ...]

        vectors toward the four cardinal directions

    PIPE_RESOURCE_PATH: str

        Pyxel resource file that needs to be loaded

    TILEMAP_PIPE: int

        tilemap from the resource file to use

    COLOR_BG_PIPE: int

        background color of the resource file tilemaps

    SFX_PRESS_SUCCESS: int

        sound effect to be played when the player successfully rotates a pipe

    SFX_PRESS_FAIL: int

        sound effect to be played when the player attempts to rotate a pipe piece cannot be rotated
        (i.e., "endpoint" pipe pieces)

    SFX_INCREASED_ACTIVE_PIPES: int

        sound effect to be played when the player connects a pair of endpoint pipes

    SFX_WIN: int

        sound effect to be played when the game is won

    COLOR_TEXT_INACTIVE: int

        text color to use over "inactive" (blue) pipes

    COLOR_TEXT_ACTIVE: int

        text color to use over "active" (green) pipes

"""

from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import islice
from random import Random
from typing import Final, TypeVar

import pyxel


T = TypeVar('T')


# cell dimension size (width and height in pixels)
DIM: Final[int] = 16
# vectors toward the four cardinal directions
DIJS: Final[tuple[tuple[int, int], ...]] = (-1, 0), (0, +1), (+1, 0), (0, -1)
# Pyxel resource file that needs to be loaded
PIPE_RESOURCE_PATH: Final[str] = 'pipe.pyxres'
# tilemap from the resource file to use
TILEMAP_PIPE: Final[int] = 0
# background color of the resource file tilemaps
COLOR_BG_PIPE: Final[int] = 0
# sound effect to be played when the player successfully rotates a pipe
SFX_PRESS_SUCCESS: Final[int] = 5
# sound effect to be played when the player attempts to rotate a pipe piece cannot be rotated
# (i.e., "endpoint" pipe pieces)
SFX_PRESS_FAIL: Final[int] = 8
SFX_PRESS_FAIL_ALT: Final[int] = 6
# sound effect to be played when the player connects a pair of endpoint pipes
SFX_INCREASED_ACTIVE_PIPES: Final[int] = 7
# sound effect to be played when the game is won
SFX_WIN: Final[int] = 10
SFX_WIN_ALT: Final[int] = 9
# text color to use over "inactive" (blue) pipes
COLOR_TEXT_INACTIVE: Final[int] = 15
# text color to use over "active" (green) pipes
COLOR_TEXT_ACTIVE: Final[int] = 14


@dataclass
class DifficultySettings:
    """Settings for a particular difficulty."""
    r: int
    c: int
    endcl: int
    pathl: int
    distl: int
    attemptc: int


# settings for the four "difficulties" trivial, easy, medium and hard
DIFFICULTY_SETTINGS = {
    'trivial': DifficultySettings(r=4,  c=6,  endcl=4,  pathl=1, distl=1, attemptc=2),
    'easy':    DifficultySettings(r=6,  c=8,  endcl=5,  pathl=2, distl=1, attemptc=2),
    'medium':  DifficultySettings(r=12, c=15, endcl=9,  pathl=3, distl=2, attemptc=3),
    'hard':    DifficultySettings(r=16, c=20, endcl=12, pathl=8, distl=4, attemptc=3),
}


def dist(i1: int, j1: int, i2: int, j2: int) -> int:
    return abs(i1 - i2) + abs(j1 - j2)


def shuffled(rand: Random, seq: Iterable[T]) -> list[T]:
    seql = list(seq)
    rand.shuffle(seql)
    return seql


def randpop(rand: Random, seq: list[T]) -> T:
    if not seq: raise ValueError("The input sequence must not be empty")
    idx = rand.randrange(len(seq))
    seq[idx], seq[-1] = seq[-1], seq[idx]
    return seq.pop()


def _try_generating_paths(settings: DifficultySettings, rand: Random) -> list[list[tuple[int, int]]]:
    r = settings.r
    c = settings.c
    endcl = settings.endcl

    adj: dict[tuple[int, int], list[tuple[int, int]]] = {(i, j): [] for i in range(r) for j in range(c)}

    strands: list[list[tuple[int, int]]] = []
    strandi: dict[tuple[int, int], int] = {}

    while len(strandi) < r * c:
        seeds = shuffled(rand, ((i, j) for i in range(r) for j in range(c) if (i, j) not in strandi))

        seeds = seeds[:endcl]

        for i, j in seeds:
            strandi[i, j] = len(strands)
            strands.append([(i, j)])

        seeds += seeds

        while seeds:
            i, j = randpop(rand, seeds)
            assert (i, j) in strandi

            def good_neighbors() -> Iterator[tuple[int, int]]:
                for di, dj in DIJS:
                    if 0 <= (ni := i + di) < r and 0 <= (nj := j + dj) < c and (ni, nj) not in strandi:
                        yield ni, nj

            if neighs := [*good_neighbors()]:
                ni, nj = rand.choice(neighs)

                # attach now
                adj[i, j].append((ni, nj))
                adj[ni, nj].append((i, j))
                strandi[ni, nj] = strandi[i, j]
                strands[strandi[ni, nj]].append((ni, nj))
                seeds.append((ni, nj))

        strands = [strand for strand in strands if len(strand) > 1]

    def cleanse(strand: list[tuple[int, int]]) -> list[tuple[int, int]]:
        [scell, ecell] = [cell for cell in strand if len(adj[cell]) == 1]
        seq: deque[tuple[int, int]] = deque([scell, scell])
        while seq[-1] != ecell:
            [ccell] = {*adj[seq[-1]]} - {seq[-2]}
            seq.append(ccell)

        ct = min(len(seq) - settings.distl, 2)
        if ct > 0: ct = rand.randint(0, ct)
        for _ in range(ct):
            if rand.randrange(2):
                seq.popleft()
            else:
                seq.pop()
        seq.popleft()
        return list(seq)

    rand.shuffle(strands)
    strands.sort(key=len, reverse=True)

    return shuffled(rand, (cleanse(strand) for strand in islice(strands, endcl)))


def _acceptable(settings: DifficultySettings, paths: list[list[tuple[int, int]]]) -> bool:
    return all(len(path) >= settings.pathl and dist(*path[0], *path[-1]) >= settings.distl for path in paths)


def _quality(paths: list[list[tuple[int, int]]]) -> float:
    return sum(len(path)**1.5 for path in paths)


def _generate_acceptable_paths(settings: DifficultySettings, rand: Random) -> Iterator[list[list[tuple[int, int]]]]:
    while True:
        if _acceptable(settings, paths := _try_generating_paths(settings, rand)):
            yield paths


def generate_paths(settings: DifficultySettings, rand: Random = Random()) -> list[list[tuple[int, int]]]:
    """Generate a list of disjoint paths in a grid, given the difficulty settings.

    This is meant to be used to generate "solvable" pipe puzzles. The paths themselves can be used
    for the "solution" of the puzzle. (Other solutions may exist---the game must be able to handle
    those as well.)

    A path is assumed to be a sequence of cells, i.e., of type `list[tuple[int, int]]`.

    The `rand` argument is optional.
    """
    return max(islice(_generate_acceptable_paths(settings, rand), settings.attemptc), key=_quality)


def _ccw(seq1: tuple[T, ...], seq2: tuple[T, ...]) -> int:
    return 0 if seq1 == seq2 else 1 + _ccw(seq1, (*seq2[1:], *seq2[:1]))


def draw_pipe(x: int, y: int, *,
        active: bool = False,
        north: bool = False,
        south: bool = False,
        east: bool = False,
        west: bool = False,
        base: bool = False,
        bg: bool = True):
    """Draw a 'pipe' piece with the given configuration at (x, y).

    The pipe will be drawn with top-left corner (x, y), `DIM` pixels long and `DIM` pixels wide.
    
    The arguments `north`, `south`, `east` and `west` represent which directions the pipe
    connects/attaches to. For typical pipes, exactly two of these must  be `True`.

    The argument `base` represents whether or not this pipe piece represents an
    "endpoint" piece. If it is `True`, then this pipe piece must connect to exactly one neighbor.

    The `active` argument represents whether the pipe piece is highlighted green. It is intended to
    be used when the pipe piece is part of a path that connects two "endpoint" pieces.

    The `bg` argument represents whether or not to make the "black" piece transparent.
    """
    
    pyxel.pal()
    
    # palette swap if active is True
    if active:
        pyxel.pal(5, 3)
        pyxel.pal(6, 7)
        pyxel.pal(12, 11)

    if base and north + south + east + west != 1:
        raise ValueError("If 'base' is True, then the pipe piece must have exactly one opening")

    tx = 1 if base else _xoff[north, east, south, west]
    ty = _ccw(_xstart[north, east, south, west], (north, east, south, west))

    pyxel.blt(x, y, TILEMAP_PIPE, DIM * tx, DIM * ty, DIM, DIM, None if bg else COLOR_BG_PIPE)

    pyxel.pal()


CWDirs = tuple[bool, bool, bool, bool]


_xoff: dict[CWDirs, int] = {}
_xstart: dict[CWDirs, CWDirs] = {}

for xo, xb in [
        (0, (False, False, False, False)),
        (2, (True,  False, False, False)),
        (3, (True,  True,  False, False)),
        (4, (True,  False, True,  False)),
        (5, (True,  True,  True,  False)),
        (6, (True,  True,  True,  True)),
    ]:
    xc0, xc1, xc2, xc3 = xc = xb
    while xc not in _xoff:
        _xoff[xc] = xo
        _xstart[xc] = xb
        xc0, xc1, xc2, xc3 = xc = xc3, xc0, xc1, xc2
