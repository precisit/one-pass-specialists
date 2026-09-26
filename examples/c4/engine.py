"""Connect Four board and the hand-written depth-limited bots used as protocol opponents.

`bot:d` in `protocol.py` is `bot_move(position, d)`: alpha-beta to depth d with a cheap heuristic
leaf (immediate threats plus centre control), centre-first move ordering, deterministic. It is the
same opponent the first version of this model was measured against, kept unchanged so that the
numbers of both versions are comparable. Exact labels come from `c4label` (see `solver.py`), not
from this file.

Bitboard layout (one extra row per column as an overflow sentinel):

        6 13 20 27 34 41 48   <- sentinel row
        5 12 19 26 33 40 47
        4 11 18 25 32 39 46
        3 10 17 24 31 38 45
        2  9 16 23 30 37 44
        1  8 15 22 29 36 43
        0  7 14 21 28 35 42
"""

from __future__ import annotations

from dataclasses import dataclass

WIDTH = 7
HEIGHT = 6
COLUMN_MASK = 0x40810204081  # lowest bit of each column
BOARD_MASK = sum(1 << (col * 7 + row) for col in range(WIDTH) for row in range(HEIGHT))
BOTTOM_MASK = sum(1 << (col * 7) for col in range(WIDTH))

# ------------------------------------------------------------------ position


@dataclass(frozen=True)
class Position:
    """`current` is the player to move; `mask` the union of both players' stones."""

    current: int
    mask: int
    moves: tuple[int, ...] = ()

    @property
    def opponent(self) -> int:
        return self.current ^ self.mask

    @property
    def stones(self) -> int:
        return bin(self.mask).count("1")

    @property
    def player_to_move_is_x(self) -> bool:
        return self.stones % 2 == 0

    def key(self) -> int:
        """Symmetry-reduced key: a board and its horizontal mirror are the same position."""
        encoding = self.current | (self.mask << 49)
        mirrored_current = 0
        mirrored_mask = 0
        for col in range(WIDTH):
            target = (WIDTH - 1 - col) * 7
            mirrored_current |= ((self.current >> (col * 7)) & 0x3F) << target
            mirrored_mask |= ((self.mask >> (col * 7)) & 0x3F) << target
        mirror = mirrored_current | (mirrored_mask << 49)
        return min(encoding, mirror)

    # -- mechanics ---------------------------------------------------------

    def legal_columns(self) -> list[int]:
        top = (self.mask >> 5) & COLUMN_MASK
        return [col for col in range(WIDTH) if not (top >> (col * 7)) & 1]

    def play(self, column: int) -> "Position":
        if column not in self.legal_columns():
            raise ValueError(f"column {column} is full")
        stone = (self.mask + (1 << (column * 7))) & (0x3F << (column * 7))
        return Position(self.opponent, self.mask | stone, self.moves + (column,))

    def winning_move(self, column: int) -> bool:
        """Does playing `column` win immediately for the player to move?

        Only meaningful for a *playable* column: on a full column the overflow sentinel row can make
        the alignment test fire. That is not hypothetical: it is what made the first version's test
        suite report a lost position as won the first time it ran. Use `winning_moves()` instead
        when you want the legal ones.
        """
        if column not in self.legal_columns():
            return False
        stone = (self.mask + (1 << (column * 7))) & (0x3F << (column * 7))
        return alignment(self.current | stone)

    def is_full(self) -> bool:
        return self.mask == BOARD_MASK

    def has_won(self) -> bool:
        """True if the player who just moved (the opponent of the player to move) has four."""
        return alignment(self.opponent)

    def render(self) -> str:
        """A compact text board, top row first, as the model sees it.

        Rendered from the *move sequence*, not from the bitboards: the bitboards are stored from the
        point of view of the player to move, so a render taken from them flips identity with every
        turn. A perspective that flips every turn is the kind of silent corruption that costs accuracy
        without ever raising.
        """
        grid = [[None] * WIDTH for _ in range(HEIGHT)]
        heights = [0] * WIDTH
        for index, column in enumerate(self.moves):
            player = "X" if index % 2 == 0 else "O"
            grid[heights[column]][column] = player
            heights[column] += 1
        rows = []
        for row in range(HEIGHT - 1, -1, -1):
            rows.append("".join(cell or "." for cell in grid[row]))
        return "\n".join(rows)


def alignment(stones: int) -> bool:
    """Four in a row (any direction) for `stones`?"""
    for shift in (1, 7, 6, 8):
        pairs = stones & (stones >> shift)
        if pairs & (pairs >> (2 * shift)):
            return True
    return False


# ------------------------------------------------------------------ a depth-limited opponent

def heuristic(position: Position) -> int:
    """Cheap leaf score for the depth-limited bots: immediate threats plus centre control.

    Deliberately shallow. The point of the comparison is a *hand-coded* opponent, not a good one.
    """
    score = 0
    for column in range(WIDTH):
        if column not in position.legal_columns():
            continue
        if position.winning_move(column):
            score += 12
        opponent_threat = Position(position.opponent, position.mask).winning_move(column)
        if opponent_threat:
            score -= 10
    score += 2 * bin(position.current & 0x1C1C1C1C1C1C).count("1")
    score -= 2 * bin(position.opponent & 0x1C1C1C1C1C1C).count("1")
    return score


def bot_move(position: Position, depth: int) -> int:
    """Alpha-beta with `depth` plies and a heuristic leaf score: the 'hand-coded AI' to beat."""
    best_score = -10**9
    best_column = position.legal_columns()[0]
    for column in sorted(position.legal_columns(), key=lambda col: abs(col - 3)):
        if position.winning_move(column):
            return column
        score = -_search(position.play(column), depth - 1, -10**9, 10**9)
        if score > best_score:
            best_score, best_column = score, column
    return best_column


def _search(position: Position, depth: int, alpha: int, beta: int) -> int:
    if position.has_won():
        return -1000 + (42 - position.stones)
    if position.is_full() or depth == 0:
        return heuristic(position)
    best = -10**9
    for column in position.legal_columns():
        score = -_search(position.play(column), depth - 1, -beta, -alpha)
        if score > best:
            best = score
        alpha = max(alpha, best)
        if alpha >= beta:
            break
    return best
