"""The v2 input contract for the Connect Four one-pass scorer (encoding "E1").

One representation everywhere (corpus, trainer, protocol, ONNX self-test, browser):

    board   uint8 [42], row-major, BOTTOM row first; 0 = empty, 1 = the player to move ("mine"),
            2 = the opponent ("theirs"). Perspective-normalised: the model never has to work out
            whose stone is whose from colours.
    context "<side>:<cells>" - 44 ASCII bytes. <side> is "1" when the player to move started the
            game, "2" otherwise (parity matters for odd/even threats); <cells> is the board as
            42 characters from ".mt". Every cell is one byte at a *fixed* position, i.e. one token
            at a constant position for the byte-level encoder - no separators, no move history.
    options "column 1" ... "column 7" for the legal columns (8 bytes each).

The byte ids follow the toolkit rule (onepass/encode.py): UTF-8 byte + 1, 0 = padding.
Scores are connect-four-ai scores for each column (positive = win, 0 = draw, negative = loss);
ILLEGAL marks a full column in int8 arrays.
"""
from __future__ import annotations

import numpy as np

CTX_BYTES = 44
OPTION_BYTES = 8
MAX_OPTIONS = 7
ILLEGAL = -128
CELL_CHARS = np.frombuffer(b".mt", dtype=np.uint8)
OPTION_TEXT = [f"column {c + 1}" for c in range(7)]


# ---------------------------------------------------------------- boards

def board_from_moves(moves) -> np.ndarray:
    """0-indexed column sequence -> board (perspective of the player to move)."""
    grid = np.zeros((6, 7), dtype=np.uint8)
    heights = [0] * 7
    for index, col in enumerate(moves):
        grid[heights[col], col] = 1 + (index % 2)  # 1 = first player, 2 = second player (absolute)
        heights[col] += 1
    if len(moves) % 2 == 1:  # second player to move: swap to the mover's perspective
        grid = np.where(grid == 0, 0, 3 - grid).astype(np.uint8)
    return grid.reshape(42)


def board_to_c4label(board: np.ndarray) -> str:
    """`b:` line for c4label: top row first, x = to move, o = opponent."""
    grid = np.asarray(board).reshape(6, 7)[::-1]
    return "b:" + "".join(".xo"[v] for v in grid.reshape(42))


def boards_from_tonyc(obs: np.ndarray) -> np.ndarray:
    """TonyCWang obs [N, 2, 6, 7] (channel 0 = to move, row 0 = TOP) -> boards [N, 42]."""
    mine = obs[:, 0, ::-1, :] > 0
    theirs = obs[:, 1, ::-1, :] > 0
    return (mine * 1 + theirs * 2).astype(np.uint8).reshape(len(obs), 42)


def mirror_boards(boards: np.ndarray) -> np.ndarray:
    return boards.reshape(-1, 6, 7)[:, :, ::-1].reshape(-1, 42)


def canonical_keys(boards: np.ndarray) -> np.ndarray:
    """Symmetry-reduced uint64 key per board (Pons-style position+mask on a 7-bit column layout)."""
    def key(b):
        grid = b.reshape(-1, 6, 7).astype(np.uint64)
        bit = np.zeros(grid.shape[1:], dtype=np.uint64)
        for col in range(7):
            for row in range(6):
                bit[row, col] = np.uint64(1) << np.uint64(col * 7 + row)
        mine = ((grid == 1) * bit).sum(axis=(1, 2), dtype=np.uint64)
        mask = ((grid > 0) * bit).sum(axis=(1, 2), dtype=np.uint64)
        return mine + mask
    return np.minimum(key(boards), key(mirror_boards(boards)))


def plies(boards: np.ndarray) -> np.ndarray:
    return (boards.reshape(-1, 42) > 0).sum(1).astype(np.uint8)


# ---------------------------------------------------------------- the model's bytes

def contexts(boards: np.ndarray) -> np.ndarray:
    """boards [N, 42] -> context bytes [N, 44] (raw ASCII, not yet +1)."""
    boards = boards.reshape(-1, 42)
    out = np.empty((len(boards), CTX_BYTES), dtype=np.uint8)
    first = plies(boards) % 2 == 0
    out[:, 0] = np.where(first, ord("1"), ord("2"))
    out[:, 1] = ord(":")
    out[:, 2:] = CELL_CHARS[boards]
    return out


def context_text(board: np.ndarray) -> str:
    return contexts(np.asarray(board).reshape(1, 42))[0].tobytes().decode("ascii")


def option_ids_table() -> np.ndarray:
    """[7, OPTION_BYTES] byte ids (+1) of "column k", in column order."""
    table = np.zeros((7, OPTION_BYTES), dtype=np.int64)
    for c, text in enumerate(OPTION_TEXT):
        raw = text.encode("ascii")
        table[c, : len(raw)] = np.frombuffer(raw, dtype=np.uint8) + 1
    return table


def legal_mask(boards: np.ndarray) -> np.ndarray:
    top = boards.reshape(-1, 6, 7)[:, 5, :]
    return top == 0
