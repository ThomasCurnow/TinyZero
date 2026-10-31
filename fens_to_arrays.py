"""
FEN -> input planes for the MLP evaluator.

Two layouts, both absolute colours (no flipping; row 0 = rank 1, col 0 = file a):

 v1, 20 planes (legacy, unchanged: your existing models use this)
   0-5   white pieces: P N B R Q K            6-11  black pieces: P N B R Q K
   12-15 castling rights: white K, white Q, black K, black Q
   16    en passant target square
   17    squares attacked by white (includes defended white pieces)
   18    squares attacked by black
   19    side to move (all ones = White to move)

 v2, 28 planes = the 20 above, unchanged, plus
   20    white pieces that are hanging (attacked by Black, not defended; kings excluded)
   21    black pieces that are hanging
   22    white pieces attacked by a lower-valued black piece (P<N=B<R<Q)
   23    black pieces attacked by a lower-valued white piece
   24    squares attacked at least twice by White
   25    squares attacked at least twice by Black
   26    side to move is in check (all ones / all zeros)
   27    passed pawns (either colour)

Because v2 starts with the v1 planes, old models keep working: encode_for_model()
looks at the model's input size (1280 -> v1, 1792 -> v2) and encodes to match.

CLI:  python fens_to_arrays.py labeled.tsv encoded.tsv [--planes 28] [--workers N]
      python fens_to_arrays.py --test
"""
import argparse
import os
from functools import partial
from multiprocessing import Pool

import chess
import numpy as np

NUM_PLANES_V1 = 20
NUM_PLANES_V2 = 28
NUM_PLANES = NUM_PLANES_V1          # kept for older imports

PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
               chess.ROOK, chess.QUEEN, chess.KING]


def _bb_to_plane(bb):
    """64-bit bitboard -> (8, 8) uint8. Row 0 = rank 1, column 0 = file a."""
    bits = np.unpackbits(np.frombuffer(bb.to_bytes(8, "little"), dtype=np.uint8),
                         bitorder="little")
    return bits.reshape(8, 8)


def _attack_info(board, color):
    """(squares attacked once+, squares attacked twice+, {piece type: attack union})."""
    once = twice = 0
    by_type = {pt: 0 for pt in PIECE_ORDER}
    for sq in chess.scan_forward(board.occupied_co[color]):
        m = board.attacks_mask(sq)
        twice |= once & m
        once |= m
        by_type[board.piece_type_at(sq)] |= m
    return once, twice, by_type


def _lower_attacked(own, enemy_by):
    """own: {piece type: mask} of the defending side; enemy_by: attack unions of the other side.
    Returns own pieces attacked by an enemy piece of lower value."""
    p, n, b, r = (enemy_by[chess.PAWN], enemy_by[chess.KNIGHT],
                  enemy_by[chess.BISHOP], enemy_by[chess.ROOK])
    return (((own[chess.KNIGHT] | own[chess.BISHOP]) & p)
            | (own[chess.ROOK] & (p | n | b))
            | (own[chess.QUEEN] & (p | n | b | r)))


def _passed_pawns(board):
    wp = board.pieces_mask(chess.PAWN, chess.WHITE)
    bp = board.pieces_mask(chess.PAWN, chess.BLACK)
    out = 0
    for color, mine, theirs in ((chess.WHITE, wp, bp), (chess.BLACK, bp, wp)):
        for sq in chess.scan_forward(mine):
            f, r = sq & 7, sq >> 3
            files = chess.BB_FILES[f]
            if f > 0:
                files |= chess.BB_FILES[f - 1]
            if f < 7:
                files |= chess.BB_FILES[f + 1]
            ahead = (chess.BB_ALL & ~((1 << (8 * (r + 1))) - 1) if color == chess.WHITE
                     else (1 << (8 * r)) - 1)
            if not (theirs & files & ahead):
                out |= 1 << sq
    return out


def fen_to_planes_n(fen, n_planes=NUM_PLANES_V2):
    """(n_planes, 8, 8) uint8 array; n_planes is 20 (v1) or 28 (v2)."""
    if n_planes not in (NUM_PLANES_V1, NUM_PLANES_V2):
        raise ValueError(f"n_planes must be 20 or 28, got {n_planes}")
    board = chess.Board(fen)
    planes = np.zeros((n_planes, 8, 8), dtype=np.uint8)

    for i, pt in enumerate(PIECE_ORDER):
        planes[i] = _bb_to_plane(board.pieces_mask(pt, chess.WHITE))
        planes[6 + i] = _bb_to_plane(board.pieces_mask(pt, chess.BLACK))

    planes[12] = board.has_kingside_castling_rights(chess.WHITE)
    planes[13] = board.has_queenside_castling_rights(chess.WHITE)
    planes[14] = board.has_kingside_castling_rights(chess.BLACK)
    planes[15] = board.has_queenside_castling_rights(chess.BLACK)
    if board.ep_square is not None:
        planes[16] = _bb_to_plane(1 << board.ep_square)

    w_once, w_twice, w_by = _attack_info(board, chess.WHITE)
    b_once, b_twice, b_by = _attack_info(board, chess.BLACK)
    planes[17] = _bb_to_plane(w_once)
    planes[18] = _bb_to_plane(b_once)
    planes[19] = board.turn == chess.WHITE
    if n_planes == NUM_PLANES_V1:
        return planes

    w_own = {pt: board.pieces_mask(pt, chess.WHITE) for pt in PIECE_ORDER}
    b_own = {pt: board.pieces_mask(pt, chess.BLACK) for pt in PIECE_ORDER}
    w_nonking = board.occupied_co[chess.WHITE] & ~board.kings
    b_nonking = board.occupied_co[chess.BLACK] & ~board.kings
    planes[20] = _bb_to_plane(w_nonking & b_once & ~w_once)
    planes[21] = _bb_to_plane(b_nonking & w_once & ~b_once)
    planes[22] = _bb_to_plane(_lower_attacked(w_own, b_by))
    planes[23] = _bb_to_plane(_lower_attacked(b_own, w_by))
    planes[24] = _bb_to_plane(w_twice)
    planes[25] = _bb_to_plane(b_twice)
    planes[26] = board.is_check()
    planes[27] = _bb_to_plane(_passed_pawns(board))
    return planes


def fen_to_planes(fen):
    """Legacy 20-plane encoding (v1). Existing code and old models use this."""
    return fen_to_planes_n(fen, NUM_PLANES_V1)


def fen_to_planes_v2(fen):
    """28-plane encoding (v1 planes + tactical planes)."""
    return fen_to_planes_n(fen, NUM_PLANES_V2)


def encode_for_model(model, fen):
    """(1, n_features) float32 input matching the model's first layer (1280 -> v1, 1792 -> v2)."""
    n_features = model.coefs_[0].shape[0]
    return fen_to_planes_n(fen, n_features // 64).reshape(1, -1).astype("float32")


def encode_line(line, n_planes=NUM_PLANES_V2):
    """'fen<TAB>score' -> '<bit string><TAB>score', or None if bad."""
    parts = line.rstrip("\n").split("\t")
    if len(parts) != 2:
        return None
    fen, score = parts
    try:
        int(score)  # skips "NA" rows
        planes = fen_to_planes_n(fen, n_planes)
    except Exception:
        return None
    bits = (planes.ravel() + 48).tobytes().decode("ascii")      # '0'/'1' characters
    return f"{bits}\t{score}\n"


def self_test():
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    p1, p2 = fen_to_planes(start), fen_to_planes_v2(start)
    assert p1.shape == (20, 8, 8) and p2.shape == (28, 8, 8)
    assert (p2[:20] == p1).all(), "v2 must start with the v1 planes"
    q = fen_to_planes(start.replace(" w ", " b "))
    assert [i for i in range(20) if not (p1[i] == q[i]).all()] == [19]
    assert p1[:6, :2].sum() == 16 and p1[:6, 2:].sum() == 0
    # hanging knight on c3 (attacked by the bishop on a1, undefended)
    t = fen_to_planes_v2("4k3/8/8/8/8/2N5/8/b3K3 w - - 0 1")
    assert t[20][2][2] == 1 and t[20].sum() == 1
    # queen on e4 attacked by a pawn
    t = fen_to_planes_v2("4k3/8/8/3p4/4Q3/8/8/4K3 w - - 0 1")
    assert t[22][3][4] == 1 and t[22].sum() == 1
    # passed pawn
    assert fen_to_planes_v2("4k3/8/8/4P3/8/8/8/4K3 w - - 0 1")[27][4][4] == 1
    assert fen_to_planes_v2("4k3/4p3/8/4P3/8/8/8/4K3 w - - 0 1")[27].sum() == 0
    print("checks passed")


if __name__ == "__main__":  # required on Windows
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", help="TSV of fen<TAB>score")
    ap.add_argument("output", nargs="?", help="TSV of bitstring<TAB>score")
    ap.add_argument("--planes", type=int, choices=(20, 28), default=28)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()

    if a.test or not (a.input and a.output):
        self_test()
        raise SystemExit

    n_in = n_out = 0
    with open(a.input, "r", encoding="utf-8") as fin, \
         open(a.output, "w", encoding="utf-8") as fout, \
         Pool(a.workers) as pool:
        for res in pool.imap(partial(encode_line, n_planes=a.planes), fin, chunksize=1000):
            n_in += 1
            if res is not None:
                fout.write(res)
                n_out += 1
            if n_in % 100000 == 0:
                print(f"{n_in} rows read, {n_out} written", flush=True)
    print(f"Done: {n_out}/{n_in} rows written ({a.planes} planes = {a.planes * 64} features)")