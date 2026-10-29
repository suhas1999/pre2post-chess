"""Rebuild the puzzle test parquets (test_B1..B5_multi_turn) for rl/verl/eval_bash/verl_eval.sh.

The test parquets are not released, but chess-pre-to-post/chess-rl-eval has the generations
of a released eval run, and each record carries the prompt (as model tokens), the player's
solution moves and the benchmark name. The opponent replies the env feeds back between
moves are not in those records, so each puzzle is looked up in the Lichess puzzle database
(by position before the setup move + setup move + first solution move) to recover them.

    python cloud/rebuild_test_sets.py --generations GEN.jsonl --puzzle-csv lichess_db_puzzle.csv \
        --out-dir /workspace/data/puzzles
"""

import argparse
import ast
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import chess
import pandas as pd

LAN_RE = re.compile(r"^[KQRBNP]([a-h][1-8])x?([a-h][1-8])(?:=([QRBN]))?[+#]?$")


def parse_lan(board, token):
    """Model LAN token (e.g. 'Pd4xc5', 'Ng1e2', 'Pe7e8=Q+', 'O-O') -> chess.Move."""
    if token.rstrip("+#") in ("O-O", "O-O-O"):
        return board.parse_san(token.rstrip("+#"))
    m = LAN_RE.match(token)
    if not m:
        raise ValueError(f"bad LAN token {token!r}")
    promo = chess.Piece.from_symbol(m.group(3).lower()).piece_type if m.group(3) else None
    move = chess.Move(chess.parse_square(m.group(1)), chess.parse_square(m.group(2)), promotion=promo)
    if move not in board.legal_moves:
        raise ValueError(f"illegal {token!r} in {board.fen()}")
    return move


def to_lan(board, move):
    """chess.Move -> model LAN as used in the training env_replies (e.g. 'Nf1xh2', 'Qc5xd4+')."""
    if board.is_castling(move):
        text = "O-O" if chess.square_file(move.to_square) == 6 else "O-O-O"
    else:
        text = board.piece_at(move.from_square).symbol().upper() + chess.square_name(move.from_square)
        text += ("x" if board.is_capture(move) else "") + chess.square_name(move.to_square)
        if move.promotion:
            text += "=" + chess.piece_symbol(move.promotion).upper()
    board.push(move)
    suffix = "#" if board.is_checkmate() else "+" if board.is_check() else ""
    board.pop()
    return text + suffix


def movetext(moves):
    """SAN movetext from the start position, e.g. '1. e4 e5 2. Nf3'."""
    board, parts = chess.Board(), []
    for i, move in enumerate(moves):
        if i % 2 == 0:
            parts.append(f"{i // 2 + 1}.")
        parts.append(board.san(move))
        board.push(move)
    return " ".join(parts)


def key(board, setup_uci, first_uci):
    return f"{board.board_fen()} {'w' if board.turn else 'b'} {setup_uci} {first_uci}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", required=True, help="a generations/*.jsonl from chess-rl-eval")
    ap.add_argument("--puzzle-csv", required=True, help="lichess_db_puzzle.csv (decompressed)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    # 1. Unique prompts from the released generations.
    puzzles = {}
    with open(args.generations) as f:
        for line in f:
            r = json.loads(line)
            puzzles.setdefault((r["input"], r["ground_truth"]), r["data_source"])
    print(f"{len(puzzles)} unique prompts: {dict(sorted(Counter(puzzles.values()).items()))}")

    # 2. Replay each prompt; index by position before the setup move.
    wanted = defaultdict(list)
    for (inp, gt), source in puzzles.items():
        tokens = inp.replace("<T>", "").split()
        board, moves = chess.Board(), []
        for tok in tokens:
            move = parse_lan(board, tok)
            moves.append(move)
            board.push(move)
        setup = board.pop()
        gt_moves = ast.literal_eval(gt)
        wanted[key(board, setup.uci(), gt_moves[0])].append((inp, gt, source, moves, gt_moves))

    # 3. Scan the Lichess puzzle DB for those keys.
    found = {}
    with open(args.puzzle_csv, newline="") as f:
        for row in csv.DictReader(f):
            uci = row["Moves"].split()
            if len(uci) < 2:
                continue
            board = chess.Board(row["FEN"])
            k = key(board, uci[0], uci[1])
            if k in wanted and k not in found:
                found[k] = row
    print(f"matched {len(found)}/{len(wanted)} positions in the puzzle DB")

    # 4. Rows in the training-parquet format.
    rows, skipped = defaultdict(list), Counter()
    for k, entries in wanted.items():
        row = found.get(k)
        if row is None:
            skipped["no DB match"] += len(entries)
            continue
        uci = row["Moves"].split()
        for inp, gt, source, moves, gt_moves in entries:
            if uci[1::2] != gt_moves:
                skipped["solution mismatch"] += 1
                continue
            board = chess.Board(row["FEN"])
            board.push_uci(uci[0])
            replies = []
            for i, u in enumerate(uci[1:], start=1):
                move = chess.Move.from_uci(u)
                if i % 2 == 0:
                    replies.append(to_lan(board, move))
                board.push(move)
            rows[source].append({
                "data_source": source,
                "prompt": movetext(moves) + " <T>",
                "ability": "chess",
                "reward_model": {"ground_truth": str(gt_moves), "style": "rule"},
                "extra_info": {
                    "FEN": row["FEN"], "Moves": row["Moves"], "PuzzleId": row["PuzzleId"],
                    "Rating": row["Rating"], "Themes": row["Themes"], "env_replies": replies,
                    "index": len(rows[source]),
                },
            })
    if skipped:
        print(f"skipped: {dict(skipped)}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for source, rs in sorted(rows.items()):
        path = out / f"{source}_multi_turn.parquet"
        pd.DataFrame(rs).to_parquet(path)
        print(f"{path}: {len(rs)} prompts")


if __name__ == "__main__":
    main()
