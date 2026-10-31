"""
Train the MLP evaluator on the OLD encoded set (encoded.tsv) plus the NEW set
(fen_8M.tsv), validating only on the fixed validation blocks from the OLD set.

    python model.py encoded.tsv fen_8M.tsv
    python model.py encoded.tsv fen_8M.tsv --stride-new 2 --rows-per-epoch 1000000
    python model.py encoded.tsv fen_8M.tsv --hidden 1024 256 64 --prefix weights_wide

Both inputs are ENCODED files (1280-char 0/1 string <TAB> centipawns), as written
by encode.py. White-perspective labels, absolute-colour planes.
"""
import argparse
import glob
import os
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

STM_COL = 19 * 64          # any column of plane 19 (side to move: 1 = White to move)
# The number of input features is read from the data (1280 = 20 planes, 1792 = 28 planes).


# ----------------------------------------------------------------- weights I/O
def save_weights_tsv(model, prefix):
    """
    Save a trained sklearn MLPRegressor as TSV files:
      <prefix>_W0.tsv, <prefix>_b0.tsv, <prefix>_W1.tsv, ...
    W_i has shape (inputs, outputs); b_i is a single row of length outputs.
    """
    # Remove files from any earlier, differently-sized net with this prefix
    for old in glob.glob(glob.escape(prefix) + "_[Wb][0-9]*.tsv"):
        os.remove(old)
    for i, (W, b) in enumerate(zip(model.coefs_, model.intercepts_)):
        np.savetxt(f"{prefix}_W{i}.tsv", W, delimiter="\t", fmt="%.9g")
        np.savetxt(f"{prefix}_b{i}.tsv", b[None, :], delimiter="\t", fmt="%.9g")


def load_weights_tsv(prefix, **mlp_kwargs):
    """
    Load the TSVs written by save_weights_tsv and return a fitted
    sklearn MLPRegressor (ReLU hidden layers, linear output).
    Extra keyword arguments (batch_size, learning_rate_init, ...) are passed
    to MLPRegressor. warm_start=True is set so calling .fit() again continues
    from the loaded weights (optimizer state starts fresh).
    """
    Ws, bs, i = [], [], 0
    while os.path.exists(f"{prefix}_W{i}.tsv"):
        Ws.append(np.loadtxt(f"{prefix}_W{i}.tsv", delimiter="\t",
                             ndmin=2, dtype=np.float32))
        bs.append(np.loadtxt(f"{prefix}_b{i}.tsv", delimiter="\t",
                             ndmin=2, dtype=np.float32).reshape(-1))
        i += 1
    if not Ws:
        raise FileNotFoundError(f"No weight files found for prefix '{prefix}'")
    for i in range(1, len(Ws)):
        if Ws[i].shape[0] != Ws[i - 1].shape[1]:
            raise ValueError(f"Layer {i} takes {Ws[i].shape[0]} inputs but layer "
                             f"{i - 1} outputs {Ws[i - 1].shape[1]}. Stale files?")

    n_in = Ws[0].shape[0]
    hidden = tuple(W.shape[1] for W in Ws[:-1])
    mlp_kwargs.setdefault("warm_start", True)
    model = MLPRegressor(hidden_layer_sizes=hidden, activation="relu", **mlp_kwargs)

    # Throwaway 1-epoch fit so sklearn creates its fitted attributes, then overwrite
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        was_warm = model.warm_start
        model.set_params(max_iter=1, verbose=False, warm_start=False)
        model.fit(np.zeros((2, n_in), dtype=np.float32), np.zeros(2, dtype=np.float32))
        model.set_params(max_iter=mlp_kwargs.get("max_iter", 200),
                         verbose=mlp_kwargs.get("verbose", False),
                         warm_start=was_warm)
    model.coefs_ = Ws
    model.intercepts_ = bs
    return model


# --------------------------------------------------------------------- dataset
def _count_lines(path):
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def check_encoded(path):
    """Fail early if the file is not an encoded (bitstring, score) TSV. Returns the feature count."""
    with open(path, encoding="ascii", errors="replace") as f:
        first = f.readline().rstrip("\n").split("\t")
    bits = first[0]
    if len(first) != 2 or len(bits) < 20 * 64 or len(bits) % 64 or set(bits) - {"0", "1"}:
        raise SystemExit(f"{path} is not an encoded file (expected a 0/1 string of 64 x planes "
                         f"characters, then a score). If it holds FENs, run fens_to_arrays.py first.")
    return len(bits)


def load_dataset(path, cache=True, max_rows=None):
    """
    Load an encoded TSV (bitstring<TAB>cp per line) into numpy arrays.

    Returns:
      X  (N, 1280 or 1792) uint8 array of 0/1, planes flattened in fens_to_arrays.py order
      cp (N,)      float32 centipawns, WHITE perspective

    With cache=True the arrays are streamed straight into <path>.X.npy and
    <path>.cp.npy (RAM use stays flat even for 8M+ rows) and memory-mapped on
    later calls. Files are written under a temporary name and renamed at the end,
    so an interrupted load can never leave a half-written cache behind.
    The cache is rebuilt if the TSV is newer. max_rows disables the cache.
    """
    x_path, cp_path = path + ".X.npy", path + ".cp.npy"
    use_cache = cache and max_rows is None
    n_features = check_encoded(path)

    if (use_cache and os.path.exists(x_path) and os.path.exists(cp_path)
            and os.path.getmtime(x_path) >= os.path.getmtime(path)):
        return np.load(x_path, mmap_mode="r"), np.load(cp_path, mmap_mode="r")

    n = _count_lines(path)
    if max_rows is not None:
        n = min(n, max_rows)

    if use_cache:
        X = np.lib.format.open_memmap(x_path + ".tmp", mode="w+", dtype=np.uint8,
                                      shape=(n, n_features))
        cp = np.lib.format.open_memmap(cp_path + ".tmp", mode="w+", dtype=np.float32,
                                       shape=(n,))
    else:
        X = np.empty((n, n_features), dtype=np.uint8)
        cp = np.empty(n, dtype=np.float32)

    with open(path, "r", encoding="ascii") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            bits, score = line.rstrip("\n").split("\t")
            X[i] = np.frombuffer(bits.encode("ascii"), dtype=np.uint8) - 48
            cp[i] = float(score)
            if (i + 1) % 500000 == 0:
                print(f"loaded {i + 1}/{n} from {os.path.basename(path)}", flush=True)

    if use_cache:
        X.flush(); cp.flush()
        del X, cp
        os.replace(x_path + ".tmp", x_path)
        os.replace(cp_path + ".tmp", cp_path)
        return np.load(x_path, mmap_mode="r"), np.load(cp_path, mmap_mode="r")
    return X, cp


def cp_to_target(cp, scale=100.0, clip=2000.0):
    """
    Centipawns -> real-valued target in pawns (cp / 100), clipped to +/-clip cp.
    Clipping matters: mate scores can be 100000 and would dominate the loss.
    Invert at inference with: cp = output * scale
    """
    return np.clip(np.asarray(cp, dtype=np.float32), -clip, clip) / scale


def as_planes(X):
    """(N, features) -> (N, planes, 8, 8), for a CNN or to inspect planes."""
    return X.reshape(-1, X.shape[1] // 64, 8, 8)


# ------------------------------------------------------- duplicate detection
def _make_R(n):
    # same seed and size as before for 1280 features, so existing cached hashes stay valid
    return np.random.default_rng(12345).integers(1, 2**63, size=n, dtype=np.uint64) | np.uint64(1)


def row_hashes(X, path, chunk=100_000):
    """64-bit hash of every row (exact duplicates share a hash). Cached on disk."""
    hp = path + ".hash.npy"
    if os.path.exists(hp) and os.path.getmtime(hp) >= os.path.getmtime(path + ".X.npy"):
        return np.load(hp)
    R = _make_R(X.shape[1] // 8)
    out = np.empty(len(X), dtype=np.uint64)
    for i in range(0, len(X), chunk):
        packed = np.packbits(np.asarray(X[i:i + chunk]), axis=1)          # (c, features/8) uint8
        out[i:i + chunk] = (packed.astype(np.uint64) * R).sum(axis=1, dtype=np.uint64)
    np.save(hp, out)
    return out


# ---------------------------------------------------------------------- training
def main(old_path, new_path, stride_old=2, stride_new=1, rows_per_epoch=2_000_000,
         hidden=(512, 128, 64), alpha=1e-3, batch_size=1024, lr=3e-4,
         max_epochs=60, patience=5, prefix="weights_13200K_28", chunk=200_000,
         n_blocks=10, block_len=2000):
    nf_old, nf_new = check_encoded(old_path), check_encoded(new_path)
    if nf_old != nf_new:
        raise SystemExit(f"{old_path} has {nf_old} features but {new_path} has {nf_new}: they use "
                         f"different plane layouts. Re-encode both from the labeled FENs with the "
                         f"same --planes setting.")
    X_old, cp_old = load_dataset(old_path)
    X_new, cp_new = load_dataset(new_path)
    y_old, y_new = cp_to_target(cp_old, clip=1000), cp_to_target(cp_new, clip=1000)
    n_old, n_new = len(X_old), len(X_new)

    # --- validation: the SAME blocks as before (from the old set only) -----------
    starts = list(range(block_len // 2, n_old - int(1.5 * block_len), n_old // n_blocks))
    va_idx = np.concatenate([np.arange(s, s + block_len) for s in starts])
    excl = np.zeros(n_old, bool)
    pad = block_len // 20
    for s in starts:
        excl[max(0, s - pad): s + block_len + pad] = True
    X_va, y_va = X_old[va_idx].astype("float32"), y_old[va_idx]
    white_va = X_va[:, STM_COL] == 1

    # --- training pool: old (minus validation) + all new, exact duplicates removed
    h_old, h_new = row_hashes(X_old, old_path), row_hashes(X_new, new_path)
    cand_old = np.flatnonzero(~excl)
    g_idx = np.concatenate([cand_old, n_old + np.arange(n_new)])
    g_h = np.concatenate([h_old[cand_old], h_new])
    n_before = len(g_idx)
    keep = ~np.isin(g_h, h_old[va_idx])                       # no position seen in validation
    n_val_overlap = int((~keep).sum())
    g_idx, g_h = g_idx[keep], g_h[keep]
    _, first = np.unique(g_h, return_index=True)              # keep first copy (old rows win)
    g_idx = g_idx[np.sort(first)]
    pool = np.concatenate([g_idx[g_idx < n_old][::stride_old],
                           g_idx[g_idx >= n_old][::stride_new]])

    print(f"old rows {n_old} | new rows {n_new} | validation rows {len(va_idx)}")
    print(f"removed {n_val_overlap} rows that also appear in validation and "
          f"{len(g_h) - len(g_idx)} exact duplicates; pool = {len(pool)} rows "
          f"({int((pool < n_old).sum())} old, {int((pool >= n_old).sum())} new)")

    def fetch(idx):
        idx = np.sort(idx)
        o, nw = idx[idx < n_old], idx[idx >= n_old] - n_old
        X = np.concatenate([X_old[o], X_new[nw]]).astype(np.float32)
        y = np.concatenate([y_old[o], y_new[nw]])
        return X, y

    rng = np.random.default_rng(0)
    ev_idx = rng.choice(pool, min(20_000, len(pool)), replace=False)
    X_ev, y_ev = fetch(ev_idx)                                # fixed sample for the train MSE print
    print(f"White to move in training sample: {(X_ev[:, STM_COL] == 1).mean():.1%}")
    base = mean_squared_error(y_va, np.full_like(y_va, y_ev.mean()))
    print(f"baseline val MSE: {base:.3f}", flush=True)

    per_epoch = min(rows_per_epoch or len(pool), len(pool))
    perm, pos = rng.permutation(pool), 0
    model = MLPRegressor(hidden_layer_sizes=hidden, alpha=alpha,
                         batch_size=batch_size, learning_rate_init=lr)
    best, stale, seen = float("inf"), 0, 0
    for epoch in range(max_epochs):
        if pos + per_epoch > len(perm):
            perm, pos = rng.permutation(pool), 0
        rows, pos = perm[pos:pos + per_epoch], pos + per_epoch
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for c in range(0, len(rows), chunk):
                Xb, yb = fetch(rows[c:c + chunk])
                model.partial_fit(Xb, yb)
        seen += len(rows)
        p = model.predict(X_va)
        tr = mean_squared_error(y_ev, model.predict(X_ev))
        va = mean_squared_error(y_va, p)
        w = mean_squared_error(y_va[white_va], p[white_va])
        b = mean_squared_error(y_va[~white_va], p[~white_va])
        print(f"epoch {epoch} ({seen / 1e6:.1f}M rows seen): train {tr:.3f}  val {va:.3f}  "
              f"[White-to-move {w:.3f}, Black-to-move {b:.3f}]  (baseline {base:.3f})",
              flush=True)
        if va < best:
            best, stale = va, 0
            save_weights_tsv(model, prefix)
        else:
            stale += 1
            if stale >= patience:
                print(f"No improvement for {patience} epochs; best val {best:.3f} saved to '{prefix}_*'")
                break
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("old_path", help="encoded old set (also supplies the validation blocks)")
    ap.add_argument("new_path", help="encoded new set (training only)")
    ap.add_argument("--stride-old", type=int, default=2)
    ap.add_argument("--stride-new", type=int, default=1)
    ap.add_argument("--rows-per-epoch", type=int, default=2_000_000,
                    help="rows per 'epoch'; 0 = whole pool each epoch")
    ap.add_argument("--prefix", default="weights_8M")
    ap.add_argument("--hidden", type=int, nargs="+", default=[512, 128, 64],
                    help="hidden layer sizes, e.g. --hidden 1024 256 64")
    a = ap.parse_args()
    main(a.old_path, a.new_path, stride_old=a.stride_old, stride_new=a.stride_new,
         rows_per_epoch=a.rows_per_epoch or None, prefix=a.prefix,
         hidden=tuple(a.hidden))