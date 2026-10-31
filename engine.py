
import os

import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor

import chess
import os
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor

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
            raise ValueError(
                f"Layer {i} takes {Ws[i].shape[0]} inputs but layer {i - 1} outputs "
                f"{Ws[i - 1].shape[1]}. Stale files from an older net? "
                f"Delete all '{prefix}_*.tsv' files and re-save.")
 
    n_in = Ws[0].shape[0]
    hidden = tuple(W.shape[1] for W in Ws[:-1])
    mlp_kwargs.setdefault("warm_start", True)
    model = MLPRegressor(hidden_layer_sizes=hidden, activation="relu", **mlp_kwargs)
 
    # Run a throwaway 1-epoch fit so sklearn creates all its fitted attributes,
    # then overwrite the weights with the loaded ones.
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



from fens_to_arrays import encode_for_model

def get_model_eval(model, fen):
    #Return the model's prediction of the position
    return model.predict(encode_for_model(model, fen))[0]

def is_eval_better(a,b,colour):
    #Helper function for the get best move function
    if colour == "white":
        if a > b:
            return True
    elif colour == "black":
        if a < b:
            return True

    return False


def get_model_best_move(model,fen,depth):
    #1 or 2 ply search function 
    board = chess.Board(fen)
    legal_moves_generator = board.legal_moves


    if board.turn == chess.WHITE:
        current_best_eval = -1000000
        best_reply_eval_ref = 1000000
        player = "white"
        opponent = "black"
    else:
        current_best_eval = 10000000
        best_reply_eval_ref = -10000000
        player = "black"
        opponent = "white"

    best_move = None
    for move in legal_moves_generator:
        new_board = board.copy()
        new_board.push(move)
        depth_2_moves = new_board.legal_moves

        if depth == 2:
            best_reply = None
            best_reply_eval = best_reply_eval_ref

            for reply in depth_2_moves:
                depth_2_board = new_board.copy()
                depth_2_board.push(reply)
                reply_fen = depth_2_board.fen()
                reply_eval = get_model_eval(model,reply_fen)
                if is_eval_better(reply_eval,best_reply_eval,opponent):
                    best_reply = reply
                    best_reply_eval = reply_eval

            if is_eval_better(best_reply_eval,current_best_eval,player):
                best_move_fen = new_board.fen()
                best_move = move
                current_best_eval = best_reply_eval
        elif depth == 1:
            eval = get_model_eval(model,new_board.fen())
            if is_eval_better(eval,current_best_eval,player):
                best_move_fen = new_board.fen()
                best_move = move
                current_best_eval = eval

    return best_move_fen, best_move

if __name__ == "__main__":
    #Example script showing usage of the engine.
    from pathlib import Path

    # Get the absolute path of the folder containing this script
    current_dir = Path(__file__).resolve().parent
    model = load_weights_tsv(str(current_dir) + "/TinyZero 4.0 Weights/weights_13M")

    example_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"

    print(get_model_eval(model,example_fen))

    print(get_model_best_move(model,example_fen,depth=2))

