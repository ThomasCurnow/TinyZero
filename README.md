# TinyZero 4.0 MLP Chess Engine
TinyZero is an open source 1M parameter chess Multi Layer Perceptron, trained on over 13M positions that are labelled with stockfish 19 at depth 10. To run it, 
you first have to load the weights using the **load_weights_tsv** function, passing in the path to the prefix of the 8 weights files. You can then call **get_model_best_move**, 
passing it a FEN string of a chess position and the model object, which will return an updated FEN with the model's move played and also a chess.Move object. You can play
TinyZero by creating a free lichess.org account and challenging the bot **TinyZero**.
