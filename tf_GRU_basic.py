import tensorflow as tf



cell = tf.nn.rnn_cell.GRUCell(cell_size) # Single GRU

md_cell = tf.nn.rnn_cell.MultiRNNCell([cell]*no_layers, state_is_tuple=False) # Multidimensional stack of GRUs for RNN

# TF function to unroll the RNN over input sequence (length specified within ip parameter X)
Hr, h = tf.nn.dynamic_rnn(md_cell, X, initial_state = Hin) 