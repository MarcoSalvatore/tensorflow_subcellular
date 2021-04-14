import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tensorflow.contrib.layers import fully_connected


def network(n_hidden_rnn, n_filt, n_hidden, filt_size, i_drop, e_drop,
    n_input=20, n_steps=1000, n_type=6):

    n_units_lstm = n_hidden_rnn
    # Input variable
    x = tf.placeholder(tf.float32, [None, n_steps, n_input], name='Input_protein')
    # Label variable
    # Sequence length variable
    seq_len = tf.placeholder(tf.int32, [None], name='Protein_length')
    # Training variable
    is_training_pl = tf.placeholder(tf.bool, name='Training')
    # Type lable variable
    type_prot = tf.placeholder(tf.int32, [None], name='Protein_type')

    # Input dropout
    x_drop = tf.contrib.layers.dropout(x, noise_shape=[tf.shape(x)[0], n_steps, 1],
        keep_prob=i_drop, is_training=is_training_pl)

    conv0 = tf.layers.conv1d(x_drop, filters=20, kernel_size=21, strides=1, padding="same", activation=tf.nn.elu, name="conv0")
    conv1 = tf.layers.conv1d(x_drop, filters=20, kernel_size=15, strides=1, padding="same", activation=tf.nn.elu, name="conv1")
    conv2 = tf.layers.conv1d(x_drop, filters=20, kernel_size=9, strides=1, padding="same", activation=tf.nn.elu, name="conv2")
    conv3 = tf.layers.conv1d(x_drop, filters=20, kernel_size=5, strides=1, padding="same", activation=tf.nn.elu, name="conv3")
    conv4 = tf.layers.conv1d(x_drop, filters=20, kernel_size=3, strides=1, padding="same", activation=tf.nn.elu, name="conv4")
    conv5 = tf.layers.conv1d(x_drop, filters=20, kernel_size=1, strides=1, padding="same", activation=tf.nn.elu, name="conv5")
    conv_drop1  = tf.concat([conv0, conv1, conv2, conv3, conv4, conv5], axis=2, name="first_module")

    conv1 = tf.layers.conv1d(conv_drop1, filters=64,
                             kernel_size=21,
                             strides=1,
                             padding="same", activation=tf.nn.elu,
                             name="conv_1")
    conv_drop = tf.contrib.layers.dropout(conv1, keep_prob=e_drop, is_training=is_training_pl)

    # Forward direction cell
    lstm_fw_cell = rnn.BasicLSTMCell(n_units_lstm, forget_bias=1.0,
                                     name="forward")
    # Backward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(n_units_lstm, forget_bias=1.0,
                                     name="backward")
    outputs, states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,
                                                      lstm_bw_cell,
                                                      conv_drop,
                                                     # initial_state_fw=l_init_cell,
                                                     # initial_state_bw=l_init_cell,
                                                      sequence_length=seq_len,
                                                      dtype=tf.float32,
                                                      )

    last_state = tf.concat(outputs, 2, name="bidirectional_rnn1")

    final_outputs = last_state  # #conv_drop_1
    max_len = tf.shape(final_outputs)[1]

    # Attention matrix, filter size 1 convolution, equivalent to dense layer.
    hUa = tf.layers.conv1d(final_outputs, filters=128, kernel_size=1,
    	strides=1, padding="same", activation=tf.nn.tanh)

    # Align matrix, filter size 1 convolution, equivalent to dense layer.
    align = tf.layers.conv1d(hUa, filters=4, kernel_size=1,
    	strides=1, padding="same", activation=None)

    # Mask for padded positions
    masks = tf.expand_dims(tf.sequence_mask(seq_len, dtype=tf.float32, maxlen=1000), axis = 2)
    a_un_masked = align * masks - (1 - masks) * 100000

    # Calculate attention values
    alphas = tf.nn.softmax(a_un_masked, axis=1)

    # Weighted hidden states
    weighted_hidden = tf.expand_dims(final_outputs,2) * tf.expand_dims(alphas, 3)

    # Weighted sum
    weighted_sum = tf.reduce_sum(weighted_hidden, axis=1)

    # Final dense layer
    weighted_out = tf.contrib.layers.fully_connected(tf.contrib.layers.flatten(weighted_sum), 64, activation_fn=tf.nn.elu)


    type_pred_layer = tf.contrib.layers.fully_connected(weighted_out, n_type, activation_fn=None)


    type_pred = tf.nn.softmax(type_pred_layer, name='Protein_pred')

    # Calculate cross-entropy for protein type prediction
    loss_type = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=type_pred_layer, labels=type_prot)#*mask_layer_sub

    # Combined loss
    xent = loss_type
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.reduce_mean(xent, name="xent_final") + sum(regularization_losses)

	# Training operation
    train_adam = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

    arg_class = tf.cast(tf.argmax(type_pred, 1), tf.int32)
    correct_pred = tf.equal(tf.round(arg_class), type_prot)
    cf_m = tf.confusion_matrix(labels=type_prot, predictions=arg_class, num_classes=n_type) #, weights=mask_layer_sub)

    return x, seq_len, is_training_pl, type_prot, train_adam, loss, type_pred, arg_class, correct_pred, cf_m
