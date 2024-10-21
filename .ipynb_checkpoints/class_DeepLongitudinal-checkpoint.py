import numpy as np
import tensorflow as tf
import random

from tensorflow.keras.layers import Dense as FC_Net
import utils_network as utils

_EPSILON = 1e-08

##### USER-DEFINED FUNCTIONS
def log(x):
    return tf.math.log(x + _EPSILON)

def div(x, y):
    return tf.divide(x, (y + _EPSILON))

def get_seq_length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), axis=2))
    tmp_length = tf.reduce_sum(used, axis=1)
    tmp_length = tf.cast(tmp_length, tf.int32)
    return tmp_length


class Model_Longitudinal_Attention:
    def __init__(self, name, input_dims, network_settings):
        self.name               = name

        # INPUT DIMENSIONS
        self.x_dim              = input_dims['x_dim']
        self.x_dim_cont         = input_dims['x_dim_cont']
        self.x_dim_bin          = input_dims['x_dim_bin']
        self.num_Event          = input_dims['num_Event']
        self.num_Category       = input_dims['num_Category']
        self.max_length         = input_dims['max_length']

        # NETWORK HYPER-PARAMETERS
        self.h_dim1             = network_settings['h_dim_RNN']
        self.h_dim2             = network_settings['h_dim_FC']
        self.num_layers_RNN     = network_settings['num_layers_RNN']
        self.num_layers_ATT     = network_settings['num_layers_ATT']
        self.num_layers_CS      = network_settings['num_layers_CS']
        self.RNN_type           = network_settings['RNN_type']
        self.FC_active_fn       = network_settings['FC_active_fn']
        self.RNN_active_fn      = network_settings['RNN_active_fn']
        self.initial_W          = network_settings['initial_W']
        
        # Regularizers in TF2.x are in keras layers, you may need to define them explicitly
        self.reg_W              = tf.keras.regularizers.L1(network_settings['reg_W'])
        self.reg_W_out          = tf.keras.regularizers.L1(network_settings['reg_W_out'])

        self._build_net()


    def _build_net(self):
        with tf.compat.v1.variable_scope(self.name):
            #### INPUT DECLARATION (replace placeholders)
            self.mb_size     = tf.keras.Input(shape=[], dtype=tf.int32, name='batch_size')
            self.lr_rate     = tf.keras.Input(shape=[], dtype=tf.float32)
            self.keep_prob   = tf.keras.Input(shape=[], dtype=tf.float32)
            self.a           = tf.keras.Input(shape=[], dtype=tf.float32)
            self.b           = tf.keras.Input(shape=[], dtype=tf.float32)
            self.c           = tf.keras.Input(shape=[], dtype=tf.float32)

            self.x           = tf.keras.Input(shape=[self.max_length, self.x_dim], dtype=tf.float32)
            self.x_mi        = tf.keras.Input(shape=[self.max_length, self.x_dim], dtype=tf.float32)
            self.k           = tf.keras.Input(shape=[1], dtype=tf.float32)
            self.t           = tf.keras.Input(shape=[1], dtype=tf.float32)

            self.fc_mask1    = tf.keras.Input(shape=[self.num_Event, self.num_Category], dtype=tf.float32)
            self.fc_mask2    = tf.keras.Input(shape=[self.num_Event, self.num_Category], dtype=tf.float32)
            self.fc_mask3    = tf.keras.Input(shape=[self.num_Category], dtype=tf.float32)

            seq_length     = get_seq_length(self.x)
            tmp_range      = tf.expand_dims(tf.range(0, self.max_length, 1), axis=0)

            self.rnn_mask1 = tf.cast(tf.less_equal(tmp_range, tf.expand_dims(seq_length - 1, axis=1)), tf.float32)            
            self.rnn_mask2 = tf.cast(tf.equal(tmp_range, tf.expand_dims(seq_length - 1, axis=1)), tf.float32)

            ### Replace raw RNN with TensorFlow 2.x RNN (LSTM, GRU, etc.)
            cell = utils.create_rnn_cell(self.h_dim1, self.num_layers_RNN, self.keep_prob, self.RNN_type, self.RNN_active_fn)

            # RNN Layer in TF2.x
            rnn_layer = tf.keras.layers.RNN(cell, return_sequences=True, return_state=True)

            # Define input to the RNN
            rnn_outputs, self.rnn_final_state = rnn_layer(self.x)

            # Transpose RNN outputs to match the expected shape (batch_time_transpose in TF1.x)
            rnn_outputs = tf.transpose(rnn_outputs, perm=[1, 0, 2])

            # Further calculations for attention and fully connected layers follow the same logic
            self.context_vec = tf.reduce_sum(tf.tile(tf.reshape(self.att_weight, [-1, self.max_length-1, 1]), 
                                                     [1, 1, self.num_layers_RNN*self.h_dim1]) * rnn_outputs, axis=1)

            # Create the final dense (fully connected) layers
            self.z_mean      = FC_Net(rnn_outputs, self.x_dim)
            self.z_std       = tf.exp(FC_Net(rnn_outputs, self.x_dim))

            epsilon          = tf.random.normal([self.mb_size, self.max_length-1, self.x_dim], mean=0.0, stddev=1.0)
            self.z           = self.z_mean + self.z_std * epsilon

            inputs = tf.concat([self.context_vec, self.x], axis=1)
            h = FC_Net(inputs, self.h_dim2, activation=self.FC_active_fn)
            h = tf.nn.dropout(h, rate=1 - self.keep_prob)

            out = []
            for _ in range(self.num_Event):
                cs_out = utils.create_FCNet(h, self.num_layers_CS, self.h_dim2, self.FC_active_fn, self.h_dim2, self.FC_active_fn, self.initial_W, self.reg_W, self.keep_prob)
                out.append(cs_out)
            out = tf.stack(out, axis=1)

            self.out = FC_Net(out, self.num_Event * self.num_Category, activation=tf.nn.softmax)

            ##### LOSSES (adjusted for TensorFlow 2.x syntax)
            self.loss_Log_Likelihood()
            self.loss_Ranking()
            self.loss_RNN_Prediction()

            self.LOSS_TOTAL     = self.a*self.LOSS_1 + self.b*self.LOSS_2 + self.c*self.LOSS_3 + tf.add_n(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES))
            self.solver         = tf.keras.optimizers.Adam(learning_rate=self.lr_rate).minimize(self.LOSS_TOTAL)


    ### LOSS-FUNCTION 1 -- Log-likelihood loss
    def loss_Log_Likelihood(self):
        sigma3 = tf.constant(1.0, dtype=tf.float32)
        I_1 = tf.sign(self.k)

        denom = 1 - tf.reduce_sum(self.fc_mask1 * self.out, axis=2)
        denom = tf.clip_by_value(denom, _EPSILON, 1 - _EPSILON)

        tmp1 = tf.reduce_sum(self.fc_mask2 * self.out, axis=2)
        tmp1 = I_1 * log(div(tmp1, denom))

        tmp2 = tf.reduce_sum(self.fc_mask2 * self.out, axis=2)
        tmp2 = (1. - I_1) * log(div(tmp2, denom))

        self.LOSS_1 = - tf.reduce_mean(tmp1 + sigma3 * tmp2)


    ### LOSS-FUNCTION 2 -- Ranking loss
    def loss_Ranking(self):
        sigma1 = tf.constant(0.1, dtype=tf.float32)

        eta = []
        for e in range(self.num_Event):
            one_vector = tf.ones_like(self.t, dtype=tf.float32)
            I_2 = tf.cast(tf.equal(self.k, e+1), dtype=tf.float32)
            I_2 = tf.linalg.diag(tf.squeeze(I_2))
            tmp_e = tf.reshape(self.out[:, e, :], [-1, self.num_Category])

            R = tf.matmul(tmp_e, tf.transpose(self.fc_mask3))
            diag_R = tf.reshape(tf.linalg.diag_part(R), [-1, 1])
            R
