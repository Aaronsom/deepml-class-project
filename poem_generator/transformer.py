from poem_generator.global_constants import EMBEDDING_DIMENSION
from keras.models import Model
import keras.backend as K
from keras.layers import *
import tensorflow as tf
import numpy as np

class Attention(Layer):

    def __init__(self, dim, mask, self_attention, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.dim = dim
        self.mask = mask
        self.self_attention = self_attention

    def build(self, input_shape):
        if self.self_attention:
            shape = input_shape[2]
        else:
            shape = input_shape[0][2]
        self.query = self.add_weight(name='query',
                                      shape=(shape, self.dim),
                                      initializer='uniform',
                                      trainable=True)
        self.key = self.add_weight(name='key',
                                      shape=(shape, self.dim),
                                      initializer='uniform',
                                      trainable=True)
        self.value = self.add_weight(name='value',
                                      shape=(shape, self.dim),
                                      initializer='uniform',
                                      trainable=True)
        super(Attention, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        if self.self_attention:
            query = tf.tensordot(x, self.query, axes=[[2], [0]])
            key = tf.tensordot(x, self.key, axes=[[2], [0]])
            value = tf.tensordot(x, self.value, axes=[[2], [0]])
        else:
            query = tf.tensordot(x[0], self.query, axes=[[2], [0]])
            key = tf.tensordot(x[1], self.key, axes=[[2], [0]])
            value = tf.tensordot(x[1], self.value, axes=[[2], [0]])
        relevancy = 1/8 * K.batch_dot(query, K.permute_dimensions(key, [0, 2, 1]))
        if self.mask:
            masking = tf.cast((1 - tf.matrix_band_part(tf.ones_like(relevancy, dtype="float32"), -1, 0)), "float16")# not working with float16? Upper triangle is != 0
            masked = tf.math.add(relevancy, -20*masking)
        else:
            masked = relevancy
        softmax = K.softmax(masked)
        scores = K.batch_dot(softmax, value)
        return scores

    def compute_output_shape(self, input_shape):
        if self.self_attention:
            return input_shape[0], input_shape[1], self.dim
        else:
            return input_shape[0][0], input_shape[0][1], self.dim


    def get_config(self):
        config = super(Attention, self).get_config()
        config["dim"] = self.dim
        config["mask"] = self.mask
        config["self_attention"] = self.self_attention
        return config


class PositionalEncoding(Layer):

    def __init__(self, n, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.n = n

    def build(self, input_shape):
        self.embedding = K.constant(value=positional_encoding(self.n))
        super(PositionalEncoding, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        input_shape = K.shape(x)
        return x + self.embedding[:, 0:input_shape[1]]

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config["n"] = self.n
        return config


def positional_encoding(n):
    position_enc = np.array([[
        [pos / np.power(10000, 2 * (j // 2) / EMBEDDING_DIMENSION) for j in range(EMBEDDING_DIMENSION)]
        for pos in range(n)]], dtype="float16")
    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    return position_enc

def encoder_block(inputs):
    heads = []
    for i in range(8):
        attention = Attention(64, mask=False, self_attention=True)(inputs)
        heads.append(attention)
    attention = Concatenate()(heads)
    attention = TimeDistributed(Dense(EMBEDDING_DIMENSION))(attention)
    attention = Dropout(0.1)(attention)
    attention = Add()([attention, inputs])
    out = TimeDistributed(BatchNormalization())(attention)

    out = TimeDistributed(Dense(EMBEDDING_DIMENSION, activation="relu"))(out)
    out = Dropout(0.1)(out)
    out = Add()([inputs, out])
    out = TimeDistributed(BatchNormalization())(out)
    return out

def decoder_block(encoder_inputs, inputs):
    heads = []
    for i in range(8):
        attention = Attention(64, mask=True, self_attention=True)(inputs)
        heads.append(attention)
    attention = Concatenate()(heads)
    attention = TimeDistributed(Dense(EMBEDDING_DIMENSION))(attention)
    attention = Dropout(0.1)(attention)
    attention = Add()([attention, inputs])
    out = TimeDistributed(BatchNormalization())(attention)

    heads = []
    for i in range(8):
        attention = Attention(64, mask=False, self_attention=False)([out, encoder_inputs])
        heads.append(attention)
    attention = Concatenate()(heads)
    attention = TimeDistributed(Dense(EMBEDDING_DIMENSION))(attention)
    attention = Dropout(0.1)(attention)
    attention = Add()([attention, out])
    out = TimeDistributed(BatchNormalization())(attention)

    out = TimeDistributed(Dense(EMBEDDING_DIMENSION, activation="relu"))(out)
    out = Dropout(0.1)(out)
    out = Add()([inputs, out])
    out = TimeDistributed(BatchNormalization())(out)
    return out

def transformer(n, embedding, vocab_len):
    inputs = Input(shape=(None, ))
    embedding = Embedding(input_dim=vocab_len, output_dim=EMBEDDING_DIMENSION, weights=[embedding], trainable=False)(inputs)
    embedding = PositionalEncoding(n)(embedding)
    encoder = Dropout(0.1)(embedding)
    for i in range(4):
        encoder = encoder_block(encoder)
    decoder = Dropout(0.1)(embedding)
    for i in range(4):
        decoder = decoder_block(encoder, decoder)
    out = Dropout(0.1)(encoder)
    out = TimeDistributed(Dense(vocab_len, activation="softmax"))(out)
    model = Model(inputs=inputs, outputs=out)
    return model