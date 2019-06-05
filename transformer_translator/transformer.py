from keras.models import Model
import keras.backend as K
#import tensorflow.keras.backend as K
from keras.layers import *
#from tensorflow.keras.layers import *
#from tensorflow.keras import Model
import tensorflow as tf
import numpy as np

class Attention(Layer):

    def __init__(self, dim, embedding_dim, mask, heads, self_attention, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.dim = dim
        self.embedding_dim = embedding_dim
        self.mask = mask
        self.heads = heads
        self.self_attention = self_attention

    def build(self, input_shape):
        self.query = self.add_weight(name='query',
                                     shape=(self.embedding_dim, self.dim*self.heads),
                                     initializer='uniform',
                                     trainable=True)
        self.key = self.add_weight(name='key',
                                   shape=(self.embedding_dim, self.dim*self.heads),
                                   initializer='uniform',
                                   trainable=True)
        self.value = self.add_weight(name='value',
                                     shape=(self.embedding_dim, self.dim*self.heads),
                                     initializer='uniform',
                                     trainable=True)
        self.reduction = self.add_weight(name='reduction',
                                         shape=(self.dim*self.heads, self.embedding_dim),
                                         initializer='uniform',
                                         trainable=True)
        super(Attention, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        if self.self_attention:
            query = tf.tensordot(x, self.query, axes=[[2], [0]])
            key = tf.tensordot(x, self.key, axes=[[2], [0]])
            value = tf.tensordot(x, self.value, axes=[[2], [0]])
        else:
            query = tf.tensordot(x[0], self.query, axes=[[2], [0]])
            key = tf.tensordot(x[1], self.key, axes=[[2], [0]])
            value = tf.tensordot(x[1], self.value, axes=[[2], [0]])

        all_scores = []
        for i in range(self.heads):
            head_query = query[:, :, i*self.dim:(i+1)*self.dim]
            head_key = key[:, :, i*self.dim:(i+1)*self.dim]
            head_value = value[:, :, i*self.dim:(i+1)*self.dim]

            relevancy = 1/8 * K.batch_dot(head_query, K.permute_dimensions(head_key, [0, 2, 1]))
            if self.mask:
                masking = tf.cast((1 - tf.matrix_band_part(tf.ones_like(relevancy, dtype="float32"), -1, 0)), K.floatx())# not working with float16? Upper triangle is != 0
                masked = tf.math.add(relevancy, -50*masking)
            else:
                masked = relevancy
            softmax = K.softmax(masked)
            scores = K.batch_dot(softmax, head_value)
            all_scores.append(scores)
        all_scores = tf.concat(all_scores, axis=2)
        reduced = tf.tensordot(all_scores, self.reduction, axes=[[2], [0]])
        return reduced

    def compute_output_shape(self, input_shape):
        if self.self_attention:
            return input_shape[0], input_shape[1], self.embedding_dim
        else:
            return input_shape[0][0], input_shape[0][1], self.embedding_dim

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):
        config = super(Attention, self).get_config()
        config["dim"] = self.dim
        config["mask"] = self.mask
        config["heads"] = self.heads
        config["self_attention"] = self.self_attention
        config["embedding_dim"] = self.embedding_dim
        return config


class PositionalEncoding(Layer):

    def __init__(self, n, embedding_dim, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.n = n
        self.embedding_dim = embedding_dim

    def build(self, input_shape):
        self.embedding = K.constant(value=positional_encoding(self.n, self.embedding_dim))
        super(PositionalEncoding, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        input_shape = K.shape(x)
        return x + self.embedding[:, 0:input_shape[1]]

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config["n"] = self.n
        config["embedding_dim"] = self.embedding_dim
        return config


def positional_encoding(n, embedding_dim):
    position_enc = np.array([[
        [pos / np.power(10000, 2 * (j // 2) / embedding_dim) for j in range(embedding_dim)]
        for pos in range(n)]], dtype=K.floatx())
    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    return position_enc

def encoder_block(inputs, heads, embedding_dim):
    attention = Attention(64, embedding_dim, heads=heads, mask=False, self_attention=True)(inputs)
    attention = Dropout(0.1)(attention)
    attention = Add()([attention, inputs])
    out = TimeDistributed(BatchNormalization())(attention)

    out = TimeDistributed(Dense(embedding_dim, activation="relu"))(out)
    out = Dropout(0.1)(out)
    out = Add()([inputs, out])
    out = TimeDistributed(BatchNormalization())(out)
    return out

def decoder_block(encoder_inputs, inputs, heads, embedding_dim):
    attention = Attention(64, embedding_dim, heads=heads, mask=True, self_attention=True)(inputs)
    attention = Dropout(0.1)(attention)
    attention = Add()([attention, inputs])
    out = TimeDistributed(BatchNormalization())(attention)

    attention = Attention(64, embedding_dim, heads=heads, mask=False, self_attention=False)([out, encoder_inputs])
    attention = Dropout(0.1)(attention)
    attention = Add()([attention, out])
    out = TimeDistributed(BatchNormalization())(attention)

    out = TimeDistributed(Dense(embedding_dim, activation="relu"))(out)
    out = Dropout(0.1)(out)
    out = Add()([inputs, out])
    out = TimeDistributed(BatchNormalization())(out)
    return out

def transformer(max_len, vocab_len, mask_id=None, embedding=None, single_out=True,
                blocks=6, heads=8,  embedding_dim=300):
    input_source = Input(shape=(None,))
    input_target = Input(shape=(None,))
    if mask_id is not None:
        masked_in_source = Masking(mask_value=mask_id)(input_source)
        masked_in_target = Masking(mask_value=mask_id)(input_target)
    else:
        masked_in_source = input_source
        masked_in_target = input_target
    if embedding is None:
        embedding_source = Embedding(input_dim=vocab_len, output_dim=embedding_dim)(masked_in_source)
        embedding_target = Embedding(input_dim=vocab_len, output_dim=embedding_dim)(masked_in_target)
    else:
        embedding_source = Embedding(input_dim=vocab_len, output_dim=embedding_dim, weights=[embedding])(masked_in_source)
        embedding_target = Embedding(input_dim=vocab_len, output_dim=embedding_dim, weights=[embedding])(masked_in_target)
    embedding_source = PositionalEncoding(max_len, embedding_dim)(embedding_source)
    embedding_target = PositionalEncoding(max_len, embedding_dim)(embedding_target)
    encoder = Dropout(0.1)(embedding_source)
    for i in range(blocks):
        encoder = encoder_block(encoder, heads, embedding_dim)
    decoder = Dropout(0.1)(embedding_target)
    for i in range(blocks):
        decoder = decoder_block(encoder, decoder, heads, embedding_dim)
    out = Dropout(0.1)(decoder)
    if single_out:
        out = Lambda(lambda x: x[:, -1])(out)
        out = Dense(vocab_len, activation="softmax")(out)
    else:
        out = TimeDistributed(Dense(vocab_len, activation="softmax"))(out)
    model = Model(inputs=[input_source, input_target], outputs=out)
    return model