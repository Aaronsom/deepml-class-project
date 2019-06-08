from keras.callbacks import ModelCheckpoint, CSVLogger
import keras.optimizers as optimizer
from keras.backend import set_floatx, set_epsilon
from keras.models import load_model
from transformer_translator.data_generator import DataGenerator
from transformer_translator.transformer import transformer, PositionalEncoding, Attention
from transformer_translator.ted_data_preprocessor import get_data, get_encoding_info
from transformer_translator.predictor import TranslationCallback
from transformer_translator.noam_schedule import NoamSchedule
import numpy as np
import keras.backend as K
import tensorflow as tf
import os
from datetime import datetime


def masked_sparse_categorical_accuracy(mask_id):
    def sparse_categorical_accuracy(y_true, y_pred):
        y_true = K.max(y_true, axis=-1)
        y_pred = K.cast(K.argmax(y_pred, axis=-1), K.floatx())
        mask = K.not_equal(y_true, mask_id)
        masked_true = tf.boolean_mask(y_true, mask)
        masked_pred = tf.boolean_mask(y_pred, mask)
        accuracy = K.mean(K.cast(K.equal(masked_true, masked_pred), K.floatx()))
        return accuracy
        #return K.cast(K.equal(K.max(y_true, axis=-1),
        #                      K.cast(K.argmax(y_pred, axis=-1), K.floatx())),
        #              K.floatx())
    return sparse_categorical_accuracy

if __name__ == "__main__":
    #set_floatx("float16")
    #set_epsilon(1e-04)

    languages = ["en", "de"]
    max_len = 64
    epochs = 300
    batch_size = 32
    embedding_dim = 300
    load = False
    path = None
    start_epoch = 0


    training, dev, test = get_data(languages, data_folder="data")
    #training = [training[0][:int(0.5*len(training[0]))], training[1][:int(0.5*len(training[0]))]] #train with 1 language direction
    #dev = [dev[0][:int(0.5*len(dev[0]))], dev[1][:int(0.5*len(dev[0]))]]
    vocab_len, mask_id = get_encoding_info(languages, data_folder="data")

    start_steps = start_epoch * len(training[0])

    if load:
        model = load_model(os.path.join(path, "model.hdf5"),
                           custom_objects={"PositionalEncoding": PositionalEncoding,
                                           "Attention": Attention, "sparse_categorical_accuracy": masked_sparse_categorical_accuracy(mask_id)})
    else:
        model = transformer(200, vocab_len, embedding_dim=embedding_dim, hidden_dim=1024,
                            blocks=6, heads=5, dropout=0.1, single_out=False, mask_id=mask_id)
        model.compile(optimizer=optimizer.Adam(beta_1=0.9, beta_2=0.98, epsilon=1e-9),
                  loss="sparse_categorical_crossentropy", metrics=[masked_sparse_categorical_accuracy(mask_id)])

    model.summary()

    if not load:
        path = os.path.join("out", datetime.now().strftime('%Y-%m-%d_%H-%M'))
        os.makedirs(path, exist_ok=True)

    training_generator = DataGenerator(training, mask_id=mask_id, max_len=max_len, batch_size=batch_size)
    validation_generator = DataGenerator(dev, mask_id=mask_id, max_len=max_len, batch_size=batch_size)
    callbacks = [ModelCheckpoint(os.path.join(path, "best-model.hdf5"), save_best_only=True),
                 ModelCheckpoint(os.path.join(path, "model.hdf5"), save_best_only=False),
                 CSVLogger(os.path.join(path, "log.csv"), append=True),
                 TranslationCallback(languages, max_len=max_len),
                 NoamSchedule(warmup_steps=8000, learning_rate=0.1, start_steps=start_steps)]
    model.fit_generator(
        training_generator, initial_epoch=start_epoch, epochs=epochs, callbacks=callbacks, validation_data=validation_generator, workers=1)
