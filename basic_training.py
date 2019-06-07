from keras.callbacks import ModelCheckpoint, CSVLogger
import keras.optimizers as optimizer
from keras.backend import set_floatx, set_epsilon
from keras.models import load_model
from transformer_translator.data_generator import DataGenerator
from transformer_translator.transformer import transformer, PositionalEncoding, Attention
from transformer_translator.ted_data_preprocessor import get_data, get_encoding_info
from transformer_translator.predictor import TranslationCallback
import numpy as np
import keras.backend as K
import tensorflow as tf

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
    load = True
    start_epoch = 0

    training, dev, test = get_data(languages, data_folder="data")
    vocab_len, mask_id = get_encoding_info(languages, data_folder="data")

    if load:
        model = load_model("out/model.hdf5",
                           custom_objects={"PositionalEncoding": PositionalEncoding,
                                           "Attention": Attention, "sparse_categorical_accuracy": masked_sparse_categorical_accuracy(mask_id)})
    else:
        model = transformer(200, vocab_len, embedding_dim=embedding_dim, blocks=6, heads=5, single_out=False, mask_id=mask_id)
        model.compile(optimizer=optimizer.Adam(),
                  loss="sparse_categorical_crossentropy", metrics=[masked_sparse_categorical_accuracy(mask_id)])

    model.summary()

    training_generator = DataGenerator(training, mask_id=mask_id, max_len=max_len, batch_size=batch_size)
    validation_generator = DataGenerator(dev, mask_id=mask_id, max_len=max_len, batch_size=batch_size)
    callbacks = [ModelCheckpoint("out/best-model.hdf5", save_best_only=True),
                 ModelCheckpoint("out/model.hdf5", save_best_only=False),
                 CSVLogger("out/log.csv", append=True),
                 TranslationCallback(languages, max_len=max_len)]
    model.fit_generator(
        training_generator, initial_epoch=start_epoch, epochs=epochs, callbacks=callbacks, validation_data=validation_generator, workers=1)
