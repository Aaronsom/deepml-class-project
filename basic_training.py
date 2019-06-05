from keras.callbacks import ModelCheckpoint, CSVLogger
import keras.optimizers as optimizer
from keras.backend import set_floatx, set_epsilon

from transformer_translator.data_generator import DataGenerator
from transformer_translator.transformer import transformer
from transformer_translator.ted_data_preprocessor import get_data, get_encoding_info
from transformer_translator.predictor import TranslationCallback
import numpy as np
import keras.backend as K

def sparse_categorical_accuracy(y_true, y_pred):
    return K.cast(K.equal(K.max(y_true, axis=-1),
                          K.cast(K.argmax(y_pred, axis=-1), K.floatx())),
                  K.floatx())

if __name__ == "__main__":
    #set_floatx("float16")
    #set_epsilon(1e-04)

    languages = ["en", "de"]
    max_len = 50
    epochs = 300
    batch_size = 32
    embedding_dim = 300

    training, dev, test = get_data(languages, data_folder="data")
    vocab_len, mask_id = get_encoding_info(languages, data_folder="data")

    model = transformer(200, vocab_len, embedding_dim=embedding_dim, blocks=3, heads=5, single_out=False, mask_id=mask_id)
    model.summary()
    model.compile(optimizer=optimizer.Adam(),
                  loss="sparse_categorical_crossentropy", metrics=[sparse_categorical_accuracy])

    training_generator = DataGenerator(training, mask_id=mask_id, max_len=max_len, batch_size=batch_size)
    validation_generator = DataGenerator(dev, mask_id=mask_id, max_len=max_len, batch_size=batch_size)
    callbacks = [ModelCheckpoint("out/model.hdf5", save_best_only=True),
                 CSVLogger("out/log.csv", append=True),
                 TranslationCallback(languages, max_len=max_len)]
    model.fit_generator(
        training_generator, epochs=epochs, callbacks=callbacks, validation_data=validation_generator, workers=1)
