from keras.callbacks import ModelCheckpoint, CSVLogger
import keras.optimizers as optimizer
from keras.backend import set_floatx, set_epsilon

from transformer_translator.data_generator import DataGenerator
from transformer_translator.transformer import transformer
from transformer_translator.ted_data_preprocessor import get_data, get_encoding_info

if __name__ == "__main__":
    #set_floatx("float16")
    #set_epsilon(1e-04)

    languages = ["en", "de"]
    max_len = 40
    epochs = 20
    batch_size = 32
    validation_split = 0.9
    embedding_dim = 300

    training, dev, test = get_data(languages, data_folder="data")

    vocab_len, mask_id = get_encoding_info(languages, data_folder="data")

    model = transformer(100, vocab_len, embedding_dim=embedding_dim, blocks=2, heads=5, single_out=False, mask_id=mask_id)
    model.summary()
    model.compile(optimizer=optimizer.Adam(),
                  loss="sparse_categorical_crossentropy", metrics=["categorical_accuracy"])

    training_generator = DataGenerator(training, mask_id=mask_id, max_len=max_len, batch_size=batch_size)
    validation_generator = DataGenerator(dev, mask_id=mask_id, max_len=max_len, batch_size=batch_size)
    callbacks = [ModelCheckpoint("out/model.hdf5", save_best_only=True),
                 CSVLogger("out/log.csv", append=True)]
    model.fit_generator(
        training_generator, epochs=epochs, callbacks=callbacks, validation_data=validation_generator, workers=1)
