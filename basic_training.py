from keras.callbacks import ModelCheckpoint, CSVLogger
import keras.optimizers as optimizer
from keras.backend import set_floatx, set_epsilon

from transformer_translator.data_generator import DataGenerator
from transformer_translator.transformer import transformer

if __name__ == "__main__":
    #set_floatx("float16")
    #set_epsilon(1e-04)
    epochs = 20
    batch_size = 512
    validation_split = 0.9

    vocab_len = 9999 #TODO

    model = transformer(100, vocab_len, blocks=2, heads=5, single_out=False)
    model.summary()
    model.compile(optimizer=optimizer.Adam(),
                  loss="categorical_crossentropy", metrics=["accuracy"])

    training_generator = DataGenerator()
    validation_generator = DataGenerator()
    callbacks = [ModelCheckpoint("out/model.hdf5", save_best_only=True),
                 CSVLogger("out/log.csv", append=True)]
    model.fit_generator(
        training_generator, epochs=epochs, callbacks=callbacks, validation_data=validation_generator, workers=1)