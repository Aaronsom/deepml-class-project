from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, Dropout, CuDNNLSTM, CuDNNGRU, Conv1D, GlobalMaxPool1D, Bidirectional
from keras.callbacks import ModelCheckpoint, CSVLogger
import keras.optimizers as optimizer
from keras.backend import set_floatx, set_epsilon
from poem_generator.dataGenerator import TupleDataGenerator
import poem_generator.data_prepocessing as dp
import poem_generator.embedding as embedding_loader
from poem_generator.global_constants import TRAINING_DATA, EMBEDDING_DIMENSION, EMBEDDING_BINARY, MODELS_DICT
from poem_generator.transformer import transformer

def mlp(n, embedding, vocab_len):
    model = Sequential([
        Embedding(input_dim=vocab_len, output_dim=EMBEDDING_DIMENSION, input_length=n, weights=[embedding]),
        Flatten(),
        Dropout(0.2),
        Dense(n*600, activation="relu"),
        Dropout(0.2),
        Dense(n*300, activation="relu"),
        Dropout(0.2),
        Dense(n*100, activation="relu"),
        Dropout(0.2),
        Dense(vocab_len,activation="softmax"),
    ])
    return model

def lstm_rnn(n, embedding, vocab_len):
    model = Sequential([
        Embedding(input_dim=vocab_len, output_dim=EMBEDDING_DIMENSION, input_length=n, weights=[embedding]),
        CuDNNLSTM(4096, return_sequences=False),
        Dropout(0.2),
        Dense(1024, activation="relu"),
        Dropout(0.2),
        Dense(vocab_len, activation="softmax")
    ])
    return model

def gru_rnn(n, embedding, vocab_len):
    model = Sequential([
        Embedding(input_dim=vocab_len, output_dim=EMBEDDING_DIMENSION, input_length=n, weights=[embedding]),
        CuDNNGRU(512, return_sequences=False),
        Dropout(0.2),
        Dense(512, activation="relu"),
        Dropout(0.2),
        Dense(vocab_len, activation="softmax")
    ])
    return model

def cnn(n, embedding, vocab_len):
    model = Sequential([
        Embedding(input_dim=vocab_len, output_dim=EMBEDDING_DIMENSION, input_length=n, weights=[embedding]),
        Conv1D(50, 5, activation="relu"),
        GlobalMaxPool1D(),
        Dropout(0.2),
        Dense(1000, activation="relu"),
        Dropout(0.2),
        Dense(vocab_len, activation="softmax")
    ])
    return model

def bidirectional_lstm(n, embedding, vocab_len):
    model = Sequential([
        Embedding(input_dim=vocab_len, output_dim=EMBEDDING_DIMENSION, input_length=n, weights=[embedding]),
        Bidirectional(CuDNNLSTM(1024, return_sequences=False)),
        Dropout(0.2),
        Dense(512, activation="relu"),
        Dropout(0.2),
        Dense(vocab_len, activation="softmax")
    ])
    return model


if __name__ == "__main__":
    set_floatx("float16")
    set_epsilon(1e-04)
    ns = [20]
    epochs = 20
    batch_size = 512
    max_limit = 25000
    validation_split = 0.9

    poems = dp.tokenize_poems(TRAINING_DATA)
    words = set([token for poem in poems for token in poem])

    #Save embedding for generator
    embedding, dictionary = embedding_loader.get_embedding(words, binary=EMBEDDING_BINARY, limit=max_limit, save=True)

    model = transformer(100, embedding, len(dictionary))

    model.compile(optimizer=optimizer.Adam(),
                  loss="categorical_crossentropy", metrics=["accuracy"])

    generator = TupleDataGenerator(poems[:int(validation_split*len(poems))], ns, dictionary, 0, batch_size)
    validation_generator = TupleDataGenerator(poems[int(validation_split*len(poems)):], ns, dictionary, 0, batch_size)
    callbacks = [ModelCheckpoint(MODELS_DICT+"/4-model.hdf5", save_best_only=True),
                 CSVLogger(MODELS_DICT+"/log.csv", append=True, separator=';')]
    model.fit_generator(
        generator, epochs=epochs, callbacks=callbacks, validation_data=validation_generator, workers=4)
