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
import json


def masked_sparse_categorical_accuracy(mask_id):
    def sparse_categorical_accuracy(y_true, y_pred):
        y_true = K.max(y_true, axis=-1)
        y_pred = K.cast(K.argmax(y_pred, axis=-1), K.floatx())
        mask = K.not_equal(y_true, mask_id)
        masked_true = tf.boolean_mask(y_true, mask)
        masked_pred = tf.boolean_mask(y_pred, mask)
        accuracy = K.mean(K.cast(K.equal(masked_true, masked_pred), K.floatx()))
        return accuracy
        # return K.cast(K.equal(K.max(y_true, axis=-1),
        #                      K.cast(K.argmax(y_pred, axis=-1), K.floatx())),
        #              K.floatx())
    return sparse_categorical_accuracy


if __name__ == "__main__":
    # set_floatx("float16")
    # set_epsilon(1e-04)
    config = {
        "languages": ["en", "de"],
        "max_len": 64,
        "batch_size": 32,
        "blocks": 3,
        "heads": 4,
        "embedding_dim": 256,
        "hidden_dim": 1024,
        "dropout": 0.1,
        "adam_beta1": 0.9,
        "adam_beta2": 0.98,
        "warmup_steps": 8000,
        "learning_rate": 0.1,
        "mask_id": None,  # set later
        "vocab_len": None  # set later
    }

    if load:
        config = json.load(open(os.path.join(path, "config.json")))
        model = load_model(os.path.join(path, "model.hdf5"),
                           custom_objects={"PositionalEncoding": PositionalEncoding,
                                           "Attention": Attention, "sparse_categorical_accuracy": masked_sparse_categorical_accuracy(config["mask_id"])})
    else:
        config["vocab_len"], config["mask_id"] = get_encoding_info(config["languages"], data_folder="data")
        path = os.path.join("out", datetime.now().strftime('%Y-%m-%d_%H-%M'))
        os.makedirs(path, exist_ok=True)
        json.dump(config, open(os.path.join(path, "config.json"), "w"), indent=4)

        model = transformer(config["max_len"], config["vocab_len"], embedding_dim=config["embedding_dim"], hidden_dim=config["hidden_dim"],
                            blocks=config["blocks"], heads=config["heads"], dropout=config["dropout"], single_out=False, mask_id=config["mask_id"])
        model.compile(optimizer=optimizer.Adam(beta_1=config["adam_beta1"], beta_2=config["adam_beta2"], epsilon=1e-9),
                  loss="sparse_categorical_crossentropy", metrics=[masked_sparse_categorical_accuracy(config["mask_id"])])

    model.summary()

    training, dev, test = get_data(config["languages"], data_folder="data")
    # training = [training[0][:int(0.5*len(training[0]))], training[1][:int(0.5*len(training[0]))]] #train with 1 language direction
    # dev = [dev[0][:int(0.5*len(dev[0]))], dev[1][:int(0.5*len(dev[0]))]]

    start_steps = start_epoch * len(training[0])

    training_generator = DataGenerator(training, mask_id=config["mask_id"], max_len=config["max_len"], batch_size=config["batch_size"])
    validation_generator = DataGenerator(dev, mask_id=config["mask_id"], max_len=config["max_len"], batch_size=config["batch_size"])
    callbacks = [ModelCheckpoint(os.path.join(path, "best-model.hdf5"), save_best_only=True),
                 ModelCheckpoint(os.path.join(path, "model.hdf5"), save_best_only=False),
                 CSVLogger(os.path.join(path, "log.csv"), append=True),
                 TranslationCallback(config["languages"], max_len=config["max_len"]),
                 NoamSchedule(warmup_steps=config["warmup_steps"], learning_rate=config["learning_rate"], start_steps=start_steps)]
    model.fit_generator(
        training_generator, initial_epoch=start_epoch, epochs=300, callbacks=callbacks, validation_data=validation_generator, workers=1)
