from keras.callbacks import Callback
import keras.backend as K
import math


class NoamSchedule(Callback):
    def __init__(self, warmup_steps=4000, learning_rate=0.2, start_steps=0):
        self.warmup_steps = warmup_steps
        self.lr = learning_rate
        self.global_steps = start_steps

    def on_batch_begin(self, batch, logs=None):
        self.global_steps += 1
        new_lr = self.lr * min(math.pow(self.global_steps, -0.5), self.global_steps*math.pow(self.warmup_steps, -1.5))
        print(" lr "+str(new_lr))
        K.set_value(self.model.optimizer.lr, new_lr)