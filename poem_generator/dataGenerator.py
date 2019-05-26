import numpy as np
from poem_generator.embedding import tuple_to_indices
from poem_generator.data_prepocessing import multi_ngram_tupelizer
from keras.utils import Sequence

def label_smoothing(labels, vocab_len, smoothing):
    array_labels = np.zeros(shape=(len(labels), len(labels[0]), vocab_len), dtype="float16")
    array_labels = array_labels + smoothing/(vocab_len-1)
    for i, label in enumerate(labels):
        for j, val in enumerate(label):
            array_labels[i, j, val] = 1-smoothing
    return array_labels


class TupleDataGenerator(Sequence):

    def __init__(self, poems, ns, dictionary, smoothing, batch_size, shuffle=True):
        self.vocab_len = len(dictionary)
        self.smoothing = smoothing
        self.batch_size = batch_size
        self.shuffle = shuffle
        tuples = multi_ngram_tupelizer(poems, ns)
        self.data = [[np.array(item) for item in zip(*tuple_to_indices(tuple, dictionary, True))] for tuple in tuples]
        length = sum([len(tuple) for tuple in tuples])
        print(f"Training on {self.vocab_len} words with {length} {ns}-tuples")
        self.indexes = [(i, batch) for i, data in enumerate(self.data)
                        for batch in np.split(
                np.random.permutation(len(data[0]))[:self.batch_size*int(len(data[0])/self.batch_size)], int(len(data[0])/self.batch_size))]
        self.on_epoch_end()

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)


    def __getitem__(self, item):
        idx, batch_idx = self.indexes[item]
        data = self.data[idx][0][batch_idx]
        labels = self.data[idx][1][batch_idx]
        labels = label_smoothing(labels, self.vocab_len, self.smoothing)
        return data, labels

    def __len__(self):
        return len(self.indexes)
