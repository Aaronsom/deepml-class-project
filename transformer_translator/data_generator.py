import numpy as np
from keras.utils import Sequence
from transformer_translator.ted_data_preprocessor import get_data


class DataGenerator(Sequence):
    def __init__(self, data, mask_id, max_len=50, batch_size=32, shuffle=True):
        """"
        Args:
            data: Pair of source and target sentences as lists
        """
        filter = [len(data[0][i])<= max_len or len(data[1][i])<= max_len for i in range(len(data[0]))]
        self.shuffle = shuffle
        self.sources = np.array(data[0])[filter]
        self.targets = np.array(data[1])[filter]
        self.indices = np.arange(len(self.sources))
        self.batch_size = batch_size
        self.mask_id = mask_id
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, item):
        indices = self.indices[item*self.batch_size: (item+1)*self.batch_size]

    def __len__(self):
        return len(self.indexes)
