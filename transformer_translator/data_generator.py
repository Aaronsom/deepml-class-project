import numpy as np
from keras.utils import Sequence



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
        print(f"{len(self.indices)} sentencens with max length of {max_len}. {len(data[0])} sentences in total.")
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, item):
        indices = self.indices[item*self.batch_size: (item+1)*self.batch_size]
        input_source = self.pad(self.sources[indices])
        input_targets = self.pad(self.targets[indices])
        output_labels = np.expand_dims(self.pad(self.targets[indices], extra_padding=1)[:, 1:], axis=-1)
        return [input_source, input_targets], output_labels

    def __len__(self):
        return len(self.indices)//self.batch_size

    def pad(self, batch, extra_padding=0):
        max_len = max([len(b) for b in batch])
        batch = np.array([b + [self.mask_id]*(max_len+extra_padding-len(b)) for b in batch])
        return batch
