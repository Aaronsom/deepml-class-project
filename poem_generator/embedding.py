from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import pickle
import poem_generator.data_prepocessing as dp
from poem_generator.global_constants import START_OF_SEQUENCE_TOKEN, END_OF_LINE_TOKEN, END_OF_SEQUENCE_TOKEN, \
    OUT_OF_VOCAB_TOKEN, PADDING_TOKEN, EMBEDDING_PATH, EMBEDDING_DIMENSION, MODELS_DICT

EMBEDDING_FILE = MODELS_DICT+"/embedding.pkl"
DICTIONARY_FILE = MODELS_DICT+"/dict.pkl"


def get_embedding(words, binary=False, save=False, load=False, limit=40000):
    if load:
        embedding = pickle.load(open(EMBEDDING_FILE, "rb"))
        dictionary = pickle.load(open(DICTIONARY_FILE, "rb"))
        return embedding, dictionary
    entire_embedding = KeyedVectors.load_word2vec_format(EMBEDDING_PATH, limit=limit, binary=binary)
    # init with special tokens
    embedding = [np.zeros(shape=EMBEDDING_DIMENSION), np.random.normal(size=EMBEDDING_DIMENSION),
                 np.random.normal(size=EMBEDDING_DIMENSION),
                 np.random.normal(size=EMBEDDING_DIMENSION), np.random.normal(size=EMBEDDING_DIMENSION)]
    dictionary = {PADDING_TOKEN: 0, OUT_OF_VOCAB_TOKEN: 1, END_OF_SEQUENCE_TOKEN: 2, END_OF_LINE_TOKEN: 3, START_OF_SEQUENCE_TOKEN: 4}
    for word in words:
        if word in dictionary:
            continue
        try:
            word_embedding = entire_embedding[word]
        except KeyError:
            continue # keep vocab small
        dictionary[word] = len(dictionary)
        embedding.append(word_embedding)
    embedding = np.array(embedding)
    if save:
        pickle.dump(embedding, open(EMBEDDING_FILE, "wb"))
        pickle.dump(dictionary, open(DICTIONARY_FILE, "wb"))
    return embedding, dictionary

def tuple_to_indices(ngram_tuples, dictionary, remove_oov_labels=False):
    index_ngram_tuples = []
    for ngram, label in ngram_tuples:
        index_ngram = []
        index_label = []
        for word in ngram:
            try:
                index_ngram.append(dictionary[word])
            except KeyError:
                index_ngram.append(dictionary[OUT_OF_VOCAB_TOKEN])
        for word in label:
            try:
                index_label.append(dictionary[word])
            except KeyError:
                index_label.append(dictionary[OUT_OF_VOCAB_TOKEN])
        # try:
        #     index_label = dictionary[label]
        # except KeyError:
        #     if remove_oov_labels:
        #         continue
        #     else:
        #         index_label = dictionary[OUT_OF_VOCAB_TOKEN]
        index_ngram_tuples.append((index_ngram, index_label))
    return index_ngram_tuples

if __name__ == "__main__":
    poems = dp.tokenize_poems("../data/train_poems.txt")
    words = set([token for poem in poems for token in poem])
    embedding, dictionary = get_embedding(words)

    ngram_tuples = dp.ngram_tuplelizer([poems[0]], 7)
    converted_tuple = tuple_to_indices(ngram_tuples, dictionary)
    print(converted_tuple)


