import pandas as pd
import bpe
import os
from itertools import product
import pickle

EOS_TOKEN = "[eos]"
SOS_TOKEN = "[sos]"
HTLM_CODES = ["&quot;", "&apos;", "&amp;"]

DATA_FOLDER = "../data/"

def language_token(language):
    return f"[{language}]"

def word_tokenizer(text):
    """
    Dataset is already tokenized
    """
    return text.split(" ")

def ted_preprocessing(languages, data_folder=DATA_FOLDER):
    for type in ["_train", "_dev", "_test"]:
        ted_talks = pd.read_csv(os.path.join(data_folder, "all_talks"+type+".tsv"), sep="\t")[languages]
        for lang in languages:
            ted_talks[lang] = ted_talks[lang].apply(lambda x: f"{x} {EOS_TOKEN}")
        ted_talks.to_csv(os.path.join(data_folder, "_".join(languages)+type+".tsv"), sep="\t")

def fit_bpe(languages, data_folder=DATA_FOLDER):
    corpus = []
    for type in ["_train", "_dev", "_test"]:
        ted_talks = pd.read_csv(os.path.join(data_folder, "_".join(languages)+type+".tsv"), sep="\t")[languages]
        for lang in languages:
            ted_lang = ted_talks[lang].dropna()
            sentences = ted_lang[~ted_lang.str.contains("__null__|__NULL__|_ _ NULL _ _")].values.tolist()
            corpus.extend(sentences)
    encoder = bpe.Encoder(required_tokens=[EOS_TOKEN, SOS_TOKEN]+HTLM_CODES+[language_token(lang) for lang in languages],
                          word_tokenizer=word_tokenizer)
    encoder.fit(corpus)
    encoder.save(os.path.join(data_folder, "_".join(languages)+"_bpe.json"))

def create_data(languages, data_folder=DATA_FOLDER):
    encoding = bpe.Encoder.load(os.path.join(data_folder, "_".join(languages)+"_bpe.json"))
    encoding.word_tokenizer = word_tokenizer
    for type in ["_train", "_dev", "_test"]:
        data = []
        ted_talks = pd.read_csv(os.path.join(data_folder, "_".join(languages)+type+".tsv"), sep="\t")[languages]
        for pair in product(languages, repeat=2):
            pair = list(pair)
            if pair[0] != pair[1]:
                pair_talks = ted_talks[pair].copy()
                pair_talks = pair_talks[pair_talks.apply(lambda r: ~r.str.contains("__null__|__NULL__|_ _ NULL _ _"))].dropna()
                pair_talks[pair[0]] = pair_talks[pair[0]].apply(lambda x: f"{language_token(pair[1])} {x}")
                pair_talks[pair[1]] = pair_talks[pair[1]].apply(lambda x: f"{SOS_TOKEN} {x}")
                pairs_as_list = pair_talks.values.tolist()
                pairs_as_list = pairs_as_list[:int(0.1*len(pairs_as_list))]
                pairs_as_list = list(map(list, zip(*[[list(encoding.transform([pair[0]]))[0], list(encoding.transform([pair[1]]))[0]]
                                           for pair in pairs_as_list])))
                if len(data) == 0:
                    data.extend(pairs_as_list)
                else:
                    data[0].extend(pairs_as_list[0])
                    data[1].extend(pairs_as_list[1])
                print(f"{type} Language pair {pair}: {len(pairs_as_list[0])} sentences")
        pickle.dump(data, open(os.path.join(data_folder, "_".join(languages)+type+"_enc.pkl"), "wb"))

def get_data(languages, data_folder=DATA_FOLDER):
    datas = []
    for type in ["_train", "_dev", "_test"]:
        data = pickle.load(open(os.path.join(data_folder, "_".join(languages)+type+"_enc.pkl"), "rb"))
        datas.append(data)
    return datas

def get_encoding_info(languages, data_folder=DATA_FOLDER):
    encoding = bpe.Encoder.load(os.path.join(data_folder, "_".join(languages)+"_bpe.json"))
    return encoding.vocab_size, list(encoding.transform([encoding.PAD]))[0][0]

def get_encoder(languages, data_folder=DATA_FOLDER):
    encoding = bpe.Encoder.load(os.path.join(data_folder, "_".join(languages)+"_bpe.json"))
    encoding.word_tokenizer = word_tokenizer
    return encoding

if __name__ == "__main__":
    #ted_preprocessing(["en", "de"])
    #fit_bpe(["en", "de"])
    create_data(["en", "de"])
