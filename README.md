# Multilingual Machine Translation with the Transformer Model
This is a class project for Deep Learning: Architectures and Methods at the TU Darmstadt in the sumemr semester of 2019.
We code and train a Transformer for multilingual machine translation in Keras.

## How to use this repo
Check requirements.txt for the required packages and install them as needed.

Download the dataset from https://github.com/neulab/word-embeddings-for-nmt and run ted_data_preprocessor.py for the languages you want.

You then can use and adapt basic_training.py to train your model.

Finally, use preditor.py to translate new sentences. Note that there is no tokenizer, so you need to manually tekenize the input at the moment.
