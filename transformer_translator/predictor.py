from transformer_translator.ted_data_preprocessor import get_encoder, language_token, SOS_TOKEN, EOS_TOKEN
from transformer_translator.transformer import Attention, PositionalEncoding
import numpy as np
from keras.models import load_model
from keras.callbacks import Callback


class Predictor:
    def __init__(self, model, languages, data_folder="data"):
        self.model = model
        self.encoding = get_encoder(languages, data_folder=data_folder)
        self.sos_idx = list(self.encoding.transform([SOS_TOKEN]))[0][0]
        self.eos_idx = list(self.encoding.transform([EOS_TOKEN]))[0][0]

    def predict(self, sentence, target_language, max_length=50, sample=True, attempts=1,  only_best_out=False):
        source = list(self.encoding.transform([f"{language_token(target_language)} "+sentence]))
        print(f"Source: {list(self.encoding.inverse_transform(source))}")
        source = np.array(source)
        outputs = []
        probabilities = []
        for i in range(attempts):
            output = np.array([self.sos_idx])
            last_output_idx = self.sos_idx
            iterations = 0
            output_probability = 1
            while iterations < max_length and last_output_idx != self.eos_idx:
                prediction = self.model.predict([source, np.array([output])])[:, -1].squeeze()
                if sample:
                    last_output_idx = np.random.choice(self.encoding.vocab_size, 1, p=prediction).item()
                else:
                    last_output_idx = np.argmax(prediction).item()
                output_probability *= prediction[last_output_idx].item()
                output = np.append(output, last_output_idx)
                iterations += 1
            output = output.tolist()

            if iterations == max_length:
                print(f"Translation stopped because max length of {max_length} was reached.")
                translated = list(self.encoding.inverse_transform([output[1:]]))[0]
            else:
                translated = list(self.encoding.inverse_transform([output[1:-1]]))[0]
            if not only_best_out:
                print(translated)
                print(f"Confidence {output_probability}")
            outputs.append(translated)
            probabilities.append(output_probability)
        print(f"Best translation out of {attempts}:")
        max_prob = np.argmax(probabilities)
        print(outputs[max_prob])
        print(f"Confidence {probabilities[max_prob]}")


DEFAULT_SENTENCE_LANGUAGE_PAIRS_EN_DE = [
    ("We who are diplomats , we are trained to deal with conflicts between states and issues between states .", "de"),
    ("Und wir wissen nicht , wie mit ihnen umzugehen ist .", "en"),
    ("Those portraits make us rethink how we see each other .", "de"),
    ("Er saß neben mir und ich blickte ihn an .", "en")]


class TranslationCallback(Callback):
    def __init__(self, languages, data_folder="data", max_len=50, sentence_language_pairs=DEFAULT_SENTENCE_LANGUAGE_PAIRS_EN_DE):
        super(TranslationCallback, self).__init__()
        self.max_len = max_len
        self.languages = languages
        self.data_folder = data_folder
        self.sentences = sentence_language_pairs
        self.predictor = None

    def on_epoch_end(self, epoch, logs=None):
        if self.predictor is None:
            predictor = Predictor(self.model, self.languages, self.data_folder)
        for pair in self.sentences:
            predictor.predict(pair[0], pair[1], max_length=self.max_len, only_best_out=True)


if __name__ == "__main__":
    model = load_model("../out/2019-06-09_15-40/best-model.hdf5",
                       custom_objects={"PositionalEncoding": PositionalEncoding, "Attention": Attention})
    predictor = Predictor(model, ["en", "de"], "../data")
    for pair in DEFAULT_SENTENCE_LANGUAGE_PAIRS_EN_DE:
        predictor.predict(pair[0], pair[1], max_length=50, attempts=10, only_best_out=True)
