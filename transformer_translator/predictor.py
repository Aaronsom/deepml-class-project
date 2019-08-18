from transformer_translator.ted_data_preprocessor import get_encoder, language_token, SOS_TOKEN, EOS_TOKEN
from transformer_translator.transformer import Attention, PositionalEncoding
import numpy as np
from keras.models import load_model
from keras.callbacks import Callback


class Beam:
    def __init__(self, max_length, eos, output, ll=0, nll=0, length=1):
        self.output = output
        self.loglikelihood = ll
        self.normalized_loglikelihood = nll
        self.length = length
        self.eos = eos
        self.max_length = max_length
        self.done = False

    def update(self, idx, probability):
        self.output = np.append(self.output, idx)
        self.loglikelihood += np.log(probability)
        self.length += 1
        self.normalized_loglikelihood = self.loglikelihood/self.length
        if self.length == self.max_length or idx == self.eos:
            self.done = True

    def clone(self):
        return Beam(self.max_length, self.eos, self.output, self.loglikelihood, self.normalized_loglikelihood, self.length)

    def extend(self, idxs, probs):
        beams = []
        for idx, prob in zip(idxs, probs):
            beam = self.clone()
            beam.update(idx, prob)
            beams.append(beam)
        return beams


class Predictor:
    def __init__(self, model, languages, data_folder="data"):
        self.model = model
        self.encoding = get_encoder(languages, data_folder=data_folder)
        self.sos_idx = list(self.encoding.transform([SOS_TOKEN]))[0][0]
        self.eos_idx = list(self.encoding.transform([EOS_TOKEN]))[0][0]

    def predict(self, sentence, target_language, max_length=50, mode="greedy", attempts_beam=1):
        if mode is "greedy":
            self.greedy(sentence, target_language, max_length)
        elif mode is "sample":
            self.sample(sentence, target_language, max_length, attempts_beam)
        elif mode is "beam":
            self.beam(sentence, target_language, max_length, attempts_beam)

    def greedy(self, sentence, target_language, max_length=50):
        source = list(self.encoding.transform([f"{language_token(target_language)} "+sentence]))
        print(f"Source: {list(self.encoding.inverse_transform(source))}")
        source = np.array(source)
        output = np.array([self.sos_idx])
        last_output_idx = self.sos_idx
        iterations = 0
        output_probability = 0
        while iterations < max_length and last_output_idx != self.eos_idx:
            prediction = self.model.predict([source, np.array([output])])[:, -1].squeeze()
            last_output_idx = np.argmax(prediction).item()
            output_probability += np.log(prediction[last_output_idx])
            output = np.append(output, last_output_idx)
            iterations += 1
        output = output.tolist()

        if iterations == max_length:
            print(f"Translation stopped because max length of {max_length} was reached.")
            translated = list(self.encoding.inverse_transform([output[1:]]))[0]
        else:
            translated = list(self.encoding.inverse_transform([output[1:-1]]))[0]
        print(translated)
        print(f"Normalized log likelihood {output_probability/len(output)}")

    def sample(self, sentence, target_language, max_length=50, attempts=5):
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
                last_output_idx = np.random.choice(self.encoding.vocab_size, 1, p=prediction).item()
                output_probability *= prediction[last_output_idx].item()
                output = np.append(output, last_output_idx)
                iterations += 1
            output = output.tolist()

            if iterations == max_length:
                print(f"Translation stopped because max length of {max_length} was reached.")
                translated = list(self.encoding.inverse_transform([output[1:]]))[0]
            else:
                translated = list(self.encoding.inverse_transform([output[1:-1]]))[0]
            outputs.append(translated)
            probabilities.append(output_probability)
        print(f"Best translation out of {attempts}:")
        max_prob = np.argmax(probabilities)
        print(outputs[max_prob])
        print(f"Confidence {probabilities[max_prob]}")

    def _update_beams(self, beams, predictions, beam_size):
        new_beams = [beam for beam in beams if beam.done]
        for i, beam in enumerate([beam for beam in beams if not beam.done]):
            idxs = np.argpartition(predictions[i], -beam_size)[-beam_size:]
            probs = predictions[i, idxs]
            new_beams.extend(beam.extend(idxs, probs))
        return new_beams

    def _cull_beams(self, beams, beam_size):
        values = [beam.normalized_loglikelihood for beam in beams]
        if len(values) == beam_size:
            idxs = np.arange((beam_size))
        else:
            idxs = np.argpartition(values, -beam_size)[-beam_size:]
        return [beams[idx] for idx in idxs]

    def beam(self, sentence, target_language, max_length=50, beam_size=12):
        source = np.array(list(self.encoding.transform([f"{language_token(target_language)} "+sentence])))
        print(f"Source: {list(self.encoding.inverse_transform(source))}")
        first_output = np.array([self.sos_idx])
        beams = [Beam(max_length, self.eos_idx, first_output)]

        while not all([beam.done for beam in beams]):
            outputs = np.stack([beam.output for beam in beams if not beam.done])
            repeated_source = np.stack([source for _ in range(len(outputs))]).reshape(len(outputs), -1)
            predictions = self.model.predict([repeated_source, outputs])[:, -1]
            beams = self._update_beams(beams, predictions, beam_size)
            beams = self._cull_beams(beams, beam_size)

        best_idx = np.argmax([beam.normalized_loglikelihood for beam in beams]).item()
        output = beams[best_idx].output
        output = output.tolist()
        norm_loglike = beams[best_idx].normalized_loglikelihood

        if beams[best_idx].length == max_length:
            print(f"Translation stopped because max length of {max_length} was reached.")
            translated = list(self.encoding.inverse_transform([output[1:]]))[0]
        else:
            translated = list(self.encoding.inverse_transform([output[1:-1]]))[0]

        print(translated)
        print(f"Normalized log likelihood {norm_loglike}")

sentence_pairs = {"en_de": [
        ("We who are diplomats , we are trained to deal with conflicts between states and issues between states .", "de"),
        ("Und wir wissen nicht , wie mit ihnen umzugehen ist .", "en"),
        ("Those portraits make us rethink how we see each other .", "de"),
        ("Er saß neben mir und ich blickte ihn an .", "en")],
    "en_de_es": [
        ("We who are diplomats , we are trained to deal with conflicts between states and issues between states .", "de"),
        ("Und wir wissen nicht , wie mit ihnen umzugehen ist .", "en"),
        ("Those portraits make us rethink how we see each other .", "de"),
        ("Er saß neben mir und ich blickte ihn an .", "en"),
        ("There was a big smile on his face which was unusual then , because the news mostly depressed him .", "es"),
        ("Er sah sehr glücklich aus , was damals ziemlich ungewöhnlich war , da ihn die Nachrichten meistens deprimierten .", "es"),
        ("A finales de este año habrá alrededor de mil millones de personas en el planeta que usarán activamente las redes sociales .", "de"),
        ("A finales de este año habrá alrededor de mil millones de personas en el planeta que usarán activamente las redes sociales .", "en")]
}


class TranslationCallback(Callback):
    def __init__(self, languages, data_folder="data", max_len=50):
        super(TranslationCallback, self).__init__()
        self.max_len = max_len
        self.languages = languages
        self.data_folder = data_folder
        lans = "_".join(languages)
        self.sentences = sentence_pairs[lans]
        self.predictor = None

    def on_epoch_end(self, epoch, logs=None):
        if self.predictor is None:
            predictor = Predictor(self.model, self.languages, self.data_folder)
        for pair in self.sentences:
            predictor.predict(pair[0], pair[1], max_length=self.max_len, mode="beam", attempts_beam=12)


if __name__ == "__main__":
    model = load_model("../out/2019-06-09_15-40/best-model.hdf5",
                       custom_objects={"PositionalEncoding": PositionalEncoding, "Attention": Attention})
    predictor = Predictor(model, ["en", "de"], "../data")
    for pair in sentence_pairs["en_de"]:
        predictor.predict(pair[0], pair[1], max_length=50, mode="beam", attempts_beam=12)
