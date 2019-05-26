import poem_generator.embedding as embedding_loader
from poem_generator.global_constants import START_OF_SEQUENCE_TOKEN, END_OF_LINE_TOKEN, END_OF_SEQUENCE_TOKEN, \
    PADDING_TOKEN, OUT_OF_VOCAB_TOKEN, MODELS_DICT
import numpy as np
from keras.models import load_model
from keras.backend import set_epsilon, set_floatx
from poem_generator.transformer import Attention, PositionalEncoding
import zipfile

def generate_poem(model, reverse_dictionary, dictionary, seed_length, dynamic_seed=False):
    poem = ""
    last_output = ""
    iterations = 0
    seed = np.array([dictionary[START_OF_SEQUENCE_TOKEN]]*seed_length)
    already_eol = False  # Sometimes too many eols are generated, this breaks the format
    while iterations < 60 and last_output != END_OF_SEQUENCE_TOKEN:
        last_output_dist = model.predict(np.array([seed]))[:, -1].squeeze()
        last_output_idx = np.random.choice(len(dictionary), 1, p=last_output_dist).item()
        last_output = reverse_dictionary[last_output_idx]


        iterations += 1

        if last_output == END_OF_SEQUENCE_TOKEN or iterations == 60:
            if iterations < 15:
                print("Too short. Try again\n")
                return generate_poem(model, reverse_dictionary, dictionary, seed_length, dynamic_seed)
            if already_eol:
                poem += "\n"
            else:
                poem += "\n\n"
        elif last_output == OUT_OF_VOCAB_TOKEN or last_output == PADDING_TOKEN:
            iterations -= 1
        elif last_output == START_OF_SEQUENCE_TOKEN:
            pass
        elif last_output == END_OF_LINE_TOKEN:
            if iterations>1 and not already_eol:
                already_eol = True
                poem += "\n"

        else:
            already_eol = False
            poem += last_output + " "
        if last_output != OUT_OF_VOCAB_TOKEN and last_output != PADDING_TOKEN:
            if not dynamic_seed:
                seed = np.append(seed[1:], last_output_idx)
            else:
                seed = np.append(seed, last_output_idx)
    print(poem)
    return poem

def generate_poems(num_of_poems, seed_length, output_filename, model_file):
    _, dictionary = embedding_loader.get_embedding(None, load=True)
    reverse_dictionary = {dictionary[key]: key for key in dictionary.keys()}
    model = load_model(model_file, custom_objects={"PositionalEncoding": PositionalEncoding, "Attention": Attention})
    generated_poems = ""
    for i in range(num_of_poems):
        print(f"{i+1}/{num_of_poems}")
        poem = generate_poem(model, reverse_dictionary, dictionary, seed_length)
        generated_poems += poem
    #with open(output_filename, "w", encoding="utf-8") as file:
    #    file.write(generated_poems)
    zipf = zipfile.ZipFile(output_filename, "w", zipfile.ZIP_DEFLATED)
    zipf.writestr("/poems.txt", generated_poems)
    zipf.close()

if __name__ == "__main__":
    set_floatx("float16")
    set_epsilon(1e-04)
    ns = [4]
    for n in ns:
        generate_poems(1000, n, "../generated/"+str(n)+"-poems.zip", MODELS_DICT+"/4-model.hdf5")