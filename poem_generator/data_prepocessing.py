from poem_generator.global_constants import END_OF_LINE_TOKEN, END_OF_SEQUENCE_TOKEN, START_OF_SEQUENCE_TOKEN

def tokenize_poems(file):
    poems = []
    with open(file, encoding="utf-8") as f:
        poem = []
        for line in f.readlines():
            if line == "\n":
                poem[-1] = END_OF_SEQUENCE_TOKEN
                poems.append(poem)
                poem = []
            else:
                tokens = line.rstrip("\n").lower().split(" ") + [END_OF_LINE_TOKEN]
                poem.extend(tokens)
    return poems

def multi_ngram_tupelizer(poems, ns):
    tuples = []
    for n in ns:
        tuples.append(ngram_tuplelizer(poems, n))
    return tuples

def ngram_tuplelizer(poems, n):
    tuples = []
    poems = [[START_OF_SEQUENCE_TOKEN]*n + poem for poem in poems]
    for poem in poems:
        i = 0
        while i+n < len(poem):
            tuples.append((poem[i:i+n], poem[i+1:i+n+1]))
            i += 1
    return tuples


if __name__ == "__main__":
    poems = tokenize_poems("data/train_poems.txt")
    print(ngram_tuplelizer([poems[0]], 7))