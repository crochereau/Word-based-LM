from paths import WIKIPEDIA_HOME
import random


def load(language, partition, do_shuffling=True):
    chunks = []
    path_infix = {"german" : "", "english" : "//"}[language]
    with open(WIKIPEDIA_HOME+path_infix+language+"-"+partition+".txt", "r") as in_file:
        for line in in_file:
            chunks.append(line.strip().lower())
            if len(chunks) > 20000:
                if do_shuffling:
                    random.shuffle(chunks)
                yield "".join(chunks)
                chunks = []
    yield "".join(chunks)


def training(language):
    return load(language, "train")


def dev(language, doShuffling=True):
    return load(language, "valid", do_shuffling=doShuffling)

