from paths import WIKIPEDIA_HOME

def load(language, partition, removeMarkup=True):
    if language == "italian":
        path = WIKIPEDIA_HOME+"/itwiki-"+partition+"-tagged.txt"
    elif language == "english":
        path = WIKIPEDIA_HOME+"/english-"+partition+"-tagged.txt"
    elif language == "german":
        path = WIKIPEDIA_HOME+""+language+"-"+partition+"-tagged.txt"
    else:
        assert False
    chunk = []

    with open(path, "r") as in_file:
        for line in in_file:
            index = line.find("\t")
            if index == -1:
                if removeMarkup:
                    continue
                else:
                    index = len(line)-1
            # index is the number of characters (letters) in a word

            word = line[:index]
            chunk.append(word.lower())

            if len(chunk) > 40000:
                #   random.shuffle(chunk)
                yield chunk
                chunk = []

    yield chunk


def count_elements(language, partition, removeMarkup=True):
    if language == "italian":
        path = WIKIPEDIA_HOME+"/itwiki-"+partition+"-tagged.txt"
    elif language == "english":
        path = WIKIPEDIA_HOME+"/english-"+partition+"-tagged.txt"
    elif language == "german":
        path = WIKIPEDIA_HOME+""+language+"-"+partition+"-tagged.txt"
    else:
        assert False

    counter = 0
    with open(path, "r") as in_file:
        for line in in_file:
            index = line.find("\t")
            if index == -1:
                if removeMarkup:
                    continue
                else:
                    index = len(line)-1
            # index is the number of characters (letters) in a word

            word = line[:index]
            counter += 1

    return counter


def training(language):
    return load(language, "train")


def dev(language, removeMarkup=True):
    return load(language, "valid", removeMarkup=removeMarkup)


def test(language, removeMarkup=True):
    return load(language, "test", removeMarkup=removeMarkup)



