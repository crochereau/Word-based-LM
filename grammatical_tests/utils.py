import torch

from model import WordNLM


def generate_vocab_mappings(path):
    """

    Args:
        path: path to the vocabulary

    Returns:
        stoi: Word to index mapping
        itos: Index to word mapping

    """
    with open(path, "r") as inFile:
        itos = [x.split("\t")[0] for x in inFile.read().strip().split("\n")[:50000]]
    stoi = dict([(itos[i], i) for i in range(len(itos))])
    return itos, stoi


def load_sentences(input_path):
    """
    Input: str, path to input file
    Output: loaded file
    """
    with open(input_path, "r") as f:
        output = f.read()

    return output


def tokenizer(sentences, word_to_idx):
    """
    Input: loaded text file
    Output: tokenized text
    """
    sentences = sentences.replace(".", ". <eos>")  # adding end of sentence symbols for tokenization

    sentences = sentences.split("<eos>")  # separates sentences
    sentences.pop()  # removes last element (a whitespace)
    print("number of test sentences: ", len(sentences))

    for i in range(len(sentences)):  # tokenizes sentences
        sentences[i] = sentences[i].replace(",", " ,").replace(".", " .").replace(":", " :").replace("?", " ?").replace("!", " !").replace(";", " ;")
        sentences[i] = sentences[i].lower()
        sentences[i] = sentences[i].split()
    print(sentences[:10])

    tokenized_sentences = [[] for _ in range(len(sentences))]
    for i in range(len(sentences)):
        for j in range(len(sentences[i])):
            word = sentences[i][j]
            tokenized_sentences[i].append(word_to_idx[word]+3 if word in word_to_idx else 2)

    return tokenized_sentences


def gender_tokenizer(sentences, word_to_idx):
    """
    Input: loaded text file
        sentences: # FIXME
        word_to_idx: s_to_i
    Output: tokenized text
    Difference with tokenizer(): gender stimuli have no punctuation -> slightly different preprocessing
    """
    # FIXME: stoi doit etre un param
    tokens = sentences.split()
    tokenized_sentences = [tokens[x:x + 2] for x in range(0, len(tokens), 2)]

    # Encode words into integers using vocabulary
    encoded_sentences = [[] for _ in range(len(tokenized_sentences))]
    for i in range(len(tokenized_sentences)):
        for j in range(len(tokenized_sentences[i])):
            word = tokenized_sentences[i][j]
            # 0: used for padding; 1: unused; 2: for OOV words
            encoded_sentences[i].append(word_to_idx[word] + 3 if word in word_to_idx else 2)

    for i in range(len(encoded_sentences)):
        encoded_sentences[i].insert(0, 3)
        encoded_sentences[i].append(3)

    return encoded_sentences


def load_WordNLM_model(weight_path, model, device):
    assert isinstance(model, WordNLM)
    checkpoint = torch.load(weight_path, map_location=device)
    named_modules = {"rnn": model.rnn, "output": model.output, "char_embeddings": model.char_embeddings}
    for name, module in named_modules.items():
        print(checkpoint[name].keys())
        module.load_state_dict(checkpoint[name])
    return model
