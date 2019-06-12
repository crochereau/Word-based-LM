import pickle

import numpy as np
import pandas as pd
import torch

from model import WordNLM


def encode_words(sentences, word_to_idx):
    """

    Args:
        sentences:
        word_to_idx:

    Returns:

    """
    encoded_sentences = [[] for _ in range(len(sentences))]
    for i in range(len(sentences)):
        for j in range(len(sentences[i])):
            word = sentences[i][j]
            encoded_sentences[i].append(word_to_idx[word] if word in word_to_idx else 2)

    return encoded_sentences


def generate_german_dict(path):
    # TODO: path is "vocabularies/de_dict.csv"
    """

    Args:
        path: path to vocabulary of German nouns

    Returns:
        german_dict:
            keys : singular nominative
            values: noun declensions

    """
    df = pd.read_csv(path)
    df = df[['lemma', 'genus', 'akkusativ singular', 'akkusativ singular 1', 'dativ singular',
             'dativ singular 1', 'genitiv singular', 'genitiv singular 1',
             'nominativ plural', 'nominativ plural 1', 'akkusativ plural', 'akkusativ plural 1', 'dativ plural',
             'dativ plural 1', 'genitiv plural', 'genitiv plural 1']]
    df = df.rename(columns={
        'lemma': 'nom sg',
        'akkusativ singular': 'acc sg',
        'akkusativ singular 1': 'acc1 sg',
        'dativ singular': 'dat sg',
        'dativ singular 1': 'dat1 sg',
        'genitiv singular': 'gen sg',
        'genitiv singular 1': 'gen1 sg',
        'nominativ plural': 'nom pl',
        'nominativ plural 1': 'nom1 pl',
        'akkusativ plural': 'acc pl',
        'akkusativ plural 1': 'acc1 pl',
        'dativ plural': 'dat pl',
        'dativ plural 1': 'dat1 pl',
        'genitiv plural': 'gen pl',
        'genitiv plural 1': 'gen1 pl'
    })

    df = df.replace(np.nan, '', regex=True)
    df['acc sg'] = df['acc sg'] + df['acc1 sg']
    df['dat sg'] = df['dat sg'] + df['dat1 sg']
    df['gen sg'] = df['gen sg'] + df['gen1 sg']
    df['nom pl'] = df['nom pl'] + df['nom1 pl']
    df['acc pl'] = df['acc pl'] + df['acc1 pl']
    df['dat pl'] = df['dat pl'] + df['dat1 pl']
    df['gen pl'] = df['gen pl'] + df['gen1 pl']
    df = df.drop(['acc1 sg', 'dat1 sg', 'gen1 sg', 'nom1 pl', 'acc1 pl', 'dat1 pl', 'gen1 pl'], axis=1)
    print(df.shape)
    german_dict = df.set_index('nom sg').T.to_dict('list')
    return german_dict


def gender_tokenizer(intervening_elements, sentences, word_to_idx):
    """
    Args:
        intervening_elements: number of elements between article and noun
        sentences: text file
        word_to_idx: vocabulary mapping word to unique integers

    Returns:
        encoded_sentences: list of lists of encoded sentences, size = number of sentences
    """
    tokens = sentences.split()
    tokenized_sentences = [tokens[x:x + intervening_elements] for x in range(0, len(tokens), intervening_elements)]

    # Encode words into integers using vocabulary
    encoded_sentences = [[] for _ in range(len(tokenized_sentences))]
    for i in range(len(tokenized_sentences)):
        for j in range(len(tokenized_sentences[i])):
            word = tokenized_sentences[i][j]
            # 0: used for padding; 1: unused; 2: for OOV words
            encoded_sentences[i].append(word_to_idx[word] if word in word_to_idx else 2)

    # Add dot at beginning and end of stimuli
    for i in range(len(encoded_sentences)):
        encoded_sentences[i].insert(0, 3)
        encoded_sentences[i].append(3)

    return encoded_sentences


def generate_vocab_mappings(path):
    """
    Args:
        path: path to the vocabulary

    Returns:
        stoi: Word to index mapping
        itos: Index to word mapping
    """
    with open(path, "r") as inFile:
        itos = [x.split("\t")[0] for x in inFile.read().strip().split("\n")[:50003]]
    stoi = dict((tok, i) for i, tok in enumerate(itos))

    return itos, stoi


def load_sentences(input_path):
    """
    Input: str, path to input file
    Output: loaded file
    """
    with open(input_path, "r") as f:
        output = f.read()

    return output


def load_sRNN_model(weight_path, model, device):
    """
    Args:
        weight_path: path to best saved model
        model: a language model
        device: computing device

    Returns:
        model: loaded model
    """
    assert isinstance(model, simpleRNN)
    torch.load(weight_path, map_location=device)

    return model


def load_WordNLM_model(weight_path, model, device, which_LM):
    """
    Args:
        weight_path: path to best saved model
        model: a language model
        device: computing device

    Returns:
        model: loaded model
    """
    assert isinstance(model, WordNLM)

    if which_LM == "base_model":
        checkpoint = torch.load(weight_path, map_location=device)
        named_modules = {"rnn": model.rnn, "output": model.output, "char_embeddings": model.char_embeddings}
        for name, module in named_modules.items():
            print(checkpoint[name].keys())
            module.load_state_dict(checkpoint[name])
    else:
        torch.load(weight_path, map_location=device)

    return model


def pickle_dump(input_file, path):
    """

    Args:
        input_file: file to save
        path: where to save file

    Returns:
        dumped_file: saved file

    """
    with open(path, "wb") as fp:
        dumped_file = pickle.dump(input_file, fp)
    return dumped_file


def pickle_load(path):
    """

    Args:
        path: path to saved file

    Returns:
        loaded_file: loaded saved file

    """
    with open(path, "rb") as fp:
        loaded_file = pickle.load(fp)

    return loaded_file


def tokenizer(sentences):
    """
    Input: loaded text file
    Output: tokenized text
    """
    sentences = sentences.replace(".", ". <eos>")  # adding end of sentence symbols for tokenization
    sentences = sentences.split("<eos>")  # separates sentences
    sentences.pop()  # removes last element (a whitespace)

    for i in range(len(sentences)):  # tokenizes sentences
        sentences[i] = sentences[i].replace(",", " ,").replace(".", " .").replace(":", " :")\
            .replace("?", " ?").replace("!", " !").replace(";", " ;")
        sentences[i] = sentences[i].lower()
        sentences[i] = sentences[i].split()

    return sentences