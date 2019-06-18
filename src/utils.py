import math
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


def get_sentences_probs(padded_sentences, log_predictions):
    """

    Args:
        padded_sentences: vector of encoded sentences padded with zeros, size = number of sentences
        log_predictions: matrix of per-word log probabilities, size = length of longer sentence *
                                                                    number of sentences * vocabulary size

    Returns:
        sentence_probs: list of sentences probabilities, size = number of sentences
    """

    word_log_probs = [[] for _ in range(len(padded_sentences))]

    # fetch word log probabilities from log probabilities matrix
    for sentence_idx in range(len(padded_sentences)):
        for word_idx in range(len(padded_sentences[sentence_idx])):
            word = padded_sentences[sentence_idx][word_idx]
            # leave out probabilities of padded zeros
            if word != 0:
                # append each word log probability to sentence probabilities matrix
                word_log_probs[sentence_idx].append(log_predictions[word_idx][sentence_idx][word])

    sentence_probs = []
    for i in range(len(word_log_probs)):
        sentence_probs.append(math.exp(sum(word_log_probs[i])))
    sentence_probs = np.array(sentence_probs)

    return sentence_probs


def get_words_logprobs(tokenized_sentences, model, vocab_mapping, device):
    """
    Args:
        tokenized_sentences: list of lists of encoded sentences
        model: a language model
        vocab_mapping: dictionary mapping words to unique integers
        device: computing device

    Returns:
        tokenized_sentences: vector of encoded sentences padded with zeros, size = number of sentences
        predictions: matrix of per-word log probabilities, size = length of longer sentence *
                                                                    number of sentences * vocabulary size
    """
    maxLength = max([len(x) for x in tokenized_sentences])
    print("maxLength:", maxLength)

    # padding shorter sentences with zeros
    for i in range(len(tokenized_sentences)):
        while len(tokenized_sentences[i]) < maxLength:
            tokenized_sentences[i].append(0)

    input_tensor_forward = torch.tensor([[0]+x for x in tokenized_sentences], dtype=torch.long,
                                    device=device, requires_grad=False).transpose(0, 1)
    print("input_tensor_forward done")

    #target = input_tensor_forward[1:]
    input_cut = input_tensor_forward[:-1]

    with torch.no_grad():
        model.eval()
        prediction = model(input_cut)
    predictions = prediction.detach().numpy()

    # predictions.shape = maxLength * number of sentences * vocabulary size

    """
    loss_module = torch.nn.NLLLoss(reduction='none', ignore_index=0)
    print("loss module")
    losses = loss_module(prediction.reshape(-1, len(vocab_mapping)),
                                        target.reshape(-1)).reshape(maxLength, len(tokenized_sentences))
    losses = losses.sum(0).data.cpu().numpy()
    print("losses done")
    """

    return tokenized_sentences, predictions


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


def prepare_dataset_chunks(data, stoi, args, device, train=True):
    print("Prepare chunks")
    encoded_sentences = []
    for chunk in data:
        for word in chunk:

            # In eval or test condition, introduce a random word from the vocabulary
            # FIXME: as args.char_noise_prob = 0, introduces noise all the time?
            # FIXME: What is a good value for args.char_noise_prob? (which should be called noisy_word_prob)
            # Replaced numerified by encoded_sentence
            """
            if train or random.random() > args.char_noise_prob:
                numerified.append(random.randint(3, len(stoi)))
            else:
                # get word index in word-index mapping; returns 2 if OOV word
                numerified.append(stoi.get(word, 2))
            """
            encoded_sentences.append(stoi.get(word, 2))

        if len(encoded_sentences) > (args.batch_size * args.sequence_length):   # 40,001 > 128 * 10
            seq_length = args.sequence_length

            first_mult = (len(encoded_sentences) // (args.batch_size * seq_length)) # 31

            cutoff = first_mult * args.batch_size * seq_length   # 39,680 = 31 * 128 * 10
            selected_sentences = encoded_sentences[:cutoff]   # len = 39,680
            encoded_sentences = encoded_sentences[cutoff:]   # len = 321

            # size of selected sentences: (31, 10, 128)
            selected_sentences = torch.tensor(selected_sentences, dtype=torch.long,
                                    device=device).view(args.batch_size, -1, seq_length).transpose(0, 1).transpose(1, 2)
            number_of_sequences = selected_sentences.size()[0]   # 31

            for i in range(number_of_sequences):
                yield selected_sentences[i]   # size: 10 * 128 = args.sequence_length * args_batch.size
        else:
            print("Skipping")


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