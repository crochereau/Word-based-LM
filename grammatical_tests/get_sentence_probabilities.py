import math
import numpy as np
#import tqdm

import torch



def compute_logprob(tokenized_sentences, model, vocab_mapping, device):
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
        losses: vector of NLL losses, size = number of sentences
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

    target = input_tensor_forward[1:]
    print("target done")
    input_cut = input_tensor_forward[:-1]
    print("input_cut done")

    model.eval()
    print("evaluation done")
    prediction = model(input_cut)
    predictions = prediction.detach().numpy()
    print("predictions done")
    # predictions.shape = maxLength * number of sentences * vocabulary size

    loss_module = torch.nn.NLLLoss(reduction='none', ignore_index=0)
    print("loss module")
    losses = loss_module(prediction.reshape(-1, len(vocab_mapping)),
                                        target.reshape(-1)).reshape(maxLength, len(tokenized_sentences))
    losses = losses.sum(0).data.cpu().numpy()
    print("losses done")

    return tokenized_sentences, predictions, losses


def per_sentence_probabilities(padded_sentences, log_predictions):
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



