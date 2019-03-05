import math

import torch


def compute_logprob(tokenized_sentences, model, vocab_mapping, device):
    """
    Compute log probabilities of words of input sentences.
    Input:
        tokenized_sentences: list of lists encoding sentences
        model: a language model
        vocab_mapping: itos
        device: device
    Outputs:
        - vector containing padded sentences,
        - prediction table containing log probabilities for each word of vocabulary, for each word of input sentences
            size: length_of_longer_input_sentence * number_of_sentences * vocabulary_size
    """
    maxLength = max([len(x) for x in tokenized_sentences])
    print("maxLength:", maxLength)

    # padding shorter sentences with zeros
    for i in range(len(tokenized_sentences)):
        while len(tokenized_sentences[i]) < maxLength:
            tokenized_sentences[i].append(0)
    print("tokenized_sentences:", tokenized_sentences)

    input_tensor_forward = torch.tensor([[0]+x for x in tokenized_sentences], dtype=torch.long,
                                        device=device, requires_grad=False).transpose(0, 1)

    target = input_tensor_forward[1:]
    input_cut = input_tensor_forward[:-1]

    model.eval()
    prediction = model(input_cut)
    predictions = prediction.detach().numpy()

    loss_module = torch.nn.NLLLoss(reduction='none', ignore_index=0)
    losses = loss_module(prediction.reshape(-1, len(vocab_mapping)+3), target.reshape(-1)).reshape(maxLength, len(tokenized_sentences))
    losses = losses.sum(0).data.cpu().numpy()

    return tokenized_sentences, predictions, losses


def per_sentence_probabilities(padded_sentences, log_predictions):
    """
    Get per-word log probabilities for each input sentence from prediction table
    Compute per-sentence probabilities.
    Input:
    Output:
    """

    word_log_probs = [[] for _ in range(len(padded_sentences))]

    # get per-word log probabilities from log_predictions table
    for i in range(len(padded_sentences)):
        for j in range(len(padded_sentences[i])):
            k = padded_sentences[i][j]
            if k != 0:      # because of padding with zeros
                word_log_probs[i].append(log_predictions[j][i][k])

    sentence_probs = []
    for i in range(len(word_log_probs)):
        sentence_probs.append(math.exp(sum(word_log_probs[i])))

    return sentence_probs
