"""Test LM"""

import math
import random
import sys
import time

import torch
import torch.nn as nn

from paths import LOG_HOME
from paths import MODELS_HOME
from utils import generate_vocab_mappings, load_WordNLM_model, prepare_dataset_chunks

import corpus_iterator_wiki_words
from lm_argparser import parser
from model import WordNLM

CHAR_VOCAB_PATH = "vocabularies/german-wiki-word-vocab-50000.txt"


def front_pass(args, device, model, numeric, criterion, hidden, beginning, vocab, print_here=False, train=True):
    # batch_size = len(numeric)   # Is this it? # FIXME: remove line
    zero_beginning = torch.zeros((1, args.batch_size), dtype=torch.long).to(device)
    if hidden is None or (train and random.random() > 0.9):
        hidden = None
        beginning = zero_beginning
    elif hidden is not None:
        hidden = tuple(x.data.detach() for x in hidden)

    numeric = torch.cat([beginning, numeric], dim=0).to(device=device)
    beginning = numeric[numeric.size()[0]-1].view(1, args.batch_size)

    input_tensor = numeric[:-1]
    target_tensor = numeric[1:]

    log_probs = model.forward(input_tensor)
    loss = criterion(log_probs.view(-1, len(vocab)), target_tensor.view(-1))

    return loss, target_tensor.view(-1).size()[0], hidden, beginning


def run_epoch_eval(args, device, model, criterion, dev_chars, vocab):
    dev_loss = 0
    dev_char_count = 0
    hidden, beginning = None, None
    for counter, numeric in enumerate(dev_chars):
        print_here = (counter % 50 == 0)
        loss, number_of_characters, hidden, beginning = front_pass(args, device, model, numeric, criterion, hidden,
                                                                   beginning, vocab,print_here=print_here, train=False)
        dev_loss += number_of_characters * loss.cpu().data.numpy()
        dev_char_count += number_of_characters
    dev_loss /= dev_char_count
    print(f"Validation loss {dev_loss}")
    return dev_loss


def main():
    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    itos, stoi = generate_vocab_mappings(CHAR_VOCAB_PATH)
    print('len vocabulary:', len(stoi))
    model = WordNLM(args.word_embedding_size, len(itos), args.hidden_dim, args.layer_num)
    model.to(device)
    if args.load_from is not None:
        if args.load_from == "base_model":
            weight_path = MODELS_HOME + "/" + args.load_from + ".pth.tar"
        else:
            weight_path = MODELS_HOME + args.load_from
        model = load_WordNLM_model(weight_path, model, device, args.load_from)
    else:
        assert False

    criterion = nn.NLLLoss(ignore_index=0)

    total_start_time = time.time()
    test_losses = []

    test_ppl = []

    # Testing over one epoch
    test_data = corpus_iterator_wiki_words.test(args.language)
    test_encoded_words = prepare_dataset_chunks(test_data, stoi, args, device, train=False)
    model.eval()

    test_loss = run_epoch_eval(args, device, model, criterion, test_encoded_words, itos)
    test_losses.append(test_loss)
    test_ppl.append(math.exp(test_loss))

    with open(LOG_HOME+args.language+"_"+__file__+"_"+str(args.my_id), "w") as out_file:
        print("Test loss:", file=out_file)
        print(" ".join([str(x) for x in test_losses]), file=out_file)
        print("Perplexity:", file=out_file)
        print(" ".join([str(x) for x in test_ppl]), file=out_file)
        print(" ".join(sys.argv), file=out_file)
        print(str(args), file=out_file)

    # Momentum peut pas marcher si pas 0
    #learning_rate *= args.lr_decay
    #optim = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.0)  # 0.02, 0.9


if __name__ == "__main__":
    main()