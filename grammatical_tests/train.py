"""Train LM"""

import argparse
import random
import sys
import time

import tqdm
import torch
import torch.nn as nn

from paths import LOG_HOME
from paths import MODELS_HOME
from utils import generate_vocab_mappings, load_WordNLM_model
from train_data_functions import prepare_dataset_chunks
from model import WordNLM
import corpus_iterator_wiki

CHAR_VOCABS = {"german": "vocabularies/german-wiki-word-vocab-50000.txt",
               "italian": "vocabularies/italian-wiki-word-vocab-50000.txt",
               "english": "vocabularies/english-wiki-word-vocab-50000.txt"}


def get_args(*input_args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", dest="language", type=str)
    parser.add_argument("--load-from", dest="load_from", type=str)
    parser.add_argument("--save-to", dest="save_to", type=str)
    parser.add_argument("--batch_size", type=int, default=random.choice([128, 128, 128, 256]))
    parser.add_argument("--char_embedding_size", type=int, default=random.choice([100, 200, 200, 300, 300, 300, 300, 1024]))
    parser.add_argument("--hidden_dim", type=int, default=random.choice([1024]))
    parser.add_argument("--layer_num", type=int, default=random.choice([1, 2]))
    parser.add_argument("--weight_dropout_in", type=float, default=random.choice([0.0, 0.0, 0.0, 0.01]))
    parser.add_argument("--weight_dropout_hidden", type=float, default=random.choice([0.0, 0.05, 0.15, 0.2]))
    parser.add_argument("--char_dropout_prob", type=float, default=random.choice([0.0, 0.0, 0.001, 0.01, 0.01]))
    parser.add_argument("--char_noise_prob", type = float, default=random.choice([0.0, 0.0]))
    parser.add_argument("--learning_rate", type = float, default= random.choice([0.6, 0.7, 0.8, 0.9, 1.0,1.0,  1.1, 1.1, 1.2, 1.2, 1.2, 1.2, 1.3, 1.3, 1.4, 1.5, 1.6]))
    parser.add_argument("--my_id", type=int, default=random.randint(0,1000000000))
    parser.add_argument("--sequence_length", type=int, default=random.choice([50]))
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--lr_decay", type=float, default=random.choice([0.7, 0.9, 0.95, 0.98, 0.98, 1.0]))

    args=parser.parse_args()


    if "MYID" in args.save_to:
        args.save_to = args.save_to.replace("MYID", str(args.myID))

    print(args)
    return args


def plus(it1, it2):
    for x in it1:
        yield x
    for x in it2:
        yield x


# FIXME: WTF, use the forward from the model
def front_pass(model, numeric, criterion, hidden, beginning, train=True, print_here=False):
    batch_size = len(numeric)   # Is this it?
    zero_beginning = torch.zeros((1, args.batch_size)).to(device)
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
    loss = criterion(log_probs.view(-1, len(itos)), target_tensor.view(-1))
    """
    if print_here:
        lossTensor = print_loss(log_probs.view(-1, len(itos)+3), target_tensor.view(-1)).view(-1, args.batchSize)
        losses = lossTensor.data.cpu().numpy()
        numericCPU = numeric.cpu().data.numpy()
        print(("NONE", itos[numericCPU[0][0]-3]))
        for i in range((args.sequence_length)):
            print((losses[i][0], itos[numericCPU[i+1][0]-3]))
    """
    return loss, target_tensor.view(-1).size()[0], hidden, beginning


def run_epoch_train(optim, model, criterion, training_chars):
    optim.zero_grad()
    start_time = time.time()
    train_chars = 0
    hidden, beginning = None, None
    loss_has_been_bad = 0
    for counter, numeric in enumerate(training_chars):
        print_here = (counter % 50 == 0)
        loss, char_counts, hidden, beginning = front_pass(model, numeric, criterion, hidden, beginning,
                                                          print_here=print_here, train=True)
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), 5.0)
        optim.step()

        if loss > 15:
            loss_has_been_bad += 1
        else:
            loss_has_been_bad = 0
        if loss_has_been_bad > 100:
            print("Loss exploding, has been bad for a while")
            print(loss)
            quit()
        train_chars += char_counts
        if print_here:
            print(f"Counter {counter}, loss {loss}")
            print(f"Chars per sec {train_chars/(time.time()-start_time)}")
    return loss_has_been_bad


def run_epoch_eval(model, criterion, dev_chars):
    dev_loss = 0
    dev_char_count = 0
    hidden, beginning = None, None
    for counter, numeric in enumerate(dev_chars):
        print_here = (counter % 50 == 0)
        loss, number_of_characters, hidden, beginning = front_pass(model, criterion, numeric, hidden, beginning,
                                                                   print_here=print_here, train=False)
        dev_loss += number_of_characters * loss.cpu().data.numpy()
        dev_char_count += number_of_characters
    dev_loss /= dev_char_count
    print(f"Loss of {dev_loss}")
    return dev_loss


def main():
    args = get_args()
    char_vocab_path = CHAR_VOCABS[args.language]
    itos, stoi = generate_vocab_mappings(char_vocab_path)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = WordNLM(args.char_embedding_size, len(itos), args.hidden_dim, args.layer_num,
                    args.weight_dropout_in, args.weight_dropout_hidden, args.char_dropout_prob)
    model.to(device)
    model.train()
    criterion = nn.NLLLoss(ignore_index=0)

    if args.load_from is not None:
        weight_path = torch.load(MODELS_HOME+args.load_from+".pth.tar")
        model = load_WordNLM_model(weight_path, model, device)

    # FIXME: Recuperer l'optimizer?
    learning_rate = args.learning_rate
    optim = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.0)

    total_start_time = time.time()
    dev_losses = []
    max_num_epochs = 10000
    for epoch in tqdm.tqdm(range(max_num_epochs), total=max_num_epochs, desc="Doing epoch"):
        training_data = corpus_iterator_wiki.training(args.language)
        training_chars = prepare_dataset_chunks(training_data, stoi, args, device, train=True)
        model.train()
        run_epoch_train(optim, model, criterion, training_chars)

        dev_data = corpus_iterator_wiki.dev(args.language)
        dev_chars = prepare_dataset_chunks(dev_data, stoi, args, device, train=False)
        model.eval()
        dev_loss = run_epoch_eval(model, criterion, dev_chars)
        dev_losses.append(dev_loss)

        early_stop = len(dev_losses) > 1 and dev_losses[-1] > dev_losses[-2]
        if early_stop:
            print(f"Stopping training at epoch {epoch}")
            break
        if (time.time() - total_start_time) / 60 > 4200:
            print("Breaking early to get some result within 72 hours")
            break
        if args.save_to is not None:
            save_path = MODELS_HOME+"/"+args.save_to+".pth.tar"
            torch.save(dict([(name, module.state_dict()) for name, module in named_modules.items()]), save_path)
        with open(LOG_HOME+"/"+args.language+"_"+__file__+"_"+str(args.myID), "w") as out_file:
            print(" ".join([str(x) for x in dev_losses]), file=out_file)
            print(" ".join(sys.argv), file=out_file)
            print(str(args), file=out_file)
        # FIXME: Why is optimizer - Mention to dupoux
        # Momentum peut pas marcher si pas 0
        learning_rate *= args.lr_decay
        # FIXME: Pk cette ligne?
        optim = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.0)  # 0.02, 0.9
