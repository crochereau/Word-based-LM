"""
Replication of gender experiment: new pipeline
"""

import argparse
import logging

import torch

from paths import MODELS_HOME
from utils import load_sentences, gender_tokenizer, generate_vocab_mappings, load_WordNLM_model
from sentence_probabilities import compute_logprob, per_sentence_probabilities
from model import WordNLM

CHAR_VOCAB_PATH = "vocabularies/german-wiki-word-vocab-50000.txt"


def get_args(*in_args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", dest="language", type=str, default="german")
    parser.add_argument("--load-from", dest="load_from", type=str, default="wiki-german-nospaces-bptt-words-966024846")
    parser.add_argument("--char_embedding_size", type=int, default=200)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--layer_num", type=int, default=2)
    parser.add_argument("--test_to_perform", type=str, required=False)
    args = parser.parse_args()
    print(args)
    return args


def test1(params):
    do_stuff


TESTS = {
    "syntax": test1,
}


def main():
    args = get_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    itos, stoi = generate_vocab_mappings(CHAR_VOCAB_PATH)
    model = WordNLM(args.char_embedding_size, len(itos), args.hidden_dim, args.layer_num)
    model.to(device)
    if args.load_from is not None:
        weight_path = MODELS_HOME+"/"+args.load_from+".pth.tar"
        model = load_WordNLM_model(weight_path, model, device)
    else:
        assert False
    model.eval()

    """
    if args.test_to_perform == "syntax":
        resultat = TESTS[args.test_to_perform](parameteres)
    elif args.test_to_perform == "syntax":
        pass
    """

    # Loading stimuli. They contain no OOV words.
    gender_path = "stimuli/german-gender-Gender=Fem-nothing-noOOVs.txt"
    gender_test_sentences = load_sentences(gender_path)

    gender_tokens = gender_tokenizer(gender_test_sentences, stoi)
    print(gender_tokens)
    print(len(gender_tokens))

    # We check whether all sentences have the correct number of tokens.
    for idx, value in enumerate(gender_tokens):
        assert len(value) == 4, f"{idx}, {len(value)}"

    numericalized_gender_sentences, gender_logprobs, gender_losses = compute_logprob(gender_tokens, model, stoi, device)

    print("number of sentences:", len(numericalized_gender_sentences))
    print("shape of log probabilities prediction:", gender_logprobs.shape)
    print("size of losses: ", len(gender_losses))

    gender_results = per_sentence_probabilities(numericalized_gender_sentences, gender_logprobs)
    print(gender_results)


    #for idx, value in enumerate(gender_results):
    #print(idx, value)
    #if value == 1.0:
    #del gender_results[idx]


    # Probability of each gender

    print(len(gender_results)/3)
    der_list = gender_results[0::3]
    die_list = gender_results[1::3]
    das_list = gender_results[2::3]
    print(len(der_list), len(die_list), len(das_list))

    der_prob = sum(der_list)/sum(gender_results)
    die_prob = sum(die_list)/sum(gender_results)
    das_prob = sum(das_list)/sum(gender_results)
    print("der probability:", der_prob)
    print("die probability:", die_prob)
    print("das probability:", das_prob)

    print(sum(gender_results))
    print(sum(die_list))
    print(sum(das_list))
    print(sum(der_list))

    """
    # ### Checking what the max log probabilities are and to which words they correspond

    # to finish
    max_probs_values, max_probs_idx = torch.max(log_prediction, 2)
    print(max_probs_values.size(), max_probs_idx.size())
    print(max_probs_values, max_probs_idx)

    """

if __name__ == "__main__":
    main()