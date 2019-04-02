"""
Replication of gender experiment: new pipeline
"""

import argparse
import pickle

import torch

from paths import MODELS_HOME
from utils import generate_vocab_mappings, load_WordNLM_model, pickle_dump
from model import WordNLM
from tests import gender_test, syntactic_test

CHAR_VOCAB_PATH = "vocabularies/german-wiki-word-vocab-50000.txt"

TEST_PATHS = {
    "3_args_grammatical": "input_sentences/grammatical_sentences.txt",
    "3_args_ungrammatical": "input_sentences/ungrammatical_sentences.txt"
              }

RESULTS_PATHS = {
    "gender":"results/genders.txt",
    "3_args_grammatical":"results/3_gram_probs.txt",
    "3_args_ungrammatical":"results/3_ungram_probs.txt"
                 }
TESTS = {
    "gender": gender_test,
    "syntax": syntactic_test
         }


def get_args(*in_args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", dest="language", type=str, default="german")
    parser.add_argument("--load-from", dest="load_from", type=str, default="wiki-german-nospaces-bptt-words-966024846")
    parser.add_argument("--char_embedding_size", type=int, default=200)
    parser.add_argument("--grammaticality", type=str, required=False)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--layer_num", type=int, default=2)
    parser.add_argument("--test_to_perform", type=str, required=False)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    itos, stoi = generate_vocab_mappings(CHAR_VOCAB_PATH)
    print('len vocabulary:', len(stoi))
    model = WordNLM(args.char_embedding_size, len(itos), args.hidden_dim, args.layer_num)
    model.to(device)
    if args.load_from is not None:
        weight_path = MODELS_HOME+"/"+args.load_from+".pth.tar"
        model = load_WordNLM_model(weight_path, model, device)
    else:
        assert False
    model.eval()

    if args.test_to_perform == "gender":
        parameters = {"gender_model": model, "gender_device": device, "vocab_mapping": stoi}
    elif args.test_to_perform == "syntax":
        path = TEST_PATHS[args.grammaticality]
        parameters = {"path": path, "syntactic_model": model, "syntactic_device": device, "vocab_mapping": stoi}
    elif args.test_to_perform == "test2":
        pass

    result = TESTS[args.test_to_perform](**parameters)
    print(result)

    if args.test_to_perform == "gender":
        result_name = RESULTS_PATHS["gender"]
    elif args.test_to_perform == "syntax":
        result_name = RESULTS_PATHS[args.grammaticality]

    pickle_dump(result, result_name)

    return result


if __name__ == "__main__":
    main()