import argparse

import torch

from paths import MODELS_HOME
from utils import generate_vocab_mappings, load_WordNLM_model, pickle_dump
from model import WordNLM
from tests import gender_test, syntactic_test

BASE_DATASETS_PATH = "input_sentences/"
BASE_RESULTS_PATH = "results/"
CHAR_VOCAB_PATH = "vocabularies/german-wiki-word-vocab-50000.txt"

DATASETS_PATHS = {
    "masc_grammatical":f"{BASE_DATASETS_PATH}masc_grammatical_sentences.txt",
    "grammatical":f"{BASE_DATASETS_PATH}grammatical_sentences.txt",
    "masc_ungrammatical":f"{BASE_DATASETS_PATH}masc_ungrammatical_sentences.txt",
    "ungrammatical":f"{BASE_DATASETS_PATH}ungrammatical_sentences.txt"
                  }

RESULTS_PATHS = {
    "gender":"genders.txt",
    "masc_grammatical":"base_model_masc_gram_probs.txt",
    "grammatical":"base_model_gram_probs.txt",
    "masc_ungrammatical":"base_model_masc_ungram_probs.txt",
    "ungrammatical": "base_model_ungram_probs.txt"
                 }
TESTS = {
    "gender": gender_test,
    "syntax": syntactic_test,
         }


def get_args(*in_args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", dest="language", type=str, default="german")
    parser.add_argument("--load_from", dest="load_from", type=str, default="base_model")
    parser.add_argument("--word_embedding_size", type=int, default=200)
    parser.add_argument("--dataset", type=str, required=False)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--layer_num", type=int, default=2)
    parser.add_argument("--test", type=str)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    itos, stoi = generate_vocab_mappings(CHAR_VOCAB_PATH)
    print('len vocabulary:', len(stoi))
    model = WordNLM(args.word_embedding_size, len(itos), args.hidden_dim, args.layer_num)
    model.to(device)
    if args.load_from is not None:
        if args.load_from == "base_model":
            weight_path = MODELS_HOME+"/"+args.load_from+".pth.tar"
        else:
            weight_path = MODELS_HOME + args.load_from
        model = load_WordNLM_model(weight_path, model, device, args.load_from)
    else:
        assert False
    model.eval()

    if args.test == "gender":
        parameters = {"gender_model": model, "gender_device": device, "vocab_mapping": stoi}
    elif args.test == "syntax":
        path = DATASETS_PATHS[args.dataset]
        parameters = {"path": path, "syntactic_model": model, "syntactic_device": device, "vocab_mapping": stoi}
    elif args.test == "test2":
        passll

    result = TESTS[args.test](**parameters)
    print(result)

    if args.test == "gender":
        result_name = BASE_RESULTS_PATH + args.load_from + "_" + RESULTS_PATHS["gender"]
    elif args.test == "syntax":
        result_name = BASE_RESULTS_PATH + args.load_from + "_" + RESULTS_PATHS[args.dataset]

    pickle_dump(result, result_name)

    return result


if __name__ == "__main__":
    main()