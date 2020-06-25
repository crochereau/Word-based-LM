import argparse

import torch

from paths import MODELS_HOME, DATASETS_PATHS, RESULTS_PATHS, TESTS, BASE_RESULTS_PATH, CHAR_VOCAB_PATH
from utils import generate_vocab_mappings, load_WordNLM_model, pickle_dump
from model import WordNLM


def get_args(*in_args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_from", dest="load_from", type=str, default="LSTM", help="TBD")
    parser.add_argument("--word_embedding_size", type=int, default=200)
    parser.add_argument("--dataset", type=str, required=False, choices=list(DATASETS_PATHS.keys()),
                        help="Path to the text file containing sentences")
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--layer_num", type=int, default=2)
    parser.add_argument("--test", type=str, choices=list(TESTS.keys()))
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
        if args.load_from == "LSTM":
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
        pass

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