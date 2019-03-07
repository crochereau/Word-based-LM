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

def gender_test(mode, gender_model, vocab_mapping, gender_device):
    """

    Args:
        mode: number of intervening elements between article and noun

    Returns: # FIXME

    """
    gender_probs = [[0,0,0] for _ in range(3)]
    for gender_idx, gender in enumerate(["Gender=" + gender for gender in ["Masc", "Fem", "Neut"]]):
        # Loading stimuli
        gender_path = f"stimuli/german-gender-{gender}-{mode}-noOOVs.txt"
        # "stimuli/german-gender-Gender=Fem-nothing-noOOVs.txt"
        gender_test_sentences = load_sentences(gender_path)
        gender_tokens = gender_tokenizer(gender_test_sentences, vocab_mapping)
        # print(gender_tokens)
        print(f"gender: {gender}")
        print("number of stimuli:", len(gender_tokens))
        print("number of tokens in stimulus: ", len(gender_tokens[0]))

        # We check whether all sentences have the correct number of tokens.
        # for idx, value in enumerate(gender_tokens):
            # assert len(value) == 4, f"{idx}, {len(value)}"

        numericalized_gender_sentences, gender_logprobs, gender_losses = compute_logprob(gender_tokens, gender_model, vocab_mapping,
                                                                                         gender_device)

        print("number of sentences:", len(numericalized_gender_sentences))
        print("shape of log probabilities prediction:", gender_logprobs.shape)
        print("size of losses: ", len(gender_losses))

        gender_results = per_sentence_probabilities(numericalized_gender_sentences, gender_logprobs)

        # print(len(gender_results) / 3)
        der_list = gender_results[0::3]
        die_list = gender_results[1::3]
        das_list = gender_results[2::3]
        print(len(der_list), len(die_list), len(das_list))

        der_prob = sum(der_list) / sum(gender_results)
        die_prob = sum(die_list) / sum(gender_results)
        das_prob = sum(das_list) / sum(gender_results)
        # print("der probability:", der_prob) ; print("die probability:", die_prob) ; print("das probability:", das_prob)
        gender_probs[gender_idx][0] = der_prob
        gender_probs[gender_idx][1] = die_prob
        gender_probs[gender_idx][2] = das_prob

    return gender_probs


TESTS = {
    "gender": gender_test,
}

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

    result = TESTS[args.test_to_perform]("nothing", model, stoi, device)
    # result = gender_test("nothing", model, stoi, device)
    print(result)

    """
    if args.test_to_perform == "gender":
        resultat = TESTS[args.test_to_perform](parameteres)
    elif args.test_to_perform == "gender":
        pass
    """

    #for idx, value in enumerate(gender_results):
    #print(idx, value)
    #if value == 1.0:
    #del gender_results[idx]

    # Probability of each gender
    """
    print(sum(gender_results))
    print(sum(die_list))
    print(sum(das_list))
    print(sum(der_list))
    """

    return result
    """
    # ### Checking what the max log probabilities are and to which words they correspond

    # to finish
    max_probs_values, max_probs_idx = torch.max(log_prediction, 2)
    print(max_probs_values.size(), max_probs_idx.size())
    print(max_probs_values, max_probs_idx)

    """

if __name__ == "__main__":
    main()