import numpy as np

from sentence_probabilities import compute_logprob, per_sentence_probabilities
from utils import load_sentences, gender_tokenizer


def gender_test(gender_model, vocab_mapping, gender_device):
    """
    Args:
        mode: intervening elements between article and noun: {"nothing, "adjective", "sehr + adjective",
                                                                                "sehr + extrem + adjective"}
        gender_model: model tested
        vocab_mapping: dictionary mapping words to unique integers
        gender_device: computing device
        gender_device: computing device

    Returns: confusion matrix containing the probabilities of predicting each gender

    """
    gender_probs = np.zeros(shape=(4,3,3))

    # load & tokenize stimuli
    for mode_idx, mode in enumerate(("nothing", "adjective", "sehr + adjective", "sehr + extrem + adjective")):
        for gender_idx, gender in enumerate(["Gender=" + gender for gender in ["Masc", "Fem", "Neut"]]):
            gender_path = f"stimuli/german-gender-{gender}-{mode}-noOOVs.txt"
            gender_test_sentences = load_sentences(gender_path)

            # Count number of intervening elements
            if mode == "nothing":
                intervening_elements = 2
            elif mode == "adjective":
                intervening_elements = 3
            elif mode == "sehr + adjective":
                intervening_elements = 4
            else:
                intervening_elements = 5

            gender_tokens = gender_tokenizer(intervening_elements, gender_test_sentences, vocab_mapping)

            print(f"mode: {mode}")
            print(f"gender: {gender}")
            print("number of stimuli:", len(gender_tokens))
            print(gender_tokens[0])

            # Check whether all sentences contain the correct number of tokens.
            for idx, value in enumerate(gender_tokens):
                if mode == "nothing":
                    number_tokens = 4
                elif mode == "adjective":
                    number_tokens = 5
                elif mode == "sehr + adjective":
                    number_tokens = 6
                else:
                    number_tokens = 7
                assert len(value) == number_tokens, f"{idx}, {len(value)}"

            # Compute word log probabilities
            numericalized_gender_sentences, gender_logprobs, gender_losses = compute_logprob(gender_tokens, gender_model, vocab_mapping,
                                                                                             gender_device)
            print("number of sentences:", len(numericalized_gender_sentences))
            print("shape of log probabilities prediction:", gender_logprobs.shape)
            # print("size of losses: ", len(gender_losses))

            # Compute sentence probabilities
            gender_results = per_sentence_probabilities(numericalized_gender_sentences, gender_logprobs)

            argmax_list = []
            for idx in range(0, len(gender_results), 3):
                gender_argmax = np.argmax(gender_results[idx:idx + 3])
                argmax_list.append(gender_argmax)

            der_prob = argmax_list.count(0) / (len(gender_results)/3)
            die_prob = argmax_list.count(1) / (len(gender_results)/3)
            das_prob = argmax_list.count(2) / (len(gender_results)/3)

            gender_probs[mode_idx][gender_idx][0] = der_prob
            gender_probs[mode_idx][gender_idx][1] = die_prob
            gender_probs[mode_idx][gender_idx][2] = das_prob

    return gender_probs
