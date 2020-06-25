import numpy as np
import tqdm
import math

from typing import Mapping
from utils import get_words_logprobs, load_sentences, gender_tokenizer, get_sentences_probs, tokenizer, encode_words


def gender_test(gender_model, gender_device, vocab_mapping: Mapping[str, int]):
    """
    Args:
        gender_model: model tested
        gender_device: computing device
        vocab_mapping: dictionary mapping words to unique integers

    Returns: confusion matrix containing the probabilities of predicting each gender

    """
    gender_probs = np.zeros(shape=(4, 3, 3))

    # load & tokenize stimuli
    modes = ("nothing", "adjective", "sehr + adjective", "sehr + extrem + adjective")
    for mode_idx, mode in tqdm.tqdm(enumerate(modes), total=len(modes)):
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
            numericalized_gender_sentences, gender_logprobs = get_words_logprobs(gender_tokens, gender_model, vocab_mapping,
                                                                                             gender_device)
            print("number of sentences:", len(numericalized_gender_sentences))
            print("shape of log probabilities prediction:", gender_logprobs.shape)

            # Compute sentence probabilities
            gender_results = get_sentences_probs(numericalized_gender_sentences, gender_logprobs)

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


def syntactic_test(path, syntactic_model, syntactic_device, vocab_mapping, batch_size: int = 72):
    """
    Args:
        syntactic_model: model tested
        vocab_mapping: dictionary mapping words to unique integers
        syntactic_device: computing device
        batch_size: batch size to use while computing logprobs

    Returns: list of log probabilities assigned to each sentence
    """

    # load & tokenize stimuli
    test_sentences = load_sentences(path)
    tokenized_sentences = tokenizer(test_sentences)
    encoded_tokens = encode_words(tokenized_sentences, vocab_mapping)
    print("number of sentences after encoding tokens:", len(encoded_tokens), len(encoded_tokens[-1]))


    num_steps = math.ceil(len(encoded_tokens) / batch_size)
    all_probs = []
    for i in tqdm.trange(num_steps, desc="Computing logprobs"):
        sent_tok_ids, logprobs = get_words_logprobs(encoded_tokens[i*batch_size:(i+1)*batch_size],
                                                           syntactic_model,
                                                                vocab_mapping, syntactic_device)
        all_probs.extend(get_sentences_probs(sent_tok_ids, logprobs))

    return all_probs


