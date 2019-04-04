from itertools import permutations, combinations_with_replacement

import spacy

from utils import generate_german_dict, load_sentences


BASE_PATH = "input_sentences/"
GRAMMATICAL_END = "masc_grammatical_sentences.txt"
UNGRAMMATICAL_END = "masc_ungrammatical_sentences.txt"
TEST_SENTENCES_PATH = BASE_PATH + "sentences.txt"


def spacy_parser(nlp, sentences):
    doc = nlp(sentences)
    return doc


def sentence_segmenter(doc):
    sentences = []
    for sentence in doc.sents:
        sentences.append(sentence.text)
    return sentences


def spacy_tokenizer(nlp, number_sentences, sentence_list):

    doc_list = []
    for sentence in range(number_sentences):
        doc_list.append(nlp(sentence_list[sentence]))

    tokens = [[] for _ in range(number_sentences)]
    for sentence_idx, sentence in enumerate(doc_list):
        for token in sentence:
            tokens[sentence_idx].append(token.text)

    return tokens


def select_args_w_nb(args_nb, nb_sentences, tokens):

    verb_args = [[[] for _ in range(args_nb)] for _ in range(nb_sentences)]
    for sentence_idx, sentence in enumerate(tokens):
        verb_args[sentence_idx][0] = (tokens[sentence_idx][4:7])
        verb_args[sentence_idx][1] = (tokens[sentence_idx][7:10])
        verb_args[sentence_idx][2] = (tokens[sentence_idx][10:13])

    return verb_args


# Get cases functions

def get_nom(german_dict, group_args):
    # get nominative of verb argument and corresponding article
    new_nom = []
    article = group_args[0]
    word = group_args[1]
    number = group_args[2]

    if number == "sg":
        new_nom.append(article + " " + word)

    elif number == "pl":
        pl_article = ""
        if article in {"der", "die", "das"}:
            pl_article = "die"
        if article in {"dieser", "diese", "dieses"}:
            pl_article = "diese"
        elif article in {"sein", "seine"}:
            pl_article = "seine"
        elif article in {"ihr", "ihre"}:
            pl_article = "ihre"
        elif article in {"ein", "eine"}:
            pl_article = "einige"

        pl_word = german_dict[word][4]
        new_nom.append(pl_article + " " + pl_word)

    return new_nom


def get_acc(german_dict, group_args):
    # get accusative of verb argument and corresponding article
    new_acc = []
    article = group_args[0]
    word = group_args[1]
    gender = german_dict[word][0]
    number = group_args[2]

    if number == "sg":
        sg_article = ""
        if gender == "m":
            if article == "der":
                sg_article = "den"
            elif article == "dieser":
                sg_article = "diesen"
            elif article == "sein":
                sg_article = "seinen"
            elif article == "ihr":
                sg_article = "ihren"
            else:
                sg_article = "einen"
        else:
            sg_article = article

        sg_word = german_dict[word][1]

        new_acc.append(sg_article + " " + sg_word)

    elif number == "pl":
        pl_article = ""
        if article in {"der", "die", "das"}:
            pl_article = "die"
        elif article in {"dieser", "diese", "dieses"}:
            pl_article = "diese"
        elif article in {"sein", "seine"}:
            pl_article = "seine"
        elif article in {"ihr", "ihre"}:
            pl_article = "ihre"
        elif article in {"ein", "eine"}:
            pl_article = "einige"

        pl_word = german_dict[word][5]
        new_acc.append(pl_article + " " + pl_word)

    return new_acc


def get_dat(german_dict, group_args):
    # get dative of verb argument and corresponding article
    new_dat = []
    article = group_args[0]
    word = group_args[1]
    gender = german_dict[word][0]
    number = group_args[2]

    if number == "sg":
        sg_article = ""
        if gender in {"m", "n"}:
            if article in {"der", "das"}:
                sg_article = "dem"
            elif article in {"dieser", "dieses"}:
                sg_article = "diesem"
            elif article == "sein":
                sg_article = "seinem"
            elif article == "ihr":
                sg_article = "ihrem"
            elif article == "ein":
                sg_article = "einem"
        elif gender == "f":
            if article == "die":
                sg_article = "der"
            elif article == "diese":
                sg_article = "dieser"
            elif article == "seine":
                sg_article = "seiner"
            elif article == "ihre":
                sg_article = "ihrer"
            elif article == "eine":
                sg_article = "einer"

        sg_word = german_dict[word][2]
        new_dat.append(sg_article + " " + sg_word)

    elif number == "pl":
        pl_article = ""
        if article in {"der", "die", "das"}:
            pl_article = "den"
        elif article in {"dieser", "diese", "dieses"}:
            pl_article = "diesen"
        elif article in {"sein", "seine"}:
            pl_article = "seinen"
        elif article in {"ihr", "ihre"}:
            pl_article = "ihren"
        elif article in {"ein", "eine"}:
            pl_article = "einigen"

        pl_word = german_dict[word][6]
        new_dat.append(pl_article + " " + pl_word)

    return new_dat


def get_grammatical_permutations(args_nb, german_dict, nb_sentences, verb_args):

    all_cases = [[[] for _ in range(args_nb)] for _ in range(nb_sentences)]
    for sentence_idx, sentence in enumerate(verb_args):
        for group_args_idx, group_args in enumerate(verb_args[sentence_idx]):
            nom = get_nom(german_dict, verb_args[sentence_idx][group_args_idx])
            acc = get_acc(german_dict, verb_args[sentence_idx][group_args_idx])
            dat = get_dat(german_dict, verb_args[sentence_idx][group_args_idx])
            all_cases[sentence_idx][group_args_idx].append(nom)
            all_cases[sentence_idx][group_args_idx].append(acc)
            all_cases[sentence_idx][group_args_idx].append(dat)

    idxs = [0, 1, 2]
    idxs_perm = list(permutations(idxs))
    grammatical_combinations = [[[] for _ in range(6)] for _ in range(nb_sentences)]
    for stc_idx, stc in enumerate(all_cases):
        for idx in range(6):
            for sub_idx in range(3):
                tmp_perm_idx = idxs_perm[idx][sub_idx]
                grammatical_combinations[stc_idx][idx].append(all_cases[stc_idx][sub_idx][tmp_perm_idx])

    grammatical_case_permutations = [[] for _ in range(nb_sentences)]
    for stc_idx, stc in enumerate(grammatical_combinations):
        for idx in range(6):
            tmp_perm = list(permutations(grammatical_combinations[stc_idx][idx]))
            grammatical_case_permutations[stc_idx].append(tmp_perm)

    return all_cases, grammatical_case_permutations


def get_ungrammatical_permutations(args_nb, case_permutations, nb_sentences):

    """
    # O : Nominative, 1 : Accusative, 2: Dative
    case_idxs = [0, 1, 2]
    nb_case_repeats = list(combinations_with_replacement(case_idxs, args_nb))

    # remove grammatical combination & one case combinations
    for idx, value in enumerate(nb_case_repeats):
        if value in {(0, 1, 2), (0, 0, 0), (1, 1, 1), (2, 2, 2)}:
            nb_case_repeats.remove(value)

    # generate cases permutations
    idxs_permutations = [[] for _ in range(len(nb_case_repeats))]
    for idx, value in enumerate(nb_case_repeats):
        tmp_idx_perm = list(permutations(value))
        idxs_permutations[idx] = tmp_idx_perm
    print(idxs_permutations)

    # remove duplicates
    ungrammatical_idxs = []
    for idx, value in enumerate(idxs_permutations):
        for i in set(idxs_permutations[idx]):
            ungrammatical_idxs.append(i)
    """
    # O : Nominative, 1 : Accusative, 2: Dative
    ungrammatical_idxs = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 2, 0), (2, 0, 0), (0, 0, 2), (0, 1, 1), (1, 1, 0),
                          (1, 0, 1),(1, 2, 1), (2, 1, 1), (1, 1, 2),(2, 2, 0), (2, 0, 2), (0, 2, 2),  (1, 2, 2),
                          (2, 2, 1), (2, 1, 2)]

    # generate verb arguments ungrammatical combinations
    ungrammatical_combinations = [[[] for _ in range(len(ungrammatical_idxs))] for _ in range(nb_sentences)]
    for stc_idx, stc in enumerate(case_permutations):
        for case_violation_idx, case_violation in enumerate(ungrammatical_idxs):
            for sub_idx in range(args_nb):
                perm_idx = ungrammatical_idxs[case_violation_idx][sub_idx]
                ungrammatical_combinations[stc_idx][case_violation_idx].append(
                    case_permutations[stc_idx][sub_idx][perm_idx])

    # permute verb arguments positions
    ungrammatical_case_permutations = [[] for _ in range(nb_sentences)]
    for stc_idx, stc in enumerate(ungrammatical_combinations):
        for case_violation_idx, case_violation in enumerate(ungrammatical_combinations[stc_idx]):
            tmp_perm = list(permutations(ungrammatical_combinations[stc_idx][case_violation_idx]))
            ungrammatical_case_permutations[stc_idx].append(tmp_perm)

    return ungrammatical_case_permutations


def generate_dataset(case_permutations, end_path, nb_sentences, tokens):

    dataset = ""
    counter = 0
    for stc_idx, stc in enumerate(tokens):
        for case_perm_idx, case_perm in enumerate(case_permutations[stc_idx]):
            for case_idx, case in enumerate(case_permutations[stc_idx][case_perm_idx]):
                principal_clause = f"{tokens[stc_idx][0]} {tokens[stc_idx][1]}, {tokens[stc_idx][3]}"

                verb_arg = ""
                for vb_arg_idx, vb_arg in enumerate(case_permutations[stc_idx][case_perm_idx][case_idx]):
                    verb_arg += f"{vb_arg[0]} "

                verb_subordinate = f"{tokens[stc_idx][-3]} {tokens[stc_idx][-2]}"
                dataset += f"{principal_clause} {verb_arg}{verb_subordinate}.\n"

                counter += 1
    counter = counter/nb_sentences
    print("number of generated sentences per original sentence:", counter)

    output_file = open(BASE_PATH + end_path, 'w')
    output_file.write(dataset)
    output_file.close()
    
    return dataset


def main(verb_args_number, input_path):

    sentences_w_number = load_sentences(input_path)
    sentences_wo_number = sentences_w_number.replace('sg ', '').replace('pl ', '')

    de_nlp = spacy.load('de_core_news_sm')
    doc_w_number = spacy_parser(de_nlp, sentences_w_number)
    doc_wo_number = spacy_parser(de_nlp, sentences_wo_number)

    de_dict = generate_german_dict("vocabularies/de_dict.csv")

    list_sentences_w_number = sentence_segmenter(doc_w_number)
    list_sentences_wo_number = sentence_segmenter(doc_wo_number)
    assert len(list_sentences_w_number) == len(list_sentences_wo_number)
    sentences_nb = len(list_sentences_wo_number)
    print("number of original sentences:", sentences_nb)

    tokens_w_nb = spacy_tokenizer(de_nlp, sentences_nb, list_sentences_w_number)
    tokens_wo_nb = spacy_tokenizer(de_nlp, sentences_nb, list_sentences_wo_number)

    args_w_nb = select_args_w_nb(verb_args_number, sentences_nb, tokens_w_nb)
    print(args_w_nb)

    all_cases, grammatical_permutations = get_grammatical_permutations(verb_args_number,
                                                        de_dict, sentences_nb, args_w_nb)
    grammatical_dataset = generate_dataset(grammatical_permutations,GRAMMATICAL_END, sentences_nb, tokens_wo_nb)

    ungrammatical_permutations = get_ungrammatical_permutations(verb_args_number, all_cases, sentences_nb)
    ungrammatical_dataset = generate_dataset(ungrammatical_permutations,UNGRAMMATICAL_END, sentences_nb, tokens_wo_nb)

    return grammatical_dataset, ungrammatical_dataset



if __name__ == "__main__":
    main(3, "input_sentences/masc_dataset.txt")
