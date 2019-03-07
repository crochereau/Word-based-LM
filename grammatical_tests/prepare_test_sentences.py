import spacy

from experiments import CHAR_VOCAB_PATH
from utils import generate_vocab_mappings, load_sentences, tokenizer

itos, stoi = generate_vocab_mappings(CHAR_VOCAB_PATH)
input_sentences = load_sentences("input_sentences/sentences.txt")
# input_tokens = tokenizer(input_sentences, stoi)
# print(len(input_tokens))
# print(input_tokens)

# TODO: download spacy for german, to shuffle nouns with their article

# TODO: create functions to shuffle noun phrases, to change article case, to combine both


nlp = spacy.load('de_core_news_sm')
doc = nlp(input_sentences)
for token in doc:
    print(token.text, token.pos_, token.dep_)

# to group noun phrases:
# for np in doc.noun_chunks:
    # np.text

def shuffle_noun_phrases(sentence):
    return shuffled_sentence

def change_case(word):
    return word_new_case