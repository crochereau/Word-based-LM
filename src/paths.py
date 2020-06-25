from tests import gender_test, syntactic_test

BASE_DATASETS_PATH = "data/"
BASE_RESULTS_PATH = "results/"
CHAR_VOCAB_PATH = "vocabularies/german-wiki-word-vocab-50000.txt"

CHAR_VOCAB_HOME = "/vocabularies/"  # this is part of the repo

base_path = "/Users/charlotterochereau/Documents/Stage_ENS/"

MODELS_HOME = base_path + "checkpoints/"  # for storing neural network parameters
LOG_HOME = base_path + "checkpoints/"  # for logging validation losses
FIGURES_HOME = base_path + "figures/"  # for storing visualizations
CORPORA_HOME = base_path + "corpora/"  # location of Wikipedia corpora

DATASETS_PATHS = {
    "grammatical": f"{BASE_DATASETS_PATH}grammatical_sentences.txt",
    "ungrammatical": f"{BASE_DATASETS_PATH}ungrammatical_sentences.txt",
    "data2020": f"{BASE_DATASETS_PATH}dataset2020.txt",
    "data2019": f"{BASE_DATASETS_PATH}dataset2019.txt",
    "testouille": f"{BASE_DATASETS_PATH}testouille.txt"
                  }
RESULTS_PATHS = {
    "gender": "genders.pkl",
    "grammatical": "gram_probs.pkl",
    "ungrammatical": "ungram_probs.pkl",
    "data2020": "dataset2020_probs.pkl",
    "data2019": "dataset2019_probs.pkl",
    "testouille": "testouille_probs.pkl"
                 }
TESTS = {
    "gender": gender_test,
    "syntax": syntactic_test,
         }
